#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> Option<String> {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(&prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            reasoning_effort.clone(),
            reasoning.clone(),
            None,
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return Some(strip_invisible_chars(&content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(&content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        reasoning_effort,
        reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    /// Wall-clock time of the h2 LLM call, in milliseconds. `Some(ms)` only
    /// when `handy2::post_process` actually attempted an LLM call for this
    /// dictation (h2 enabled and routing didn't short-circuit before the
    /// call); `None` when no LLM call happened at all — h2 disabled, stock
    /// post-processing used instead, or post-processing skipped entirely —
    /// so the summary log can print an honest "n/a" instead of a fake 0ms.
    pub llm_ms: Option<u64>,
}

/// One-hotkey UX: does this dictation's text carry a spoken h2 route trigger
/// (e.g. "Polish command, ...")? Pure and cheap — safe to call on every
/// transcription regardless of which binding fired. `false` whenever h2 is
/// disabled, so a spoken trigger phrase is inert text when the feature is off.
fn h2_spoken_hotword(settings: &AppSettings, final_text: &str) -> bool {
    settings.h2_enabled
        && crate::handy2::routing::route(final_text, &settings.h2_routes)
            .route
            .trigger
            .is_some()
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    foreground_app: Option<String>,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    process_transcription_output_with_settings(
        &settings,
        transcription,
        post_process,
        foreground_app,
    )
    .await
}

/// Core of [`process_transcription_output`], taking already-loaded
/// [`AppSettings`] instead of an `AppHandle`. Split out so the gate logic
/// (including the Task E5 verbatim-app override) is directly unit-testable
/// with `#[tokio::test]` — `get_settings(app: &AppHandle)` unconditionally
/// calls `app.store(...).expect(...)`, which panics without a live
/// `tauri_plugin_store` registration, and this codebase has no working
/// pattern for constructing a real `AppHandle` in a test (see the
/// `TranscriptionManager`/`AudioRecordingManager` note in
/// `managers::transcription`'s streaming tests for prior art reaching the
/// same conclusion). Mirrors the existing `h2_spoken_hotword(settings:
/// &AppSettings, ...)` extraction already used in this file.
async fn process_transcription_output_with_settings(
    settings: &AppSettings,
    transcription: &str,
    post_process: bool,
    foreground_app: Option<String>,
) -> ProcessedTranscription {
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;
    let mut llm_ms: Option<u64> = None;

    if let Some(converted_text) = maybe_convert_chinese_variant(settings, transcription).await {
        final_text = converted_text;
    }

    // App-aware verbatim list (Task E5): dictating into a terminal or coding
    // agent almost always wants exact words, not an LLM rewrite. This beats
    // EVEN the dedicated force-post-process binding (`post_process == true`)
    // and the spoken hotword trigger below — that's the point of the list;
    // remove an app from settings to re-enable LLM routing for it.
    let verbatim_app = foreground_app.as_deref().filter(|name| {
        crate::handy2::app_context::is_verbatim_app(name, &settings.h2_verbatim_apps)
    });

    // One-hotkey UX: the raw binding also routes through h2 when the user
    // SPOKE a route trigger ("Polish command, ..."). The dedicated
    // post-process binding still forces it unconditionally. The verbatim-app
    // check above overrides both — folded into this same condition (rather
    // than a separate leading `if is_verbatim { skip } else if ...`) so the
    // `else if final_text != transcription` bookkeeping arm below stays
    // shared and still runs for verbatim apps (e.g. Chinese-variant
    // conversion, which is deterministic and NOT an LLM call, must still be
    // recorded even when LLM routing itself is suppressed).
    let spoken_hotword = h2_spoken_hotword(settings, &final_text);
    let would_have_routed = post_process || spoken_hotword;
    if let Some(name) = verbatim_app {
        if would_have_routed {
            // Only log when suppression actually changed the outcome — a
            // verbatim app that wasn't going to route anyway (plain text,
            // primary binding, no hotword) isn't worth a log line.
            info!("h2: verbatim app '{}', skipping LLM", name);
        }
    }
    if verbatim_app.is_none() && would_have_routed {
        // Handy 2.0: when enabled, route through our hotword + memory pipeline;
        // otherwise use Handy's stock post-processing unchanged.
        let processed = if settings.h2_enabled {
            let (text, ms) = crate::handy2::post_process(settings, &final_text).await;
            llm_ms = ms;
            text
        } else {
            post_process_transcription(settings, &final_text).await
        };
        if let Some(processed_text) = processed {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
        llm_ms,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // App-aware verbatim list (Task E5): capture the foreground app RIGHT
        // NOW, before anything below (tray icon, recording overlay) can
        // possibly perturb window focus. "Focus at start == paste target
        // under PTT," so this is the only reliable moment to read it — by
        // the time process_transcription_output runs (in stop()'s spawned
        // task, possibly seconds later), the foreground window could be
        // Handy's own overlay or wherever the user alt-tabbed to. Stored
        // keyed by binding_id and consumed exactly once in stop().
        crate::handy2::app_context::remember(
            binding_id,
            crate::handy2::app_context::foreground_process_name(),
        );

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);

            // Segment-streaming: transcribe audio in the background WHILE the
            // user keeps talking, so most of the transcript already exists
            // by the time they release the key. No-op (falls back to the
            // normal single-shot transcribe() at stop time) if disabled.
            if settings.streaming_enabled {
                tm.begin_streaming(app.clone(), binding_id.clone());
            }
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            let dictation_start = Instant::now();
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save.
                    //
                    // finalize_streaming() transcribes only the small leftover
                    // tail (everything already covered by prior segments was
                    // transcribed in the background while the user was still
                    // talking) — that's the whole point of segment-streaming.
                    // It falls back to the normal single-shot transcribe()
                    // internally whenever no session exists for this binding
                    // (streaming disabled, or begin_streaming didn't start one),
                    // so this call site doesn't need to re-check the setting.
                    let transcription_time = Instant::now();
                    let transcription_result = tm.finalize_streaming(&binding_id, samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            // Consume the foreground app captured at recording
                            // start (Task E5) for this binding. take() removes
                            // the entry so it's used exactly once per dictation.
                            let foreground_app = crate::handy2::app_context::take(&binding_id);
                            let processed = process_transcription_output(
                                &ah,
                                &transcription,
                                post_process,
                                foreground_app,
                            )
                            .await;

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let llm_ms = processed.llm_ms;
                                let final_text = processed.final_text;
                                let stt_ms = transcription_time.elapsed().as_millis() as u64;
                                ah.run_on_main_thread(move || {
                                    match utils::paste(final_text, ah_clone.clone()) {
                                        Ok(()) => {
                                            let paste_ms = paste_time.elapsed().as_millis() as u64;
                                            let total_ms = dictation_start.elapsed().as_millis() as u64;
                                            debug!(
                                                "Text pasted successfully in {:?}",
                                                paste_time.elapsed()
                                            );
                                            // Log: dictation: stop->paste <total>ms (stt <stt_ms>ms, llm <llm_ms>ms, paste <paste_ms>ms)
                                            // llm_ms is "n/a" when no LLM call was attempted (h2 disabled, or
                                            // routing short-circuited before the call) rather than a fake 0ms.
                                            let llm_ms_str = match llm_ms {
                                                Some(ms) => format!("{}ms", ms),
                                                None => "n/a".to_string(),
                                            };
                                            info!("dictation: stop->paste {}ms (stt {}ms, llm {}, paste {}ms)", total_ms, stt_ms, llm_ms_str, paste_ms);
                                        },
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            debug!("Global Shortcut Transcription error: {}", err);
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!("TranscribeAction::stop initiated async transcription task");
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod h2_gate_tests {
    //! One-hotkey UX: covers the `h2_spoken_hotword` predicate that feeds
    //! `process_transcription_output`'s post-process gate
    //! (`verbatim_app.is_none() && (post_process || spoken_hotword)`).
    //! Doesn't re-test `routing::route` itself (see `handy2::routing::tests`)
    //! or `is_verbatim_app` itself (see `handy2::app_context::tests`) — only
    //! that these predicates are wired to it correctly and respect the
    //! `h2_enabled` kill switch and the verbatim-app override.
    use super::h2_spoken_hotword;
    use crate::settings::get_default_settings;

    #[test]
    fn no_trigger_phrase_is_false_even_with_h2_enabled() {
        let mut settings = get_default_settings();
        settings.h2_enabled = true;
        assert!(!h2_spoken_hotword(
            &settings,
            "um so the panel needs three new appraisers"
        ));
    }

    #[test]
    fn trigger_phrase_is_true_when_h2_enabled() {
        let mut settings = get_default_settings();
        settings.h2_enabled = true;
        assert!(h2_spoken_hotword(
            &settings,
            "Polish command, tell the credit team the appraisal is approved"
        ));
    }

    #[test]
    fn trigger_phrase_is_false_when_h2_disabled() {
        // The feature kill switch wins even if the text would otherwise match
        // a route trigger — a disabled h2 must never let the primary/raw
        // binding silently start calling an LLM.
        let mut settings = get_default_settings();
        settings.h2_enabled = false;
        assert!(!h2_spoken_hotword(
            &settings,
            "Polish command, tell the credit team the appraisal is approved"
        ));
    }

    #[test]
    fn hotword_mid_sentence_is_content_not_a_trigger() {
        let mut settings = get_default_settings();
        settings.h2_enabled = true;
        assert!(!h2_spoken_hotword(
            &settings,
            "I told them the Polish command thing was a feature"
        ));
    }

    /// The actual gate expression as it appears in
    /// `process_transcription_output`: `post_process || spoken_hotword`.
    /// Exercises all four combinations so a future edit to that composition
    /// can't silently invert the dedicated binding's unconditional force or
    /// fail to extend it to the primary binding.
    #[test]
    fn gate_boolean_composition_covers_all_four_combinations() {
        let mut h2_on = get_default_settings();
        h2_on.h2_enabled = true;
        let plain = "just a plain sentence";
        let triggered = "Polish command, tell the credit team the appraisal is approved";

        // (dedicated binding, spoken hotword) -> gate result
        let cases = [
            (false, h2_spoken_hotword(&h2_on, plain), false), // primary + plain -> no LLM (unchanged)
            (false, h2_spoken_hotword(&h2_on, triggered), true), // primary + hotword -> LLM (new)
            (true, h2_spoken_hotword(&h2_on, plain), true), // dedicated + plain -> LLM (unchanged)
            (true, h2_spoken_hotword(&h2_on, triggered), true), // dedicated + hotword -> LLM (unchanged)
        ];

        for (post_process, spoken_hotword, expected_gate) in cases {
            assert_eq!(
                post_process || spoken_hotword,
                expected_gate,
                "post_process={post_process} spoken_hotword={spoken_hotword}"
            );
        }
    }

    /// Task E5: the full three-variable gate expression as it appears in
    /// `process_transcription_output`:
    /// `verbatim_app.is_none() && (post_process || spoken_hotword)`.
    /// Exercises all eight combinations so a future edit can't silently let
    /// the verbatim-app override lose to the dedicated binding's force, or
    /// fail to suppress the spoken-hotword path — the whole point of the
    /// list is that it beats BOTH.
    #[test]
    fn verbatim_app_overrides_both_post_process_and_spoken_hotword() {
        let is_verbatim = [false, true];
        let post_process = [false, true];
        let spoken_hotword = [false, true];

        for verbatim in is_verbatim {
            for pp in post_process {
                for hotword in spoken_hotword {
                    let gate = !verbatim && (pp || hotword);
                    let expected = if verbatim {
                        false // verbatim app always wins, no matter what else is true
                    } else {
                        pp || hotword
                    };
                    assert_eq!(
                        gate, expected,
                        "verbatim={verbatim} post_process={pp} spoken_hotword={hotword}"
                    );
                }
            }
        }
    }

    /// Task E5 end-to-end wiring proof: calls the real gate function
    /// (`process_transcription_output_with_settings`, the `AppHandle`-free
    /// core of `process_transcription_output` — see that function's doc
    /// comment for why this split exists) with the DEDICATED force-post-process
    /// binding (`post_process = true`) AND a foreground app that's on the
    /// default verbatim list. A unit test of `is_verbatim_app` alone proves
    /// only the string-matching predicate; it does NOT prove this predicate
    /// is actually wired into the gate that guards the LLM call. This test
    /// would catch, for example, a regression where the verbatim check was
    /// added as a separate leading `if is_verbatim { skip } else if
    /// post_process || spoken_hotword { ... }` — which compiles fine and
    /// looks correct, but silently drops the `else if final_text !=
    /// transcription` bookkeeping arm for verbatim apps because Rust
    /// `if`/`else if` arms are mutually exclusive.
    #[tokio::test]
    async fn verbatim_app_suppresses_llm_even_with_dedicated_force_binding() {
        let mut settings = get_default_settings();
        settings.h2_enabled = true;
        settings.h2_routes = crate::settings::default_h2_routes();
        // h2_verbatim_apps is left at its default (includes "powershell").
        assert!(
            crate::handy2::app_context::is_verbatim_app("powershell", &settings.h2_verbatim_apps),
            "test premise: 'powershell' must be in the default verbatim list"
        );

        let transcription = "git commit dash m fix the bug";
        let result = super::process_transcription_output_with_settings(
            &settings,
            transcription,
            true, // dedicated force-post-process binding: would normally ALWAYS route
            Some("powershell".to_string()),
        )
        .await;

        assert_eq!(
            result.llm_ms, None,
            "no LLM call should have been attempted — the verbatim app must suppress \
             routing even though post_process=true would otherwise force it"
        );
        assert_eq!(
            result.final_text, transcription,
            "verbatim apps get the exact raw transcript, not an LLM rewrite"
        );
        assert_eq!(
            result.post_processed_text, None,
            "no post-processed text should be recorded when the LLM never ran"
        );
    }

    /// Companion case for the same wiring: with NO verbatim app in play, the
    /// dedicated binding still forces routing exactly as before (regression
    /// guard for the refactor that introduced
    /// `process_transcription_output_with_settings`). Uses h2_enabled=false
    /// so this exercises the stock `post_process_transcription` path rather
    /// than requiring a live Ollama endpoint.
    #[tokio::test]
    async fn non_verbatim_app_does_not_suppress_the_dedicated_binding() {
        let mut settings = get_default_settings();
        settings.h2_enabled = false; // stock post-processing path, no network dependency

        let transcription = "just a plain sentence";
        let result = super::process_transcription_output_with_settings(
            &settings,
            transcription,
            true, // dedicated force-post-process binding
            Some("notepad".to_string()),
        )
        .await;

        // With h2 disabled and no post-process providers/prompt configured to
        // actually change the text, post_process_transcription legitimately
        // may return None (falls back to raw) — the point here is only that
        // the verbatim override did NOT short-circuit the gate itself: the
        // final text is unchanged from the raw transcript either way, but
        // critically this path is NOT the same code path as the verbatim
        // case above (no "skip" log, gate condition evaluates true going in).
        assert_eq!(result.final_text, transcription);
    }

    /// Task E5 review finding 1 (test-gap): pins the CORRECTED gate shape —
    /// `if verbatim_app.is_none() && would_have_routed { LLM } else if
    /// final_text != transcription { bookkeeping }` — against a regression to
    /// the BROKEN naive three-arm shape: `if verbatim_app.is_some() {} else
    /// if would_have_routed {…} else if final_text != transcription {…}`.
    ///
    /// The two tests above (`verbatim_app_suppresses_llm_even_with_dedicated_force_binding`
    /// and its companion) both pass against the naive shape too, because in
    /// both of them `final_text == transcription` going in — Chinese-variant
    /// conversion never ran (default `selected_language` is `"auto"`), so
    /// there's nothing for the naive shape's separate `is_some() {}` arm to
    /// wrongly swallow. The naive shape's bug only manifests when
    /// `final_text != transcription` for a verbatim app: its mutually
    /// exclusive `if`/`else if` arms mean the leading `if verbatim_app.is_some()
    /// {}` (a no-op arm) executes and control never reaches the `else if
    /// final_text != transcription` bookkeeping arm, so a real, non-LLM text
    /// change (Chinese-variant conversion) is silently DROPPED —
    /// `post_processed_text` stays `None` even though the pasted/history text
    /// differs from the raw transcript.
    ///
    /// This test sets `selected_language = "zh-Hant"` (Traditional Chinese),
    /// which per `maybe_convert_chinese_variant` applies OpenCC's `S2tw`
    /// config (Simplified -> Traditional). The input
    /// "国家计算机软件" is Simplified Chinese; independently verified via a
    /// throwaway probe against this exact `ferrous-opencc` version that
    /// `S2tw` converts it to "國家計算機軟件" (a real, deterministic,
    /// non-empty change — no model or network involved, matching this
    /// function's doc comment). A verbatim-app match (foreground app
    /// "powershell", on the default `h2_verbatim_apps` list) is also in play,
    /// so the ONLY way `post_processed_text` can be populated here is via the
    /// `else if final_text != transcription` bookkeeping arm — exactly the
    /// arm the naive shape drops.
    #[tokio::test]
    async fn verbatim_app_still_records_a_real_non_llm_text_change() {
        let mut settings = get_default_settings();
        settings.h2_enabled = true;
        settings.h2_routes = crate::settings::default_h2_routes();
        settings.selected_language = "zh-Hant".to_string();
        // h2_verbatim_apps is left at its default (includes "powershell").
        assert!(
            crate::handy2::app_context::is_verbatim_app("powershell", &settings.h2_verbatim_apps),
            "test premise: 'powershell' must be in the default verbatim list"
        );

        let transcription = "国家计算机软件"; // Simplified Chinese
        let expected_converted = "國家計算機軟件"; // Traditional, via OpenCC S2tw
        assert_ne!(
            transcription, expected_converted,
            "test premise: the OpenCC conversion must actually change the text"
        );

        let result = super::process_transcription_output_with_settings(
            &settings,
            transcription,
            true, // dedicated force-post-process binding: would normally ALWAYS route
            Some("powershell".to_string()),
        )
        .await;

        assert_eq!(
            result.llm_ms, None,
            "no LLM call should have been attempted — verbatim app suppresses routing"
        );
        assert_eq!(
            result.final_text, expected_converted,
            "the Chinese-variant conversion must still apply even for a verbatim app \
             (it's deterministic local text normalization, not an LLM rewrite)"
        );
        assert_eq!(
            result.post_processed_text,
            Some(expected_converted.to_string()),
            "the conversion result must be preserved in post_processed_text via the \
             `else if final_text != transcription` bookkeeping arm — this is exactly what \
             the naive fallthrough-drop regression would silently swallow"
        );
    }
}
