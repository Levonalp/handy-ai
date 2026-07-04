use crate::audio_toolkit::{apply_custom_words, filter_transcription_output, words_to_digits};
use crate::audio_toolkit::{constants, vad::VoiceActivityDetector, SileroVad};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::{EngineType, ModelManager};
use crate::settings::{
    get_settings, AppSettings, ModelUnloadTimeout, OrtAcceleratorSetting, WhisperAcceleratorSetting,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::Serialize;
use specta::Type;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::thread;
use std::time::{Duration, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        cohere::CohereModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    whisper_cpp::{WhisperEngine, WhisperInferenceParams},
    SpeechModel, TranscribeOptions,
};

/// Silero VAD frame size in samples (30ms @ 16kHz, matching
/// `audio_toolkit::vad::silero::SILERO_FRAME_SAMPLES`, which is private to
/// that module — recomputed here from the same public constants).
const STREAM_VAD_FRAME_SAMPLES: usize = (constants::WHISPER_SAMPLE_RATE * 30 / 1000) as usize;
const STREAM_VAD_FRAME_MS: u64 = 30;

/// Index just past the last silence run of `>= min_silence_ms` within
/// `samples`, or `None` if no qualifying run exists.
///
/// Scans `samples` in fixed 30ms VAD frames (any trailing partial frame is
/// left un-analyzed and simply falls after the returned cut, i.e. it becomes
/// part of the carry-over). A "run" is a maximal span of consecutive
/// non-speech frames; a run qualifies once its length in frames reaches
/// `min_silence_ms / 30` (integer division, matching the brief's spec).
/// When multiple qualifying runs exist, the cut lands just past the LAST one
/// scanning left-to-right, so speech that resumes after a long pause is kept
/// intact in the carry-over rather than split mid-run.
fn last_silence_cut(
    samples: &[f32],
    vad: &mut dyn VoiceActivityDetector,
    min_silence_ms: u64,
) -> Option<usize> {
    let frames_needed = (min_silence_ms / STREAM_VAD_FRAME_MS) as usize;
    if frames_needed == 0 {
        return None;
    }

    let mut cut: Option<usize> = None;
    let mut run_len: usize = 0;

    let frame_count = samples.len() / STREAM_VAD_FRAME_SAMPLES;
    for i in 0..frame_count {
        let start = i * STREAM_VAD_FRAME_SAMPLES;
        let end = start + STREAM_VAD_FRAME_SAMPLES;
        let frame = &samples[start..end];

        let is_speech = vad.is_voice(frame).unwrap_or(false);

        if is_speech {
            run_len = 0;
        } else {
            run_len += 1;
            if run_len >= frames_needed {
                // This run currently qualifies; keep pushing the candidate
                // cut forward while the run keeps growing so it always lands
                // just past the run's true end once speech resumes (or the
                // scan ends).
                cut = Some(end);
            }
        }
    }

    cut
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    Cohere(CohereModel),
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

/// Mutable state for one in-progress streaming-transcription session, keyed
/// by binding_id in `TranscriptionManager::streaming_sessions`. Wrapped in
/// its own `Mutex` (separate from the outer map's) so the poll thread can
/// hold the lock across an entire drain-VAD-transcribe-commit cycle without
/// blocking unrelated map operations — and so `finalize_streaming` blocks on
/// that same lock until any in-flight poll iteration has fully committed its
/// segment, rather than racing a join against a half-updated session.
struct StreamingSessionInner {
    /// Transcribed segment texts, in chronological order. Raw engine output
    /// only — the deterministic post-STT pipeline runs once, in
    /// `finalize_streaming`, on the fully joined text.
    segments: Vec<String>,
    /// Samples drained but not yet transcribed (after the last confirmed
    /// silence cut in the most recent poll, or the whole drain if no cut was
    /// found yet). Prepended to the next drain's samples before the next
    /// VAD pass, so a speech burst spanning two polls isn't cut mid-word.
    carry: Vec<f32>,
}

struct StreamingSession {
    inner: Mutex<StreamingSessionInner>,
    /// Set by `finalize_streaming` before it takes `inner`'s lock for the
    /// last time, so a poll iteration that wakes up after finalization has
    /// already begun (but before the session is removed from the map) can
    /// notice and exit immediately instead of draining/transcribing a
    /// segment that finalize_streaming can no longer see.
    finalized: AtomicBool,
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
    streaming_sessions: Arc<Mutex<HashMap<String, Arc<StreamingSession>>>>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
            streaming_sessions: Arc::new(Mutex::new(HashMap::new())),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .map_or(false, |a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        let engine = self.lock_engine();
        engine.is_some()
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = self.model_manager.get_model_path(model_id)?;

        // Create appropriate engine based on model type
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        let loaded_engine = match model_info.engine_type {
            EngineType::Whisper => {
                let engine = WhisperEngine::load(&model_path).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Whisper(engine)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::Cohere => {
                let engine = CohereModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load cohere model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Cohere(engine)
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading || self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();

        debug!("Audio vector length: {}", audio.len());

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        let validated_language = if settings.selected_language == "auto" {
            "auto".to_string()
        } else {
            let is_supported = self
                .model_manager
                .get_model_info(&settings.selected_model)
                .map(|info| {
                    info.supported_languages.is_empty()
                        || info
                            .supported_languages
                            .contains(&settings.selected_language)
                })
                .unwrap_or(true);

            if is_supported {
                settings.selected_language.clone()
            } else {
                warn!(
                    "Language '{}' not supported by current model, falling back to auto-detect",
                    settings.selected_language
                );
                "auto".to_string()
            }
        };

        // Perform transcription with the appropriate engine.
        let result = self.run_engine(&audio, &settings, &validated_language)?;

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        info!(
            "Transcription completed in {}ms{}",
            (et - st).as_millis(),
            translation_note
        );

        let final_result = self.post_stt_pipeline(result.text, &settings);

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }

    /// Dispatch to whichever engine is currently loaded and run inference.
    /// Assumes the caller already confirmed a model is loaded and resolved
    /// `validated_language`. Uses `catch_unwind` to prevent an engine panic
    /// from poisoning the engine mutex, which would hang the app on every
    /// subsequent transcription attempt.
    fn run_engine(
        &self,
        audio: &[f32],
        settings: &AppSettings,
        validated_language: &str,
    ) -> Result<transcribe_rs::TranscriptionResult> {
        let mut engine_guard = self.lock_engine();

        // Take the engine out so we own it during transcription.
        // If the engine panics, we simply don't put it back (effectively unloading it)
        // instead of poisoning the mutex.
        let mut engine = match engine_guard.take() {
            Some(e) => e,
            None => {
                return Err(anyhow::anyhow!(
                    "Model failed to load after auto-load attempt. Please check your model settings."
                ));
            }
        };

        // Release the lock before transcribing — no mutex held during the engine call
        drop(engine_guard);

        let transcribe_result = catch_unwind(AssertUnwindSafe(
            || -> Result<transcribe_rs::TranscriptionResult> {
                match &mut engine {
                    LoadedEngine::Whisper(whisper_engine) => {
                        let whisper_language = if validated_language == "auto" {
                            None
                        } else {
                            let normalized = if validated_language == "zh-Hans"
                                || validated_language == "zh-Hant"
                            {
                                "zh".to_string()
                            } else {
                                validated_language.to_string()
                            };
                            Some(normalized)
                        };

                        let params = WhisperInferenceParams {
                            language: whisper_language,
                            translate: settings.translate_to_english,
                            initial_prompt: if settings.custom_words.is_empty() {
                                None
                            } else {
                                Some(settings.custom_words.join(", "))
                            },
                            ..Default::default()
                        };

                        whisper_engine
                            .transcribe_with(audio, &params)
                            .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))
                    }
                    LoadedEngine::Parakeet(parakeet_engine) => {
                        let params = ParakeetParams {
                            timestamp_granularity: Some(TimestampGranularity::Segment),
                            ..Default::default()
                        };
                        parakeet_engine
                            .transcribe_with(audio, &params)
                            .map_err(|e| anyhow::anyhow!("Parakeet transcription failed: {}", e))
                    }
                    LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                        .transcribe(audio, &TranscribeOptions::default())
                        .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                    LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                        .transcribe(audio, &TranscribeOptions::default())
                        .map_err(|e| {
                            anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                        }),
                    LoadedEngine::SenseVoice(sense_voice_engine) => {
                        let language = match validated_language {
                            "zh" | "zh-Hans" | "zh-Hant" => Some("zh".to_string()),
                            "en" => Some("en".to_string()),
                            "ja" => Some("ja".to_string()),
                            "ko" => Some("ko".to_string()),
                            "yue" => Some("yue".to_string()),
                            _ => None,
                        };
                        let params = SenseVoiceParams {
                            language,
                            use_itn: Some(true),
                        };
                        sense_voice_engine
                            .transcribe_with(audio, &params)
                            .map_err(|e| anyhow::anyhow!("SenseVoice transcription failed: {}", e))
                    }
                    LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                        .transcribe(audio, &TranscribeOptions::default())
                        .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
                    LoadedEngine::Canary(canary_engine) => {
                        let lang = if validated_language == "auto" {
                            None
                        } else {
                            Some(validated_language.to_string())
                        };
                        let options = TranscribeOptions {
                            language: lang,
                            translate: settings.translate_to_english,
                            ..Default::default()
                        };
                        canary_engine
                            .transcribe(audio, &options)
                            .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
                    }
                    LoadedEngine::Cohere(cohere_engine) => {
                        let lang = if validated_language == "auto" {
                            None
                        } else if validated_language == "zh-Hans" || validated_language == "zh-Hant"
                        {
                            Some("zh".to_string())
                        } else {
                            Some(validated_language.to_string())
                        };
                        let options = TranscribeOptions {
                            language: lang,
                            ..Default::default()
                        };
                        cohere_engine
                            .transcribe(audio, &options)
                            .map_err(|e| anyhow::anyhow!("Cohere transcription failed: {}", e))
                    }
                }
            },
        ));

        match transcribe_result {
            Ok(inner_result) => {
                // Success or normal error — put the engine back
                let mut engine_guard = self.lock_engine();
                *engine_guard = Some(engine);
                inner_result
            }
            Err(panic_payload) => {
                // Engine panicked — do NOT put it back (it's in an unknown state).
                // The engine is dropped here, effectively unloading it.
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                error!(
                    "Transcription engine panicked: {}. Model has been unloaded.",
                    panic_msg
                );

                // Clear the model ID so it will be reloaded on next attempt
                {
                    let mut current_model = self
                        .current_model_id
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    *current_model = None;
                }

                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "unloaded".to_string(),
                        model_id: None,
                        model_name: None,
                        error: Some(format!("Engine panicked: {}", panic_msg)),
                    },
                );

                Err(anyhow::anyhow!(
                    "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                    panic_msg
                ))
            }
        }
    }

    /// Raw engine inference only — no post-STT text pipeline, no activity
    /// touch/unload bookkeeping (the caller owns those, since a streaming
    /// segment call shouldn't retrigger idle-timer/unload side effects that
    /// `transcribe()` performs once per whole dictation). Shared by both
    /// `transcribe()` (single-shot) and `begin_streaming`'s poll thread
    /// (per-segment) so engine-selection logic lives in exactly one place.
    fn transcribe_raw(&self, audio: &[f32]) -> Result<String> {
        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        let settings = get_settings(&self.app_handle);

        let validated_language = if settings.selected_language == "auto" {
            "auto".to_string()
        } else {
            let is_supported = self
                .model_manager
                .get_model_info(&settings.selected_model)
                .map(|info| {
                    info.supported_languages.is_empty()
                        || info
                            .supported_languages
                            .contains(&settings.selected_language)
                })
                .unwrap_or(true);

            if is_supported {
                settings.selected_language.clone()
            } else {
                warn!(
                    "Language '{}' not supported by current model, falling back to auto-detect",
                    settings.selected_language
                );
                "auto".to_string()
            }
        };

        let result = self.run_engine(audio, &settings, &validated_language)?;
        Ok(result.text)
    }

    /// The deterministic, offline text pipeline applied to a FINAL transcript
    /// (either the whole clip in the non-streaming path, or the joined
    /// segments + tail in the streaming path): custom-word correction ->
    /// filler-word/hallucination filtering -> "scratch that" self-correction
    /// -> spoken-number-to-digit normalization -> personal memory-file
    /// corrections -> voice-snippet expansion. Must run exactly once on the
    /// fully joined text so corrections and filler-removal can't behave
    /// differently at segment boundaries than they would on one contiguous
    /// transcript. The voice-snippet check is deliberately the LAST step and
    /// lives here (not at either call site) so it stays exactly-once too: if
    /// the corrected utterance matches a snippet trigger, this function
    /// returns the pasted expansion in place of the literal words.
    fn post_stt_pipeline(&self, text: String, settings: &AppSettings) -> String {
        // Apply word correction if custom words are configured.
        // Skip for Whisper models since custom words are already passed as initial_prompt.
        let is_whisper = self
            .model_manager
            .get_model_info(&settings.selected_model)
            .map(|info| matches!(info.engine_type, EngineType::Whisper))
            .unwrap_or(false);

        let corrected_result = if !settings.custom_words.is_empty() && !is_whisper {
            apply_custom_words(
                &text,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            text
        };

        // Filter out filler words and hallucinations
        let filtered_result = filter_transcription_output(
            &corrected_result,
            &settings.app_language,
            &settings.custom_filler_words,
        );

        // Live self-corrections: "send it Monday, scratch that, send it
        // Tuesday" drops the abandoned clause and the marker itself.
        // Deterministic, offline, no LLM. Runs before ITN/corrections so a
        // scratched-out clause's words (which might otherwise get digit- or
        // vocabulary-corrected) never reach those later steps at all.
        let scratched_result = crate::handy2::scratch::apply_scratch_that(&filtered_result);

        // Convert spoken numbers to digits ("twenty three" -> "23") for English.
        // Deterministic + offline, so the fast default path stays fast (no LLM).
        let final_result = if settings.app_language.to_lowercase().starts_with("en") {
            words_to_digits(&scratched_result)
        } else {
            scratched_result
        };

        // Deterministic vocabulary corrections from the personal memory file
        // (unconditional mishear-table rows only — context-dependent rows stay
        // LLM-side). Applies in every mode, offline; rules are cached on file
        // mtime/len so the per-dictation cost is a stat + string scan.
        let final_result = crate::handy2::corrections::apply_from_memory_file(
            &final_result,
            settings.h2_memory_file_path.as_deref(),
        );

        // Voice snippets: if the WHOLE (corrected) utterance is exactly a
        // trigger phrase in the memory file's "## Snippets" table, paste the
        // pre-written block instead of the literal words. Deterministic,
        // no-LLM, checked AFTER corrections so a mis-heard word in the
        // trigger itself still resolves correctly. Reuses the same
        // mtime/len-cached memory read `apply_from_memory_file` just used
        // above — no second file read. Must stay the only snippet check in
        // the pipeline so it runs exactly once per dictation, same as the
        // corrections call above (see `post_stt_pipeline` doc comment).
        if let Some(expansion) = crate::handy2::corrections::expand_snippet_from_memory_file(
            &final_result,
            settings.h2_memory_file_path.as_deref(),
        ) {
            info!("Transcription result: voice snippet expanded");
            return expansion;
        }

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!("Transcription result: {}", final_result);
        }

        final_result
    }

    /// Start background segment-streaming for an in-progress recording:
    /// every ~1500ms, pull whatever audio has accumulated since the last
    /// poll, look for a confirmed pause (>= `streaming_min_silence_ms` of
    /// silence), and transcribe everything up to that pause immediately —
    /// so most of the transcript already exists by the time the user
    /// releases the push-to-talk key. Segment texts accumulate in
    /// `StreamingSession::segments`, emitted as `transcript-partial` after
    /// each one lands. No-op (returns immediately) if a session already
    /// exists for this binding_id.
    ///
    /// The poll thread exits when either: (a) `drain_recording` reports the
    /// binding is no longer the active recording (covers normal stop, a
    /// different binding starting, and cancellation), or (b)
    /// `finalize_streaming` has removed/finalized the session. It never
    /// polls forever: every exit path is reached within one sleep interval
    /// of the recording actually ending.
    pub fn begin_streaming(&self, app: AppHandle, binding_id: String) {
        let mut sessions = self.streaming_sessions.lock().unwrap();
        if sessions.contains_key(&binding_id) {
            debug!("begin_streaming: session already active for binding {binding_id}, ignoring");
            return;
        }
        let session = Arc::new(StreamingSession {
            inner: Mutex::new(StreamingSessionInner {
                segments: Vec::new(),
                carry: Vec::new(),
            }),
            finalized: AtomicBool::new(false),
        });
        sessions.insert(binding_id.clone(), session.clone());
        drop(sessions);

        // Weak reference to the manager so this background thread can never
        // itself be the reason `Arc::strong_count(&self.engine) > 1` stays
        // true in `TranscriptionManager::drop` — a leaked/slow-to-exit poll
        // thread must not block the app's shutdown-detection logic.
        let manager_weak: Weak<TranscriptionManager> =
            match app.try_state::<Arc<TranscriptionManager>>() {
                Some(state) => Arc::downgrade(&state),
                None => {
                    error!("begin_streaming: TranscriptionManager not registered as app state");
                    self.streaming_sessions.lock().unwrap().remove(&binding_id);
                    return;
                }
            };

        let vad_path = match app.path().resolve(
            "resources/models/silero_vad_v4.onnx",
            tauri::path::BaseDirectory::Resource,
        ) {
            Ok(p) => p,
            Err(e) => {
                error!("begin_streaming: failed to resolve VAD model path: {e}");
                self.streaming_sessions.lock().unwrap().remove(&binding_id);
                return;
            }
        };

        // Snapshot once at streaming start rather than re-reading every
        // poll: a single recording session should use one consistent
        // silence threshold for its whole duration, not one that could
        // shift mid-session if the user changes settings while dictating.
        let min_silence_ms = get_settings(&app).streaming_min_silence_ms;

        let app_for_thread = app.clone();
        let poll_binding_id = binding_id.clone();

        thread::spawn(move || {
            let mut vad = match SileroVad::new(&vad_path, 0.3) {
                Ok(v) => v,
                Err(e) => {
                    error!("Streaming poll thread: failed to create VAD: {e}");
                    if let Some(manager) = manager_weak.upgrade() {
                        manager
                            .streaming_sessions
                            .lock()
                            .unwrap()
                            .remove(&poll_binding_id);
                    }
                    return;
                }
            };

            const POLL_INTERVAL: Duration = Duration::from_millis(1500);

            loop {
                thread::sleep(POLL_INTERVAL);

                // Manager gone (app shutting down) -> nothing left to do.
                let Some(manager) = manager_weak.upgrade() else {
                    debug!("Streaming poll thread: manager dropped, exiting");
                    return;
                };

                if session.finalized.load(Ordering::Acquire) {
                    debug!("Streaming poll thread: session finalized, exiting");
                    return;
                }

                let audio_mgr = match app_for_thread.try_state::<Arc<AudioRecordingManager>>() {
                    Some(s) => s.inner().clone(),
                    None => {
                        error!("Streaming poll thread: AudioRecordingManager not available");
                        return;
                    }
                };

                // Acquire the session lock BEFORE draining the recorder, and
                // hold it across the drain. This closes a data-loss race
                // against `finalize_streaming` (Task A2 review finding I1):
                // previously, `drain_recording` ran lock-free, so it could
                // pull real already-captured samples out of the recorder's
                // buffer (via `Cmd::Drain`'s `mem::take` — those samples are
                // gone from the recorder the instant that call returns) and
                // then lose the CPU before reaching `inner.lock()`. If
                // `finalize_streaming` won that race, it would see
                // `finalized == true` on this thread's post-lock re-check and
                // return, leaving the drained batch stranded in a local
                // variable — never folded into `carry`, never transcribed,
                // and not covered by `tail` either (tail only reflects audio
                // captured after the drain).
                //
                // Making "drain -> check finalized -> commit-or-bail" one
                // atomic sequence under `inner`'s lock removes the gap:
                //   - If finalize_streaming wins the lock race, it sets
                //     `finalized = true` and proceeds against the session as
                //     it stood at that moment. When this thread then gets the
                //     lock, it sees `finalized == true` and returns WITHOUT
                //     ever calling `drain_recording` — so it never removes
                //     samples from the recorder's buffer in the first place.
                //     Nothing is lost: `finalize_streaming` is always called
                //     with `tail` already produced by a prior, completed
                //     `stop_recording()` call (see actions.rs's stop handler,
                //     which calls `rm.stop_recording()` to completion before
                //     invoking `tm.finalize_streaming()` — the two never run
                //     concurrently), so any audio still sitting in the
                //     recorder at this point is exactly what `tail` already
                //     contains.
                //   - If this thread wins the lock race, it drains (now
                //     inside the lock), commits the batch or updates `carry`,
                //     and releases the lock; `finalize_streaming` then blocks
                //     on the same lock and proceeds only after this
                //     iteration's commit is fully visible.
                //
                // Lock ordering stays one-directional (`inner` -> recorder
                // manager's internal locks, never the reverse): `drain_recording`
                // only touches `AudioRecordingManager`'s own `state`/`recorder`
                // locks internally, and nothing in `audio.rs` ever acquires a
                // `StreamingSession.inner` lock, so no deadlock is possible.
                // Holding `inner` across the drain briefly delays
                // `finalize_streaming` if it's already blocked waiting for
                // this same lock, but that's the correct/intended
                // serialization (a channel round-trip to the recorder's
                // worker thread — fast, not instant), not a new stall on
                // anything else, since these two are the only two lock
                // holders.
                let mut inner = session.inner.lock().unwrap();

                if session.finalized.load(Ordering::Acquire) {
                    debug!(
                        "Streaming poll thread: finalized before drain, exiting without draining \
                         (finalize_streaming's tail already covers any audio left in the recorder)"
                    );
                    drop(inner);
                    return;
                }

                let drained = match audio_mgr.drain_recording(&poll_binding_id) {
                    Some(d) => d,
                    None => {
                        // Recording for this binding has ended (stopped,
                        // cancelled, or superseded) — nothing more will ever
                        // arrive for this session. Stop polling; whatever is
                        // sitting in `carry` is finalize_streaming's job to
                        // pick up (via the tail it's given directly, or an
                        // already-finalized session it will no-op against).
                        debug!("Streaming poll thread: recording no longer active, exiting");
                        drop(inner);
                        return;
                    }
                };

                if drained.is_empty() {
                    drop(inner);
                    continue;
                }

                let mut buf = std::mem::take(&mut inner.carry);
                buf.extend(drained);

                match last_silence_cut(&buf, &mut vad, min_silence_ms) {
                    Some(cut) if cut > 0 => {
                        let remainder = buf.split_off(cut);
                        let ready = buf; // now just the [0, cut) portion

                        match manager.transcribe_raw(&ready) {
                            Ok(text) => {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    inner.segments.push(trimmed.to_string());
                                    let joined = inner.segments.join(" ");
                                    let _ = app_for_thread.emit("transcript-partial", joined);
                                }
                            }
                            Err(e) => {
                                warn!("Streaming segment transcription failed: {e}");
                                // Don't lose the audio: fold it back in as
                                // carry so the next segment (or finalize)
                                // still covers it.
                                let mut recovered = ready;
                                recovered.extend(remainder);
                                inner.carry = recovered;
                                drop(inner);
                                continue;
                            }
                        }

                        inner.carry = remainder;
                    }
                    _ => {
                        // No qualifying pause yet — keep accumulating.
                        inner.carry = buf;
                    }
                }

                drop(inner);
            }
        });
    }

    /// Finish a streaming session: transcribe whatever carry-over remains
    /// plus the final `tail` samples handed to us by the stop path (audio
    /// captured after the poll thread's last drain), append that as the
    /// final segment, and return every segment joined with spaces — with
    /// the deterministic post-STT pipeline applied exactly once to that
    /// joined text.
    ///
    /// Blocks until any poll iteration that was already mid-flight (past
    /// its drain, running VAD/transcribe) has fully committed its segment:
    /// this method takes the same per-session lock the poll thread commits
    /// under, so it can never observe (or overwrite) a half-updated session.
    ///
    /// If no session exists for `binding_id` (streaming was disabled, or
    /// `begin_streaming` failed to start), transcribes `tail` alone via the
    /// normal single-shot path — this is the graceful fallback, not an
    /// error.
    pub fn finalize_streaming(&self, binding_id: &str, tail: Vec<f32>) -> Result<String> {
        let session = self.streaming_sessions.lock().unwrap().remove(binding_id);

        let Some(session) = session else {
            debug!(
                "finalize_streaming: no session for binding {binding_id}, falling back to single-shot transcribe"
            );
            return self.transcribe(tail);
        };

        // Touch activity / handle the force-failure hook the same way
        // transcribe() does, since this is also a top-level "a dictation
        // just finished" entry point.
        self.touch_activity();
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Blocks here until any in-flight poll commit finishes, THEN marks
        // the session finalized while still holding the lock so a poll
        // iteration that was merely blocked on this same lock (not yet
        // re-checking `finalized`) commits its work before we read
        // `segments` below — and any iteration that hasn't reached the lock
        // yet will see `finalized == true` on its next check and bail
        // without touching `segments`/`carry` again.
        let mut inner = session.inner.lock().unwrap();
        session.finalized.store(true, Ordering::Release);

        let mut final_audio = std::mem::take(&mut inner.carry);
        final_audio.extend(tail);

        if !final_audio.is_empty() {
            match self.transcribe_raw(&final_audio) {
                Ok(text) => {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        inner.segments.push(trimmed.to_string());
                    }
                }
                Err(e) => {
                    // Surface the error only if we have nothing at all to
                    // return — if earlier segments already succeeded, a
                    // failure on just the tail shouldn't discard everything
                    // the user already said.
                    if inner.segments.is_empty() {
                        self.maybe_unload_immediately("transcription");
                        return Err(e);
                    }
                    warn!(
                        "finalize_streaming: tail transcription failed, returning {} prior segment(s) without it: {e}",
                        inner.segments.len()
                    );
                }
            }
        }

        let joined = inner.segments.join(" ");
        drop(inner);

        let settings = get_settings(&self.app_handle);
        let final_result = self.post_stt_pipeline(joined, &settings);

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }
}

/// Apply the user's accelerator preferences to the transcribe-rs global atomics.
/// Called on startup and whenever the user changes the setting.
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    let whisper_pref = match settings.whisper_accelerator {
        WhisperAcceleratorSetting::Auto => accel::WhisperAccelerator::Auto,
        WhisperAcceleratorSetting::Cpu => accel::WhisperAccelerator::CpuOnly,
        WhisperAcceleratorSetting::Gpu => accel::WhisperAccelerator::Gpu,
    };
    accel::set_whisper_accelerator(whisper_pref);
    accel::set_whisper_gpu_device(settings.whisper_gpu_device);
    info!(
        "Whisper accelerator set to: {}, gpu_device: {}",
        whisper_pref,
        if settings.whisper_gpu_device == accel::GPU_DEVICE_AUTO {
            "auto".to_string()
        } else {
            settings.whisper_gpu_device.to_string()
        }
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: i32,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    use transcribe_rs::whisper_cpp::gpu::list_gpu_devices;

    GPU_DEVICES.get_or_init(|| {
        // ggml's Vulkan backend uses FMA3 instructions internally.
        // On older CPUs without FMA3 (e.g. Sandy Bridge Xeons) this causes
        // a SIGILL crash that cannot be caught. Skip enumeration entirely
        // on those CPUs — GPU-accelerated whisper won't work there anyway.
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("fma") {
            warn!("CPU lacks FMA3 support — skipping GPU device enumeration");
            return Vec::new();
        }

        list_gpu_devices()
            .into_iter()
            .map(|d| GpuDeviceOption {
                id: d.id,
                name: d.name,
                total_vram_mb: d.total_vram / (1024 * 1024),
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub whisper: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return which accelerators are compiled into this build.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let whisper_options = vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()];

    AvailableAccelerators {
        whisper: whisper_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully
        if let Some(handle) = self.watcher_handle.lock().unwrap().take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::audio_toolkit::vad::VadFrame;

    /// Deterministic stand-in for `SileroVad` in unit tests: a frame is
    /// "speech" if its RMS amplitude exceeds a fixed threshold. Avoids
    /// depending on the ONNX model file for pure logic tests.
    struct FakeAmplitudeVad {
        threshold: f32,
    }

    impl VoiceActivityDetector for FakeAmplitudeVad {
        fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> anyhow::Result<VadFrame<'a>> {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            if rms > self.threshold {
                Ok(VadFrame::Speech(frame))
            } else {
                Ok(VadFrame::Noise)
            }
        }
    }

    fn fake_vad() -> FakeAmplitudeVad {
        FakeAmplitudeVad { threshold: 0.1 }
    }

    /// `n` samples of silence (exact zeros — well under any amplitude threshold).
    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// `n` samples of a loud sine "speech" burst (amplitude well above threshold).
    fn speech(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.8 * (i as f32 * 0.4).sin()).collect()
    }

    /// Convert a duration in ms to an exact whole number of VAD frames' worth
    /// of samples, so test buffers align cleanly on frame boundaries.
    fn ms_to_frame_aligned_samples(ms: u64) -> usize {
        let frames = (ms / STREAM_VAD_FRAME_MS) as usize;
        frames * STREAM_VAD_FRAME_SAMPLES
    }

    #[test]
    fn no_silence_returns_none() {
        let mut vad = fake_vad();
        let samples = speech(ms_to_frame_aligned_samples(1000));
        assert_eq!(last_silence_cut(&samples, &mut vad, 500), None);
    }

    #[test]
    fn empty_buffer_returns_none() {
        let mut vad = fake_vad();
        assert_eq!(last_silence_cut(&[], &mut vad, 500), None);
    }

    #[test]
    fn silence_shorter_than_threshold_is_not_a_cut() {
        let mut vad = fake_vad();
        // 300ms silence gap, threshold requires 500ms.
        let mut samples = speech(ms_to_frame_aligned_samples(500));
        samples.extend(silence(ms_to_frame_aligned_samples(300)));
        samples.extend(speech(ms_to_frame_aligned_samples(500)));
        assert_eq!(last_silence_cut(&samples, &mut vad, 500), None);
    }

    #[test]
    fn qualifying_silence_run_in_the_middle_cuts_just_past_it() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(500);
        let gap = ms_to_frame_aligned_samples(600); // >= 500ms threshold
        let speech_b = ms_to_frame_aligned_samples(500);

        let mut samples = speech(speech_a);
        samples.extend(silence(gap));
        samples.extend(speech(speech_b));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        assert_eq!(cut, Some(speech_a + gap));
    }

    #[test]
    fn silence_run_exactly_at_threshold_counts() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(500);
        // Exactly 500ms = 16 frames * 30ms = 480ms... min_silence_ms/30 uses
        // integer division, so 500/30 = 16 frames = 480ms is the actual
        // frame-quantized threshold. Use that exact quantized length.
        let frames_needed = 500 / STREAM_VAD_FRAME_MS;
        let gap = (frames_needed as usize) * STREAM_VAD_FRAME_SAMPLES;

        let mut samples = speech(speech_a);
        samples.extend(silence(gap));
        samples.extend(speech(ms_to_frame_aligned_samples(500)));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        assert_eq!(cut, Some(speech_a + gap));
    }

    #[test]
    fn trailing_silence_at_buffer_end_cuts_at_buffer_end() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(500);
        let trailing_gap = ms_to_frame_aligned_samples(700);

        let mut samples = speech(speech_a);
        samples.extend(silence(trailing_gap));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        assert_eq!(cut, Some(speech_a + trailing_gap));
    }

    #[test]
    fn multiple_qualifying_runs_cut_at_the_last_one() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(400);
        let gap_1 = ms_to_frame_aligned_samples(600);
        let speech_b = ms_to_frame_aligned_samples(400);
        let gap_2 = ms_to_frame_aligned_samples(900);
        let speech_c = ms_to_frame_aligned_samples(400);

        let mut samples = speech(speech_a);
        samples.extend(silence(gap_1));
        samples.extend(speech(speech_b));
        samples.extend(silence(gap_2));
        samples.extend(speech(speech_c));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        // Must land just past gap_2 (the LAST qualifying run), not gap_1.
        assert_eq!(cut, Some(speech_a + gap_1 + speech_b + gap_2));
    }

    #[test]
    fn extended_silence_run_cuts_past_its_full_length_not_just_past_threshold() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(500);
        // A single silence run much longer than the 500ms threshold: the cut
        // must land at the end of the WHOLE run, not at the first instant it
        // qualified partway through.
        let long_gap = ms_to_frame_aligned_samples(2000);
        let speech_b = ms_to_frame_aligned_samples(500);

        let mut samples = speech(speech_a);
        samples.extend(silence(long_gap));
        samples.extend(speech(speech_b));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        assert_eq!(cut, Some(speech_a + long_gap));
    }

    #[test]
    fn trailing_partial_frame_is_excluded_from_the_scan() {
        let mut vad = fake_vad();
        let speech_a = ms_to_frame_aligned_samples(500);
        let gap = ms_to_frame_aligned_samples(600);
        let mut samples = speech(speech_a);
        samples.extend(silence(gap));
        // Append a partial (sub-frame) tail that should simply be ignored by
        // the scan (frame_count = len / FRAME_SAMPLES truncates it away).
        samples.extend(speech(STREAM_VAD_FRAME_SAMPLES / 2));

        let cut = last_silence_cut(&samples, &mut vad, 500);
        assert_eq!(cut, Some(speech_a + gap));
    }

    #[test]
    fn zero_min_silence_returns_none() {
        let mut vad = fake_vad();
        let samples = silence(ms_to_frame_aligned_samples(1000));
        // frames_needed = 0/30 = 0 -> guarded to return None rather than
        // treating every frame as a trivially-qualifying zero-length run.
        assert_eq!(last_silence_cut(&samples, &mut vad, 0), None);
    }

    // ---------------------------------------------------------------------
    // StreamingSessionInner bookkeeping (session state transitions in
    // isolation, no recorder/engine involved).
    // ---------------------------------------------------------------------

    #[test]
    fn session_segments_join_in_order() {
        let inner = StreamingSessionInner {
            segments: vec![
                "hello".to_string(),
                "world".to_string(),
                "again".to_string(),
            ],
            carry: Vec::new(),
        };
        assert_eq!(inner.segments.join(" "), "hello world again");
    }

    #[test]
    fn session_carry_over_prepends_to_next_drain() {
        // Mirrors the poll thread's `buf = take(carry); buf.extend(drained)`
        // step: carry from a previous iteration must be the PREFIX of the
        // next scan, not lost or reordered.
        let mut inner = StreamingSessionInner {
            segments: Vec::new(),
            carry: vec![1.0, 2.0, 3.0],
        };
        let drained = vec![4.0, 5.0];

        let mut buf = std::mem::take(&mut inner.carry);
        buf.extend(drained);

        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(inner.carry.is_empty());
    }

    // ---------------------------------------------------------------------
    // Regression test for Task A2 review finding I1: a data-loss race
    // between the poll thread's drain and `finalize_streaming`.
    //
    // The bug (pre-fix): the poll thread called `drain_recording` — which
    // irrevocably removes samples from the recorder's buffer — BEFORE
    // acquiring `session.inner`'s lock. If it was then descheduled and
    // `finalize_streaming` acquired the lock first (setting `finalized` and
    // reading `carry`), the poll thread would wake up, re-check `finalized`
    // under the lock, see `true`, and return — silently discarding the
    // batch it had already drained (it's a local variable at that point,
    // never folded into `carry`, and outside `tail`'s coverage too).
    //
    // The fix: the drain now happens INSIDE the same `session.inner` lock
    // that `finalize_streaming` takes to set `finalized`/read `carry`, so
    // "drain -> check finalized -> commit" is one atomic sequence. This test
    // exercises that exact sequence directly (bypassing the real recorder
    // and AppHandle, which the poll thread cannot be driven through in a
    // unit test — see the integration test below for why) using a
    // `Barrier` to force both competing code paths to race for the lock at
    // the same instant on every iteration, rather than relying on sleep
    // timing (inherently flaky to assert on). Because the OS scheduler's
    // choice of lock winner is not itself controlled, the test repeats the
    // race many times so both orderings (poll-wins / finalize-wins) are
    // very likely to occur across the run; the assertion holds for either
    // ordering, which is the actual invariant that matters — not "the poll
    // thread must win" or "finalize must win," but "no drained sample is
    // ever unaccounted for no matter which one wins."
    #[test]
    fn drain_and_finalize_race_never_loses_a_drained_batch() {
        use std::sync::Barrier;

        const ITERATIONS: usize = 200;

        for i in 0..ITERATIONS {
            let session = Arc::new(StreamingSession {
                inner: Mutex::new(StreamingSessionInner {
                    segments: Vec::new(),
                    carry: vec![0.1, 0.2], // pre-existing carry, like a real session mid-flight
                }),
                finalized: AtomicBool::new(false),
            });

            // The batch a "drain" would irrevocably pull out of the
            // recorder's buffer this iteration. Distinct per-iteration
            // values so a mixed-up/duplicated batch would be detectable,
            // though the primary assertion below is presence-or-never-taken.
            let drained_batch = vec![1.0 + i as f32, 2.0 + i as f32, 3.0 + i as f32];
            // Tracks whether the "drain" stand-in was ever actually invoked
            // this iteration, mirroring the real `drain_recording` call
            // that — once made — cannot be undone (the samples are gone
            // from the recorder's buffer the instant it returns).
            let drain_was_called = Arc::new(AtomicBool::new(false));

            let barrier = Arc::new(Barrier::new(2));

            // Poll-thread stand-in: reproduces the exact fixed sequence from
            // `begin_streaming`'s loop body — acquire `inner.lock()` first,
            // check `finalized`, and only THEN "drain" (irrevocably) and
            // fold the result into `carry`.
            let poll_session = session.clone();
            let poll_barrier = barrier.clone();
            let poll_drained = drained_batch.clone();
            let poll_drain_called = drain_was_called.clone();
            let poll_thread = thread::spawn(move || {
                poll_barrier.wait();
                let mut inner = poll_session.inner.lock().unwrap();
                if poll_session.finalized.load(Ordering::Acquire) {
                    // Must exit WITHOUT ever "draining" — this is the crux
                    // of the fix: no drain call means no samples were ever
                    // removed from the recorder's buffer in the first
                    // place, so there is nothing to lose.
                    return;
                }
                // "Drain": irrevocable, mirrors `drain_recording` returning
                // real captured samples that no longer exist anywhere else.
                poll_drain_called.store(true, Ordering::SeqCst);
                let mut buf = std::mem::take(&mut inner.carry);
                buf.extend(poll_drained);
                // No silence cut found this iteration (simplification for
                // this test) -> whole batch becomes the new carry, exactly
                // like the `_ => inner.carry = buf` arm in production.
                inner.carry = buf;
            });

            // finalize_streaming stand-in: acquire the same lock, set
            // `finalized`, and "commit" whatever carry it sees.
            let fin_session = session.clone();
            let fin_barrier = barrier.clone();
            let finalize_thread = thread::spawn(move || {
                fin_barrier.wait();
                let mut inner = fin_session.inner.lock().unwrap();
                fin_session.finalized.store(true, Ordering::Release);
                // "Commit": in the real code this is `final_audio = carry +
                // tail`; here we just fold carry into segments so we can
                // assert on total sample count afterward.
                let committed: Vec<f32> = std::mem::take(&mut inner.carry);
                inner.segments.push(format!("{}", committed.len()));
            });

            poll_thread.join().expect("poll stand-in thread panicked");
            finalize_thread
                .join()
                .expect("finalize stand-in thread panicked");

            // The invariant: EITHER the drain was never called (poll thread
            // saw finalized==true before draining -> nothing was ever
            // removed from "the recorder," nothing to account for), OR it
            // was called and its samples are now present in `carry` or
            // already folded into a committed segment. What must NEVER
            // happen: drain_was_called == true AND the batch is nowhere —
            // that's exactly the bug (a drained-but-orphaned local
            // variable). We verify this via total sample count: the
            // pre-existing carry (2 samples) plus, if-and-only-if the drain
            // stand-in ran, the 3 drained samples, must equal the sum of
            // whatever's left in `carry` plus whatever got committed into
            // `segments`.
            let inner = session.inner.lock().unwrap();
            let remaining_carry_len = inner.carry.len();
            let committed_len: usize = inner
                .segments
                .iter()
                .map(|s| s.parse::<usize>().unwrap())
                .sum();
            drop(inner);

            let expected_total = 2 + if drain_was_called.load(Ordering::SeqCst) {
                3
            } else {
                0
            };
            assert_eq!(
                remaining_carry_len + committed_len,
                expected_total,
                "iteration {i}: samples lost across the drain/finalize race \
                 (drain_was_called={}, remaining_carry={}, committed={})",
                drain_was_called.load(Ordering::SeqCst),
                remaining_carry_len,
                committed_len,
            );
        }
    }

    // ---------------------------------------------------------------------
    // Streaming-vs-batch integration test: drive the actual segment-cutting
    // algorithm (last_silence_cut + carry-over accounting, the same
    // sequence begin_streaming's poll thread performs) over a REAL recorded
    // dictation from disk, split across multiple simulated "drains" timed
    // like the live 1500ms poll. This cannot go through
    // TranscriptionManager/AudioRecordingManager directly (both require a
    // live tauri::AppHandle<Wry> — MockRuntime is a different, incompatible
    // concrete AppHandle type, and neither manager is generic over
    // Runtime — making a from-scratch AppHandle out of reach without
    // invasive changes to both managers, well outside this task). Instead
    // this test proves the property that actually matters for correctness:
    // the segment-cutting scheme partitions the ORIGINAL sample buffer
    // exactly — no samples dropped, none duplicated, cuts land on real
    // detected silence — which is the guarantee that makes "transcribe the
    // pieces separately" equivalent to "transcribe the whole clip," modulo
    // STT's own segment-boundary wording, which the task brief explicitly
    // accepts as a non-exact-match difference.
    #[test]
    fn streaming_segmentation_reconstructs_whole_buffer_from_a_real_recording() {
        let recordings_dir = std::path::PathBuf::from(std::env::var("APPDATA").unwrap())
            .join("com.pais.handy")
            .join("recordings");
        let wav_path = recordings_dir.join("handy-1782574976.wav");

        if !wav_path.exists() {
            eprintln!(
                "skipping streaming_segmentation_reconstructs_whole_buffer_from_a_real_recording: \
                 {:?} not present on this machine",
                wav_path
            );
            return;
        }

        let full_audio = crate::audio_toolkit::read_wav_samples(&wav_path)
            .expect("failed to read real recording for integration test");
        assert!(
            full_audio.len() > constants::WHISPER_SAMPLE_RATE as usize * 20,
            "fixture recording should be a meaningfully long (20s+) real dictation"
        );

        // Real VAD, same construction as production (create_audio_recorder /
        // preload_vad), so this exercises the actual model, not the fake.
        let vad_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("models")
            .join("silero_vad_v4.onnx");
        if !vad_path.exists() {
            eprintln!(
                "skipping streaming_segmentation_reconstructs_whole_buffer_from_a_real_recording: \
                 VAD model not present at {:?}",
                vad_path
            );
            return;
        }
        let mut vad = SileroVad::new(&vad_path, 0.3).expect("failed to load real Silero VAD");

        // Simulate the poll thread: feed the recording in ~1500ms-equivalent
        // chunks (matching POLL_INTERVAL), applying the exact
        // carry-over + last_silence_cut + split_off sequence begin_streaming
        // uses, and record each emitted segment's sample RANGE (not text —
        // no live engine here) so we can verify full, non-overlapping
        // coverage of the original buffer.
        let chunk_size = constants::WHISPER_SAMPLE_RATE as usize * 1500 / 1000; // ~1.5s
        let min_silence_ms = 500u64;

        let mut carry: Vec<f32> = Vec::new();
        let mut committed_len: usize = 0; // total samples handed off as "ready" segments so far
        let mut segment_count = 0usize;

        for chunk in full_audio.chunks(chunk_size) {
            let mut buf = std::mem::take(&mut carry);
            buf.extend_from_slice(chunk);

            match last_silence_cut(&buf, &mut vad, min_silence_ms) {
                Some(cut) if cut > 0 => {
                    let remainder = buf.split_off(cut);
                    let ready = buf;
                    committed_len += ready.len();
                    segment_count += 1;
                    carry = remainder;
                }
                _ => {
                    carry = buf;
                }
            }
        }

        // Whatever never got cut becomes the "tail" finalize_streaming would
        // transcribe — added to committed_len to check full coverage.
        let final_tail_len = carry.len();

        assert_eq!(
            committed_len + final_tail_len,
            full_audio.len(),
            "segment-cutting must partition the buffer exactly: no samples dropped or duplicated"
        );

        // A ~58s real dictation with normal speech pauses should yield at
        // least one mid-recording segment cut — if it never does, either the
        // fixture has no pauses at all (unlikely for natural speech) or the
        // cutting logic isn't finding real silence, which would defeat the
        // entire point of this task (nothing gets transcribed early).
        assert!(
            segment_count >= 1,
            "expected at least one background segment cut in a ~58s real dictation, got 0 \
             (streaming would provide no early-transcription benefit)"
        );

        eprintln!(
            "streaming integration test: {} segment(s) cut over {} total samples \
             ({:.1}s), final tail {} samples ({:.1}s)",
            segment_count,
            full_audio.len(),
            full_audio.len() as f64 / constants::WHISPER_SAMPLE_RATE as f64,
            final_tail_len,
            final_tail_len as f64 / constants::WHISPER_SAMPLE_RATE as f64
        );
    }
}
