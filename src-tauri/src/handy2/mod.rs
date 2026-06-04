//! Handy 2.0 additions: spoken-hotword routing + live Obsidian memory
//! injection layered onto Handy's existing transcribe→LLM→paste pipeline.
//!
//! Ported from the standalone Handy 2.0 v1 (`router`/`prompt`/`memory`/
//! `secrets`). Pure modules are unit-tested; `post_process` orchestrates them
//! against Handy's existing `llm_client` and is invoked from `actions.rs`
//! only when `settings.h2_enabled` is true.

pub mod memory;
pub mod prompt;
pub mod routing;
pub mod secrets;

use crate::settings::AppSettings;
use log::{debug, error, warn};

/// v2 post-processing: route → memory → prompt → Ollama. Returns the formatted
/// text, or `None` to fall back to the raw transcript (never lose dictation).
pub async fn post_process(settings: &AppSettings, transcription: &str) -> Option<String> {
    let decision = routing::route(transcription, &settings.h2_routes);
    if decision.cleaned_text.is_empty() {
        debug!("h2: hotword-only utterance, nothing to format");
        return None;
    }

    let memory_contents = match memory::read(settings.h2_memory_file_path.as_deref()) {
        memory::MemoryRead::Loaded(c) => Some(c),
        memory::MemoryRead::Absent { warning } => {
            if let Some(w) = warning {
                warn!("h2: {w}");
            }
            None
        }
    };
    let built = prompt::build(
        memory_contents.as_deref(),
        decision.route.prompt_addendum.as_deref(),
    );

    let provider = match settings
        .post_process_providers
        .iter()
        .find(|p| p.id == "ollama")
        .cloned()
    {
        Some(p) => p,
        None => {
            error!("h2: ollama provider preset missing");
            return None;
        }
    };
    let api_key = match secrets::get_key() {
        Ok(k) => k,
        Err(e) => {
            warn!("h2: {e}; pasting raw transcript");
            return None;
        }
    };

    match crate::llm_client::send_chat_completion_with_schema(
        &provider,
        api_key,
        &decision.route.ollama_model,
        decision.cleaned_text.clone(),
        Some(built.system_prompt),
        None, // structured output off for the Ollama preset (revisit per spec §11)
        None,
        None,
    )
    .await
    {
        Ok(Some(text)) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Ok(_) => {
            error!("h2: empty completion");
            None
        }
        Err(e) => {
            error!("h2: LLM failed: {e}; pasting raw transcript");
            None
        }
    }
}
