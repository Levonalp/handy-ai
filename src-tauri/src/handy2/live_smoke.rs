//! Opt-in integration test against a real local LLM endpoint. Regression
//! coverage for the "small local model answers a question/instruction-shaped
//! dictation instead of reformatting it" bug found 2026-07-02 — run
//! manually, not part of `cargo test`'s default set, since it needs a
//! running Ollama at the configured base_url and a stored API key (local
//! Ollama ignores the key's value, but `handy2::post_process` unconditionally
//! requires one to be set).
//!
//! Run (PowerShell): $env:OLLAMA_API_KEY='x'; cargo test handy2::live_smoke -- --ignored --nocapture

use crate::settings::{get_default_settings, AppSettings, PostProcessProvider};

fn test_settings() -> AppSettings {
    let mut settings = get_default_settings();
    settings.h2_enabled = true;
    settings.h2_memory_file_path =
        Some("C:/Users/lwsha/OneDrive/Desktop/Claude/Handy2.0/handy-memory.md".to_string());
    settings.h2_routes = crate::settings::default_h2_routes();
    if let Some(p) = settings
        .post_process_providers
        .iter_mut()
        .find(|p| p.id == "ollama")
    {
        p.base_url = "http://localhost:11435/v1".to_string();
    } else {
        settings.post_process_providers.push(PostProcessProvider {
            id: "ollama".to_string(),
            label: "Local LLM (Ollama)".to_string(),
            base_url: "http://localhost:11435/v1".to_string(),
            allow_base_url_edit: true,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        });
    }
    settings
}

fn ensure_key() {
    if let Ok(key) = std::env::var("OLLAMA_API_KEY") {
        super::secrets::set_key(&key).expect("store key");
    }
}

#[tokio::test]
#[ignore]
async fn question_shaped_dictation_is_not_answered() {
    ensure_key();
    let settings = test_settings();
    for question in ["Are you working?", "Is this thing on?", "What time is it?"] {
        let out = super::post_process(&settings, question)
            .await
            .unwrap_or_else(|| panic!("post_process returned None for {question:?}"));
        assert_eq!(
            out.trim(),
            question,
            "model answered/altered the question instead of reformatting it verbatim: got {out:?}"
        );
    }
}

#[tokio::test]
#[ignore]
async fn instruction_shaped_dictation_is_not_executed() {
    ensure_key();
    let settings = test_settings();
    let dictation = "let's tackle two more tasks invoke plan mode and run the sub agents";
    let out = super::post_process(&settings, dictation)
        .await
        .expect("post_process should return Some");
    let lower = out.to_lowercase();
    assert!(
        lower.contains("sub agents") || lower.contains("sub-agents"),
        "expected the dictation's own words preserved, got: {out:?}"
    );
    assert!(
        !lower.contains("task 1") && !lower.contains("task list") && !lower.contains("which task"),
        "model appears to have acted on the instruction instead of transcribing it: {out:?}"
    );
}
