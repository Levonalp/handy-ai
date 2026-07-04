//! Handy 2.0 commands: Ollama key (OS credential store), h2 settings, test
//! connection. Follows Handy's command conventions (Result<_, String>).

use crate::handy2::{corrections, secrets};
use crate::settings::{get_settings, write_settings, Route};
use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub fn set_ollama_key(key: String) -> Result<(), String> {
    secrets::set_key(&key)
}

#[tauri::command]
#[specta::specta]
pub fn has_ollama_key() -> Result<bool, String> {
    secrets::has_key()
}

#[tauri::command]
#[specta::specta]
pub fn delete_ollama_key() -> Result<(), String> {
    secrets::delete_key()
}

#[tauri::command]
#[specta::specta]
pub fn set_h2_enabled(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.h2_enabled = enabled;
    // Handy 2.0 rides Handy's post-process pathway; enabling it must also turn on
    // post-processing so the "transcribe with post-process" (Ctrl+Shift+Space)
    // shortcut actually registers.
    if enabled {
        s.post_process_enabled = true;
    }
    write_settings(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_h2_memory_path(app: AppHandle, path: Option<String>) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.h2_memory_file_path = path.filter(|p| !p.trim().is_empty());
    write_settings(&app, s);
    Ok(())
}

/// Teach a correction from Settings: appends a `| heard | write |` row to the
/// configured memory file's `## Dictation Corrections` table. Applies on the
/// next dictation with no restart — `corrections::cached_memory`'s mtime/len
/// check picks up the on-disk change automatically.
#[tauri::command]
#[specta::specta]
pub fn append_correction(app: AppHandle, heard: String, write: String) -> Result<(), String> {
    let path = get_settings(&app)
        .h2_memory_file_path
        .ok_or_else(|| "No memory file path configured in Handy 2.0 settings.".to_string())?;
    corrections::append_correction_row(&path, &heard, &write)
}

#[tauri::command]
#[specta::specta]
pub fn set_h2_routes(app: AppHandle, routes: Vec<Route>) -> Result<(), String> {
    if !routes.iter().any(|r| r.trigger.is_none()) {
        return Err("Route table must contain a default route (trigger = null).".to_string());
    }
    let mut s = get_settings(&app);
    s.h2_routes = routes;
    write_settings(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn test_ollama_connection(app: AppHandle) -> Result<String, String> {
    let s = get_settings(&app);
    let provider = s
        .post_process_providers
        .iter()
        .find(|p| p.id == "ollama")
        .cloned()
        .ok_or_else(|| "Ollama provider not configured".to_string())?;
    let key = secrets::get_key()?;
    let model = s
        .h2_routes
        .iter()
        .find(|r| r.trigger.is_none())
        .map(|r| r.ollama_model.clone())
        .ok_or_else(|| "No default route configured".to_string())?;
    crate::llm_client::send_chat_completion(
        &provider,
        key,
        &model,
        "Reply with OK.".to_string(),
        None,
        None,
    )
    .await
    .map(|_| format!("Connected. Model '{model}' responded."))
}
