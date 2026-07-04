//! App-aware verbatim list (Task E5): suppress h2 LLM routing when dictation
//! starts while a terminal or coding agent is focused. Rewriting "git commit
//! dash m" into prose is actively harmful when the paste target is Claude
//! Code, VS Code, or a shell — those apps almost always want exact words.
//!
//! Design: the foreground app is captured at RECORDING START
//! (`TranscribeAction::start`), not at transcription time. By the time
//! `process_transcription_output` runs — inside a spawned task, possibly
//! seconds after the key was released — the OS foreground window is
//! unreliable (it could be Handy's own overlay, or the user could have
//! alt-tabbed away mid-dictation). Capturing at start matches "focus at
//! start == paste target under PTT" (per the task brief) and is threaded
//! through to the stop-time gate keyed by `binding_id`, since `start` and
//! `stop` are both `&self` methods on `TranscribeAction` with no shared
//! mutable instance state.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// binding_id -> captured foreground exe stem, recorded at recording start
/// and consumed (removed) at the stop-time gate. A plain process-wide map
/// keyed by binding_id avoids threading new state through `TranscribeAction`
/// itself, and entries are short-lived (start -> stop of a single dictation).
fn captured_apps() -> &'static Mutex<HashMap<String, String>> {
    static CAPTURED: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    CAPTURED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the foreground app stem for this binding at recording start.
///
/// `name: None` (no foreground window, or the Win32 call failed) actively
/// REMOVES any pre-existing entry for this `binding_id`, rather than leaving
/// it untouched. Without this, a stale entry from a prior dictation on the
/// same binding could survive and be wrongly `take()`n by a later dictation:
/// e.g. dictation 1 captures "powershell" then never reaches the stop-time
/// `take()` (cancelled, empty samples, transcription error), leaving the
/// entry in the map; dictation 2 on the same binding starts with no
/// foreground window detected (`None`) — if `remember` were a no-op here,
/// dictation 2's `take()` would return the STALE "powershell" from dictation
/// 1 and be wrongly treated as a verbatim app. Each `start()` call must fully
/// determine this binding's entry — present or absent — never inherit one
/// left over from an earlier dictation.
pub fn remember(binding_id: &str, name: Option<String>) {
    if let Ok(mut map) = captured_apps().lock() {
        match name {
            Some(name) => {
                map.insert(binding_id.to_string(), name);
            }
            None => {
                map.remove(binding_id);
            }
        }
    }
}

/// Retrieve and remove the foreground app stem captured for this binding at
/// recording start. Removing on read keeps the map from growing unboundedly
/// across the app's lifetime — each dictation's entry is consumed exactly
/// once by its own stop-time gate check.
pub fn take(binding_id: &str) -> Option<String> {
    captured_apps()
        .ok_or_lock()
        .and_then(|mut map| map.remove(binding_id))
}

/// Small local extension so a poisoned lock degrades to "no capture" instead
/// of panicking the dictation pipeline. A panicked lock-holder elsewhere in
/// the process must never turn into a crashed transcription flow here.
trait LockOrNone<T> {
    fn ok_or_lock(&self) -> Option<std::sync::MutexGuard<'_, T>>;
}

impl<T> LockOrNone<T> for Mutex<T> {
    fn ok_or_lock(&self) -> Option<std::sync::MutexGuard<'_, T>> {
        self.lock().ok()
    }
}

/// Case-insensitive substring match of the foreground app's exe stem against
/// the verbatim list. Pure and cheap — unit-tested directly; the Win32 call
/// that produces `exe_stem` is a thin, separate, not-unit-testable function
/// (`foreground_process_name`) that feeds this.
pub fn is_verbatim_app(exe_stem: &str, verbatim_apps: &[String]) -> bool {
    let stem_lower = exe_stem.to_lowercase();
    verbatim_apps
        .iter()
        .any(|app| !app.is_empty() && stem_lower.contains(&app.to_lowercase()))
}

/// Best-effort foreground-window exe stem (e.g. "Code", "WindowsTerminal",
/// "powershell") via Win32: `GetForegroundWindow` -> owning PID via
/// `GetWindowThreadProcessId` -> `OpenProcess` (query-limited-info handle,
/// the minimal access level `QueryFullProcessImageNameW` needs) ->
/// `QueryFullProcessImageNameW` -> take the path's file stem. Returns `None`
/// on any failure (no foreground window, access denied, protected process,
/// etc.) rather than propagating an error — a verbatim-app lookup that can't
/// determine the foreground app must degrade to "not a verbatim app" (normal
/// LLM routing proceeds), never panic and never block recording.
#[cfg(target_os = "windows")]
pub fn foreground_process_name() -> Option<String> {
    use windows::Win32::Foundation::{CloseHandle, HWND};
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    // SAFETY: all calls below are plain Win32 FFI with no user-provided
    // pointers; every output buffer is a locally owned, correctly sized
    // stack/Vec allocation, and every handle obtained is closed on every
    // return path (single `defer`-style guard via early return before the
    // handle is opened, then an explicit close after use).
    unsafe {
        let hwnd: HWND = GetForegroundWindow();
        if hwnd.is_invalid() {
            return None;
        }

        let mut pid: u32 = 0;
        let thread_id = GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if thread_id == 0 || pid == 0 {
            return None;
        }

        let handle = match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
            Ok(h) => h,
            Err(_) => return None,
        };

        // MAX_PATH is the conventional buffer size for this API; paths
        // exceeding it simply fail to round-trip, which is acceptable for a
        // best-effort lookup.
        let mut buf = [0u16; 260];
        let mut len: u32 = buf.len() as u32;
        let result = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut len,
        );
        let _ = CloseHandle(handle);

        if result.is_err() || len == 0 {
            return None;
        }

        let path = String::from_utf16_lossy(&buf[..len as usize]);
        std::path::Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
    }
}

/// Non-Windows builds (Linux/macOS dev/CI): the verbatim list is a
/// Windows-only capability per the task's scope (Handy's win32 build is the
/// target). Returning `None` here degrades exactly like a Win32 failure
/// would — normal LLM routing proceeds, nothing panics.
#[cfg(not(target_os = "windows"))]
pub fn foreground_process_name() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_stems_match_the_default_list() {
        let apps = crate::settings::get_default_settings().h2_verbatim_apps;
        for stem in [
            "powershell",
            "WindowsTerminal",
            "Code",
            "cmd",
            "conhost",
            "claude",
            "wt",
        ] {
            assert!(
                is_verbatim_app(stem, &apps),
                "expected '{stem}' to match the default verbatim list"
            );
        }
    }

    #[test]
    fn unrelated_app_does_not_match() {
        let apps = crate::settings::get_default_settings().h2_verbatim_apps;
        assert!(!is_verbatim_app("notepad", &apps));
        assert!(!is_verbatim_app("chrome", &apps));
        assert!(!is_verbatim_app("outlook", &apps));
    }

    #[test]
    fn matching_is_case_insensitive() {
        let apps = vec!["PowerShell".to_string()];
        assert!(is_verbatim_app("powershell", &apps));
        assert!(is_verbatim_app("POWERSHELL", &apps));
        assert!(is_verbatim_app("PowerShell", &apps));
    }

    #[test]
    fn substring_match_catches_versioned_or_suffixed_exe_names() {
        // Real-world exe stems aren't always exact: Windows Terminal's
        // process is "WindowsTerminal", VS Code's is "Code" but insiders
        // builds ship "Code - Insiders" -> stem still contains "code".
        let apps = vec!["code".to_string()];
        assert!(is_verbatim_app("Code - Insiders", &apps));
        assert!(is_verbatim_app("VSCode", &apps));
    }

    #[test]
    fn empty_verbatim_list_matches_nothing() {
        assert!(!is_verbatim_app("powershell", &[]));
        assert!(!is_verbatim_app("", &[]));
    }

    #[test]
    fn empty_list_entries_are_ignored_not_a_universal_match() {
        // A stray empty string in the settings list must not make every app
        // "match" via an empty-substring contains() check.
        let apps = vec!["".to_string()];
        assert!(!is_verbatim_app("notepad", &apps));
        assert!(!is_verbatim_app("", &apps));
    }

    #[test]
    fn remember_and_take_round_trip_by_binding_id() {
        let binding = "e5-test-binding-round-trip";
        remember(binding, Some("powershell".to_string()));
        assert_eq!(take(binding), Some("powershell".to_string()));
        // take() removes the entry: a second call finds nothing.
        assert_eq!(take(binding), None);
    }

    #[test]
    fn remember_none_stores_nothing() {
        let binding = "e5-test-binding-none";
        remember(binding, None);
        assert_eq!(take(binding), None);
    }

    /// Task E5 review finding 2 (stale-entry misfire): a `None` capture must
    /// actively clear any pre-existing entry for this binding, not leave it
    /// untouched. Reproduces the exact failure sequence: dictation 1 captures
    /// "a" but its `take()` is never reached (cancelled/empty/error path),
    /// then dictation 2 on the SAME binding starts with no foreground window
    /// detected (`None`). Without the fix, dictation 2's `take()` would
    /// return the stale "a" from dictation 1 instead of `None`.
    #[test]
    fn remember_none_clears_a_stale_prior_entry_for_the_same_binding() {
        let binding = "e5-test-binding-stale-clear";
        remember(binding, Some("a".to_string()));
        // Simulate dictation 1's take() never being reached (cancel/empty/error).
        remember(binding, None);
        assert_eq!(
            take(binding),
            None,
            "a None capture must clear the stale prior entry, not leave it for the next take()"
        );
    }

    #[test]
    fn take_on_unknown_binding_is_none_not_panic() {
        assert_eq!(take("e5-test-binding-never-remembered"), None);
    }
}
