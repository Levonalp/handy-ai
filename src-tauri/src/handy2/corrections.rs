//! Deterministic dictation corrections parsed from the personal memory file's
//! "## Dictation Corrections" table. Zero-latency, offline, applied in every
//! mode (raw + post-processed). Context-conditional rows — any row with "(" in
//! either cell, or "/" in the replacement — are left to the LLM and are NOT
//! applied here.

use regex::Regex;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub struct CorrectionRule {
    /// Compiled case-insensitive, boundary-guarded pattern for one heard variant.
    pattern: Regex,
    /// Raw heard text (for longest-first ordering and tests).
    heard: String,
    replacement: String,
    /// Original table line, so `strip_deterministic_rows` can drop exactly these.
    source_line: String,
}

const SECTION_HEADER: &str = "## Dictation Corrections";

fn boundary_regex(heard: &str) -> Option<Regex> {
    let escaped = regex::escape(heard);
    let lead = heard.chars().next().is_some_and(|c| c.is_alphanumeric());
    let trail = heard.chars().last().is_some_and(|c| c.is_alphanumeric());
    let pat = format!(
        "(?i){}{}{}",
        if lead { r"\b" } else { "" },
        escaped,
        if trail { r"\b" } else { "" }
    );
    Regex::new(&pat).ok()
}

fn table_cells(line: &str) -> Option<(String, String)> {
    let t = line.trim();
    if !t.starts_with('|') || !t.ends_with('|') || t.len() < 2 {
        return None;
    }
    let inner: Vec<&str> = t[1..t.len() - 1].split('|').collect();
    if inner.len() != 2 {
        return None;
    }
    Some((inner[0].trim().to_string(), inner[1].trim().to_string()))
}

fn is_separator_or_header(heard: &str, write: &str) -> bool {
    // Alignment separators may carry colons (|:---|---:|) after an Obsidian or
    // Prettier table re-format — treat any dash/space/colon-only cell as one.
    let sep = |s: &str| s.chars().all(|c| matches!(c, '-' | ' ' | ':'));
    sep(heard)
        || sep(write)
        || heard.eq_ignore_ascii_case("Heard (wrong)")
        || write.eq_ignore_ascii_case("Write (right)")
}

/// Parse the deterministic subset of the corrections table out of the memory
/// file text. Rows with `(` in either cell or `/` in the replacement are
/// context-conditional and skipped; ` / `-separated heard variants each become
/// their own rule. Rules are ordered longest-heard-first so overlapping
/// patterns don't shadow each other.
pub fn parse_rules(memory: &str) -> Vec<CorrectionRule> {
    let mut rules = Vec::new();
    let mut in_section = false;
    for line in memory.lines() {
        if line.trim_start().starts_with("## ") {
            in_section = line.trim_start().starts_with(SECTION_HEADER);
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((heard_cell, write_cell)) = table_cells(line) else {
            continue;
        };
        if is_separator_or_header(&heard_cell, &write_cell) {
            continue;
        }
        // Conditional rows stay LLM-only.
        if heard_cell.contains('(') || write_cell.contains('(') || write_cell.contains('/') {
            continue;
        }
        for variant in heard_cell.split(" / ") {
            let heard = variant.trim();
            if heard.is_empty() || write_cell.is_empty() {
                continue;
            }
            if let Some(pattern) = boundary_regex(heard) {
                rules.push(CorrectionRule {
                    pattern,
                    heard: heard.to_string(),
                    replacement: write_cell.clone(),
                    source_line: line.to_string(),
                });
            }
        }
    }
    rules.sort_by_key(|r| std::cmp::Reverse(r.heard.len()));
    rules
}

/// True when a match at `start` in `text` begins a sentence: start of text, or
/// preceded (ignoring spaces/tabs) by end punctuation or a newline. STT often
/// capitalizes proper-noun-shaped mishears mid-sentence ("call the Sloan
/// officer"), so the match's own case says nothing about position.
fn sentence_initial(text: &str, start: usize) -> bool {
    let last = text[..start].trim_end_matches([' ', '\t']).chars().last();
    matches!(last, None | Some('.' | '!' | '?' | '…' | '\n' | '\r'))
}

fn capitalize_first(s: &str) -> String {
    let mut cs = s.chars();
    match cs.next() {
        Some(f) => f.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}

pub fn apply(text: &str, rules: &[CorrectionRule]) -> String {
    let mut out = text.to_string();
    for rule in rules {
        let replaced = rule
            .pattern
            .replace_all(&out, |caps: &regex::Captures| {
                let m = caps.get(0).expect("group 0 always present");
                let repl_lower = rule
                    .replacement
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_lowercase());
                if repl_lower && sentence_initial(&out, m.start()) {
                    capitalize_first(&rule.replacement)
                } else {
                    rule.replacement.clone()
                }
            })
            .into_owned();
        out = replaced;
    }
    out
}

struct CachedRules {
    path: String,
    mtime: SystemTime,
    len: u64,
    rules: Arc<Vec<CorrectionRule>>,
    /// Raw memory text, kept alongside the pre-compiled rules so
    /// `expand_snippet_from_memory_file` can scan the `## Snippets` table
    /// without a second disk read. Snippet lookup is a cheap `lines()` scan
    /// (no regex compiles), so unlike `rules` there is no need to pre-parse
    /// it into its own cached structure — `expand_snippet` re-scans this
    /// string on every call, and stays the single source of truth for
    /// snippet-parsing logic (directly unit-tested, not duplicated here).
    memory_text: Arc<String>,
}

static RULE_CACHE: OnceLock<Mutex<Option<Arc<CachedRules>>>> = OnceLock::new();

/// Stat the memory file and, if the cache is stale (path/mtime/len changed),
/// re-read + re-parse it. Returns the fresh-or-cached entry, or `None` if
/// the path is unset or the file is unreadable. Shared by
/// `apply_from_memory_file` and `expand_snippet_from_memory_file` so the
/// file is read from disk at most once per change, regardless of which (or
/// both) callers run in a given dictation.
fn cached_memory(path: Option<&str>) -> Option<Arc<CachedRules>> {
    let path = path?;
    let meta = std::fs::metadata(path)
        .map_err(|e| log::debug!("corrections: memory file unreadable ({path}): {e}; skipping"))
        .ok()?;
    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let len = meta.len();

    let cache = RULE_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fresh = matches!(
        guard.as_ref(),
        Some(c) if c.path == path && c.mtime == mtime && c.len == len
    );
    if !fresh {
        match std::fs::read_to_string(path) {
            Ok(memory) => {
                *guard = Some(Arc::new(CachedRules {
                    path: path.to_string(),
                    mtime,
                    len,
                    rules: Arc::new(parse_rules(&memory)),
                    memory_text: Arc::new(memory),
                }));
            }
            Err(e) => {
                log::debug!("corrections: memory file unreadable ({path}): {e}; skipping");
                return None;
            }
        }
    }
    guard.as_ref().cloned()
}

/// Entry point for the transcription pipeline. Live-edit semantics are kept
/// (rules reload when the file's mtime or length changes) but the ~7ms
/// parse + 32-regex compile is paid only on change, not per dictation.
/// Unset/unreadable path ⇒ input returned unchanged.
pub fn apply_from_memory_file(text: &str, path: Option<&str>) -> String {
    match cached_memory(path) {
        Some(cached) => apply(text, &cached.rules),
        None => text.to_string(),
    }
}

/// Cache-aware entry point for voice-snippet expansion in the transcription
/// pipeline: reuses the same mtime/len-checked cache as
/// `apply_from_memory_file` (populated by whichever of the two runs first
/// for a given dictation) instead of re-reading the memory file, then
/// delegates to the pure, directly-tested `expand_snippet`. Unset/unreadable
/// path ⇒ `None` (never fires), matching `apply_from_memory_file`'s
/// no-op-on-missing-file behavior.
pub fn expand_snippet_from_memory_file(text: &str, path: Option<&str>) -> Option<String> {
    let cached = cached_memory(path)?;
    expand_snippet(text, &cached.memory_text)
}

/// Memory text minus the table rows `apply` already handles (they are applied
/// deterministically upstream, so the LLM prompt doesn't need them). Conditional
/// rows and all non-table content are preserved verbatim.
pub fn strip_deterministic_rows(memory: &str) -> String {
    let deterministic: std::collections::HashSet<String> = parse_rules(memory)
        .into_iter()
        .map(|r| r.source_line)
        .collect();
    memory
        .lines()
        .filter(|l| !deterministic.contains(*l))
        .collect::<Vec<_>>()
        .join("\n")
}

const LLM_SECTION_ALLOWLIST: [&str; 4] = [
    "## Dictation Corrections",
    "## Number Rules",
    "## Stock Phrases",
    "## Formatting & Style Rules",
];

const SNIPPETS_SECTION_HEADER: &str = "## Snippets";

/// Normalize an utterance for snippet-trigger comparison: trim, lowercase,
/// strip one trailing `.`/`!`/`?`/`,`. Applied identically to the spoken text
/// and to each trigger parsed from the table so "Insert approval stamp." and
/// "insert approval stamp" compare equal.
fn normalize_trigger(s: &str) -> String {
    s.trim()
        .trim_end_matches(['.', '!', '?', ','])
        .trim()
        .to_lowercase()
}

/// Deterministic voice-snippet expansion: if the WHOLE utterance (after
/// normalization) equals a trigger phrase in the memory file's `## Snippets`
/// table, return the paired expansion verbatim (`<br>` becomes a real
/// newline). No mid-text/substring firing — this is intentionally an
/// all-or-nothing match so a snippet trigger can't accidentally fire inside
/// a longer dictation. Pure function: no file I/O, no caching, so it is
/// directly unit-testable against a literal memory string.
pub fn expand_snippet(text: &str, memory: &str) -> Option<String> {
    let target = normalize_trigger(text);
    if target.is_empty() {
        return None;
    }
    let mut in_section = false;
    for line in memory.lines() {
        if line.trim_start().starts_with("## ") {
            in_section = line.trim_start().starts_with(SNIPPETS_SECTION_HEADER);
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((say_cell, paste_cell)) = table_cells(line) else {
            continue;
        };
        if is_separator_or_header(&say_cell, &paste_cell) {
            continue;
        }
        if say_cell.eq_ignore_ascii_case("Say") && paste_cell.eq_ignore_ascii_case("Paste") {
            continue;
        }
        if normalize_trigger(&say_cell) == target {
            return Some(paste_cell.replace("<br>", "\n"));
        }
    }
    None
}

/// The LLM prompt no longer carries the whole memory file: deterministic
/// layers own vocabulary/spelling, and the wrap+example own the ROLE rules.
/// Keep only sections the model still needs, minus already-applied rows.
pub fn slim_memory_for_llm(memory: &str) -> String {
    let mut kept = String::new();
    let mut keeping = false;
    for line in memory.lines() {
        if line.trim_start().starts_with("## ") {
            keeping = LLM_SECTION_ALLOWLIST
                .iter()
                .any(|h| line.trim_start().starts_with(h));
        }
        if keeping {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    strip_deterministic_rows(&kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
# Memory
## Dictation Corrections (apply these exact fixes when heard)
| Heard (wrong) | Write (right) |
|---|---|
| Praise overview | Appraisal review |
| Sloan officer | loan officer |
| G-Boost / XG boost | XGBoost |
| Valley (value context) | value |
| our bi / RB eyes | RBI / RBI's |
| Cloud RBI | Claude (Code) |
| bar war | borrower |

## Number Rules (critical)
- Hiring budget is 140,000.
";

    #[test]
    fn parses_unconditional_rows_and_variants() {
        let rules = parse_rules(FIXTURE);
        let heards: Vec<&str> = rules.iter().map(|r| r.heard.as_str()).collect();
        assert!(heards.contains(&"Praise overview"));
        assert!(heards.contains(&"G-Boost"));
        assert!(heards.contains(&"XG boost"));
        // Conditional rows skipped: '(' in heard, '/' in write, '(' in write.
        assert!(!heards.iter().any(|h| h.contains("Valley")));
        assert!(!heards.iter().any(|h| h.contains("our bi")));
        assert!(!heards.iter().any(|h| h.contains("Cloud RBI")));
    }

    #[test]
    fn applies_case_insensitively_with_word_boundaries() {
        let rules = parse_rules(FIXTURE);
        assert_eq!(
            apply("the praise overview memo about the bar war", &rules),
            "the Appraisal review memo about the borrower"
        );
        // 'bar wars' must NOT match ('s' breaks the trailing boundary).
        assert_eq!(apply("star bar wars", &rules), "star bar wars");
    }

    #[test]
    fn sentence_initial_capitalization_preserved() {
        let rules = parse_rules(FIXTURE);
        assert_eq!(
            apply("Sloan officer called.", &rules),
            "Loan officer called."
        );
        assert_eq!(
            apply("the sloan officer called", &rules),
            "the loan officer called"
        );
        // After end punctuation counts as sentence-initial.
        assert_eq!(
            apply("Done. Sloan officer next", &rules),
            "Done. Loan officer next"
        );
    }

    #[test]
    fn mid_sentence_uppercase_match_is_not_capitalized() {
        // STT capitalizes proper-noun-shaped mishears anywhere; position, not
        // the match's case, decides capitalization (review finding #2).
        let rules = parse_rules(FIXTURE);
        assert_eq!(
            apply("call the Sloan officer now", &rules),
            "call the loan officer now"
        );
    }

    #[test]
    fn alignment_separator_rows_are_not_rules() {
        // Obsidian/Prettier may rewrite |---|---| as |:---|---:| (finding #3).
        let fixture = "\
## Dictation Corrections
| Heard (wrong) | Write (right) |
|:---|---:|
| Powery | Bowery |
";
        let rules = parse_rules(fixture);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].heard, "Powery");
        // And the separator must survive prompt slimming.
        assert!(strip_deterministic_rows(fixture).contains(":---"));
    }

    #[test]
    fn cache_reloads_when_file_changes() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_corrections_cache_test.md");
        let table = |wrong: &str, right: &str| {
            format!("## Dictation Corrections\n| Heard (wrong) | Write (right) |\n|---|---|\n| {wrong} | {right} |\n")
        };
        std::fs::write(&path, table("Powery", "Bowery")).expect("write v1");
        let p = path.to_str().expect("utf8 path");
        assert_eq!(
            apply_from_memory_file("Powery here", Some(p)),
            "Bowery here"
        );
        // Different content AND different length so the (mtime,len) key flips
        // even on filesystems with coarse mtime granularity.
        std::fs::write(&path, table("Powery", "Bowery Valuation")).expect("write v2");
        assert_eq!(
            apply_from_memory_file("Powery here", Some(p)),
            "Bowery Valuation here"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn variants_map_to_same_replacement() {
        let rules = parse_rules(FIXTURE);
        assert_eq!(
            apply("use g-boost and XG boost", &rules),
            "use XGBoost and XGBoost"
        );
    }

    #[test]
    fn idempotent() {
        let rules = parse_rules(FIXTURE);
        let once = apply("praise overview", &rules);
        assert_eq!(apply(&once, &rules), once);
    }

    #[test]
    fn missing_file_is_noop() {
        assert_eq!(apply_from_memory_file("text", None), "text");
        assert_eq!(
            apply_from_memory_file("text", Some("Z:/no/such/file.md")),
            "text"
        );
    }

    #[test]
    fn strip_removes_only_deterministic_rows() {
        let stripped = strip_deterministic_rows(FIXTURE);
        assert!(!stripped.contains("Praise overview"));
        assert!(!stripped.contains("Sloan officer"));
        assert!(stripped.contains("Valley (value context)"));
        assert!(stripped.contains("our bi / RB eyes"));
        assert!(stripped.contains("Number Rules"));
    }

    #[test]
    fn slim_keeps_llm_sections_drops_deterministic_ones() {
        let mem = "\
## ⚠️ ROLE — READ FIRST\nrole text\n\
## Identity\n- Levon\n\
## Dictation Corrections (apply these exact fixes when heard)\n\
| Heard (wrong) | Write (right) |\n|---|---|\n\
| Powery | Bowery |\n| Valley (value context) | value |\n\
## Number Rules (critical)\n- budget 140,000\n\
## Stock Phrases (preserve verbatim)\n- \"Best, Levon\"\n\
## Formatting & Style Rules\n- keep imperative voice\n";
        let slim = slim_memory_for_llm(mem);
        assert!(!slim.contains("ROLE"));
        assert!(!slim.contains("Identity"));
        assert!(!slim.contains("| Powery | Bowery |")); // deterministic row gone
        assert!(slim.contains("Valley (value context)")); // conditional row kept
        assert!(slim.contains("Number Rules"));
        assert!(slim.contains("Stock Phrases"));
        assert!(slim.contains("Formatting & Style Rules"));
    }

    #[test]
    fn snippet_expands_whole_utterance_only() {
        let mem = "## Snippets\n| Say | Paste |\n|---|---|\n| insert approval stamp | Appraisal reviewed, value approved. |\n| insert signoff | Best,<br>Levon |\n";
        assert_eq!(
            expand_snippet("Insert approval stamp.", mem).as_deref(),
            Some("Appraisal reviewed, value approved.")
        );
        assert_eq!(
            expand_snippet("insert signoff", mem).as_deref(),
            Some("Best,\nLevon")
        );
        assert_eq!(
            expand_snippet("please insert approval stamp now", mem),
            None
        );
    }

    #[test]
    fn snippet_partial_or_trailing_extra_words_do_not_fire() {
        // Whole-utterance-only per the brief: a trigger that is only a
        // *prefix* of what was said must not fire either (not just the
        // "leading extra words" case the brief's own test covers).
        let mem = "## Snippets\n| Say | Paste |\n|---|---|\n| insert signoff | Best,<br>Levon |\n";
        assert_eq!(
            expand_snippet("insert signoff please", mem),
            None,
            "trigger as a prefix of a longer utterance must not fire"
        );
        assert_eq!(
            expand_snippet("insert signoff insert signoff", mem),
            None,
            "repeated trigger text must not fire (still not equal to the whole utterance)"
        );
    }

    #[test]
    fn snippet_trailing_punctuation_variants_all_normalize() {
        let mem = "## Snippets\n| Say | Paste |\n|---|---|\n| insert proceed stamp | Appraisal is approved. Please proceed. |\n";
        for spoken in [
            "insert proceed stamp",
            "insert proceed stamp.",
            "insert proceed stamp!",
            "insert proceed stamp?",
            "insert proceed stamp,",
            "  Insert Proceed Stamp  ",
            "INSERT PROCEED STAMP.",
        ] {
            assert_eq!(
                expand_snippet(spoken, mem).as_deref(),
                Some("Appraisal is approved. Please proceed."),
                "failed to normalize: {spoken:?}"
            );
        }
    }

    #[test]
    fn snippet_section_header_with_trailing_annotation_still_matches() {
        // The real seeded handy-memory.md header is
        // "## Snippets (say the phrase alone -> pastes the block verbatim)",
        // not the bare "## Snippets" the brief's own test fixture uses.
        // Section matching must use the same starts_with semantics as
        // parse_rules/LLM_SECTION_ALLOWLIST, not exact equality.
        let mem = "## Snippets (say the phrase alone \u{2192} pastes the block verbatim)\n\
                   | Say | Paste |\n|---|---|\n\
                   | insert signoff | Best,<br>Levon |\n";
        assert_eq!(
            expand_snippet("insert signoff", mem).as_deref(),
            Some("Best,\nLevon")
        );
    }

    #[test]
    fn snippet_section_is_not_in_llm_allowlist() {
        // Step 3 of the brief: '## Snippets' must NOT be exposed to the LLM
        // prompt via slim_memory_for_llm — it is a deterministic-only layer.
        let mem = "\
## Dictation Corrections\n| Heard (wrong) | Write (right) |\n|---|---|\n| Powery | Bowery |\n\
## Snippets (say the phrase alone -> pastes the block verbatim)\n\
| Say | Paste |\n|---|---|\n| insert signoff | Best,<br>Levon |\n\
## Number Rules\n- budget 140,000\n";
        let slim = slim_memory_for_llm(mem);
        assert!(
            !slim.contains("Snippets"),
            "Snippets section header leaked into the LLM prompt"
        );
        assert!(
            !slim.contains("insert signoff"),
            "Snippet trigger text leaked into the LLM prompt"
        );
        assert!(slim.contains("Number Rules"));
    }

    #[test]
    fn applies_against_the_real_memory_file_if_present() {
        let real = "C:/Users/lwsha/OneDrive/Desktop/Claude/Handy2.0/handy-memory.md";
        if let Ok(memory) = std::fs::read_to_string(real) {
            let rules = parse_rules(&memory);
            assert!(
                rules.len() >= 20,
                "expected >=20 deterministic rules, got {}",
                rules.len()
            );
            assert_eq!(
                apply("the Powery appraisal", &rules),
                "the Bowery appraisal"
            );
            assert_eq!(apply("check my sequel db", &rules), "check MySQL db");
        }
    }

    #[test]
    fn snippets_expand_against_the_real_memory_file_if_present() {
        let real = "C:/Users/lwsha/OneDrive/Desktop/Claude/Handy2.0/handy-memory.md";
        if let Ok(memory) = std::fs::read_to_string(real) {
            // Pure function against the real seeded '## Snippets' table,
            // including the trailing-punctuation/case normalization the
            // dictation pipeline relies on.
            assert_eq!(
                expand_snippet("Insert approval stamp.", &memory).as_deref(),
                Some("Appraisal reviewed, value approved.")
            );
            assert_eq!(
                expand_snippet("insert value approved", &memory).as_deref(),
                Some("Value reviewed and approved.")
            );
            assert_eq!(
                expand_snippet("insert proceed stamp", &memory).as_deref(),
                Some("Appraisal is approved. Please proceed.")
            );
            // Multi-line expansion: '<br>' in the table cell becomes a real
            // newline in the pasted text.
            assert_eq!(
                expand_snippet("insert signoff", &memory).as_deref(),
                Some("Best,\nLevon")
            );
            // Whole-utterance-only: extra words around the trigger must not
            // fire, even against the real file's content.
            assert_eq!(
                expand_snippet("please insert approval stamp for this file", &memory),
                None
            );

            // Cache-aware wrapper against the real path (not just the pure
            // function against literal text) — this is what
            // post_stt_pipeline actually calls.
            assert_eq!(
                expand_snippet_from_memory_file("insert signoff", Some(real)).as_deref(),
                Some("Best,\nLevon")
            );
            assert_eq!(
                expand_snippet_from_memory_file("not a snippet trigger at all", Some(real)),
                None
            );
        }
    }
}
