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

/// Validate a single correction cell (`heard` or `write`): rejects
/// empty/whitespace-only text, `|` (would break the Markdown table), `(`
/// (would make `parse_rules` treat the row as context-conditional/LLM-only —
/// see the module doc — so a taught row containing `(` would silently never
/// apply deterministically, which is worse than refusing it up front), and
/// any ASCII/Unicode control character (`char::is_control`) — this catches
/// `\n`/`\r` (which would splice the row across physical lines and corrupt
/// the Markdown table structure, see the module doc) plus stray `\t` and
/// other control chars in one sweep, while ordinary text, punctuation, and
/// non-ASCII letters (e.g. Chinese) still pass.
///
/// `allow_slash` is `false` for the `write` cell only: `parse_rules:82` skips
/// any row whose WRITE cell contains `/`, so a taught replacement containing
/// `/` would silently never apply, exactly like the `(` case above. The
/// HEARD cell is the opposite: `parse_rules:85` splits it on `" / "` to let
/// one row teach multiple heard-variants (e.g. `foo / phoo` both mean
/// `bar`), so `/` there is meaningful and must stay allowed. A single
/// bool-parameterized function (rather than two near-duplicate functions)
/// keeps the shared checks — empty, `|`, `(`, control chars — in one place
/// while making the `/` rule the only cell-specific branch.
fn validate_cell(label: &str, cell: &str, allow_slash: bool) -> Result<(), String> {
    if cell.trim().is_empty() {
        return Err(format!("'{label}' must not be empty."));
    }
    if cell.contains('|') {
        return Err(format!("'{label}' must not contain '|' (breaks the table row)."));
    }
    if cell.contains('(') {
        return Err(format!(
            "'{label}' must not contain '(' (would make the row context-conditional and it would never apply automatically)."
        ));
    }
    if !allow_slash && cell.contains('/') {
        return Err(format!(
            "'{label}' must not contain '/' (it would prevent the correction from applying)."
        ));
    }
    if cell.chars().any(|c| c.is_control()) {
        return Err(format!("'{label}' must not contain line breaks."));
    }
    Ok(())
}

/// Append a new `| heard | write |` row to the memory file's
/// `## Dictation Corrections` table, teaching a correction without the user
/// hand-editing Markdown.
///
/// Algorithm: read the file, scan its lines tracking whether we're inside the
/// `## Dictation Corrections` section (same `starts_with("## ")` /
/// `starts_with(SECTION_HEADER)` heading logic `parse_rules` uses, so this
/// stays in lockstep with what the parser considers the section boundary).
/// While inside the section, remember the index of the last line recognized
/// as a table row by `table_cells` (this includes the `| Heard | Write |`
/// header and the `|---|---|` separator, which is intentional: we want the
/// new row inserted after the *last* row of the table as it physically
/// appears, not just after the last data row, so a table that is only a
/// header+separator with zero data rows yet still gets the new row appended
/// directly under the separator). The section ends at the next `## ` heading
/// or EOF. If no table row was ever seen inside the section (or the section
/// itself doesn't exist), that's an error — we refuse to guess where to
/// splice text into a file that doesn't have the expected shape, rather than
/// appending somewhere that could corrupt unrelated content.
///
/// Edge cases, explicit behavior:
/// - `path` unreadable or missing on disk → `Err` (surfaced by the caller
///   when `h2_memory_file_path` is `None`/unset, or here if the path is set
///   but the file itself can't be read).
/// - Memory file has no `## Dictation Corrections` section at all → `Err`
///   naming the missing section, no write happens.
/// - Section exists but has zero table rows (not even a header) → `Err`,
///   since there is nothing to anchor "insert after the last row" to.
///
/// On success, writes the file back in place and returns `Ok(())`. The
/// existing mtime/len-keyed `RULE_CACHE` invalidates itself on the next
/// dictation once it observes the changed file, so no cache-busting call is
/// needed here.
pub fn append_correction_row(path: &str, heard: &str, write: &str) -> Result<(), String> {
    validate_cell("Heard", heard, true)?;
    validate_cell("Should be", write, false)?;
    let heard = heard.trim();
    let write = write.trim();

    let memory = std::fs::read_to_string(path)
        .map_err(|e| format!("Could not read memory file '{path}': {e}"))?;

    let mut last_row_line_idx: Option<usize> = None;
    let mut found_section = false;
    let mut in_section = false;
    let lines: Vec<&str> = memory.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("## ") {
            in_section = line.trim_start().starts_with(SECTION_HEADER);
            if in_section {
                found_section = true;
            }
            continue;
        }
        if in_section && table_cells(line).is_some() {
            last_row_line_idx = Some(idx);
        }
    }

    if !found_section {
        return Err(format!(
            "Memory file has no '{SECTION_HEADER}' section; add one before teaching a correction."
        ));
    }
    let Some(insert_after) = last_row_line_idx else {
        return Err(format!(
            "'{SECTION_HEADER}' section has no table rows to insert after."
        ));
    };

    let new_row = format!("| {heard} | {write} |");
    let mut out_lines: Vec<String> = Vec::with_capacity(lines.len() + 1);
    for (idx, line) in lines.iter().enumerate() {
        out_lines.push((*line).to_string());
        if idx == insert_after {
            out_lines.push(new_row.clone());
        }
    }
    // `Vec<&str>::lines()` drops the file's trailing newline (if any); restore
    // one so the rewritten file still ends in `\n` like a normal text file.
    let mut new_content = out_lines.join("\n");
    new_content.push('\n');

    std::fs::write(path, new_content)
        .map_err(|e| format!("Could not write memory file '{path}': {e}"))?;
    Ok(())
}

/// True when a match at `start` in `text` begins a sentence: start of text, or
/// preceded (ignoring spaces/tabs) by end punctuation or a newline. STT often
/// capitalizes proper-noun-shaped mishears mid-sentence ("call the Sloan
/// officer"), so the match's own case says nothing about position.
fn sentence_initial(text: &str, start: usize) -> bool {
    let last = text[..start].trim_end_matches([' ', '\t']).chars().last();
    matches!(last, None | Some('.' | '!' | '?' | '…' | '\n' | '\r'))
}

/// `pub(crate)` (not private) so `handy2::scratch` can reuse the same
/// sentence-initial capitalization semantics after splicing text at a
/// deletion boundary, instead of duplicating this trivial logic.
pub(crate) fn capitalize_first(s: &str) -> String {
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
    fn append_correction_row_inserts_inside_section_and_parses() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_basic_test.md");
        std::fs::write(
            &path,
            "# Memory\n\
             ## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             | Sloan officer | loan officer |\n\
             \n\
             ## Snippets\n\
             | Say | Paste |\n\
             |---|---|\n\
             | insert signoff | Best,<br>Levon |\n",
        )
        .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        append_correction_row(p, "praise overview", "appraisal review").expect("append ok");

        let after = std::fs::read_to_string(&path).expect("read back");
        // New row landed immediately after the pre-existing data row, i.e.
        // still inside '## Dictation Corrections', not after '## Snippets'
        // and not tacked on at EOF.
        let corrections_idx = after.find("## Dictation Corrections").unwrap();
        let snippets_idx = after.find("## Snippets").unwrap();
        let new_row_idx = after.find("| praise overview | appraisal review |").unwrap();
        assert!(new_row_idx > corrections_idx && new_row_idx < snippets_idx);

        let existing_row_idx = after.find("| Sloan officer | loan officer |").unwrap();
        assert!(
            new_row_idx > existing_row_idx,
            "new row must come after the prior last row, not before it"
        );

        // Snippets table must be untouched.
        assert!(after.contains("| insert signoff | Best,<br>Levon |"));

        // parse_rules picks up the taught correction and applies it.
        let rules = parse_rules(&after);
        assert_eq!(
            apply("the praise overview memo", &rules),
            "the appraisal review memo"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_inserts_after_separator_when_no_data_rows_yet() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_empty_table_test.md");
        std::fs::write(
            &path,
            "## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             ## Number Rules\n\
             - budget 140,000\n",
        )
        .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        append_correction_row(p, "bar war", "borrower").expect("append ok");

        let after = std::fs::read_to_string(&path).expect("read back");
        let separator_idx = after.find("|---|---|").unwrap();
        let number_rules_idx = after.find("## Number Rules").unwrap();
        let new_row_idx = after.find("| bar war | borrower |").unwrap();
        assert!(new_row_idx > separator_idx && new_row_idx < number_rules_idx);

        let rules = parse_rules(&after);
        assert_eq!(apply("the bar war called", &rules), "the borrower called");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_at_eof_with_no_trailing_heading() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_eof_test.md");
        // No trailing newline and no section after Dictation Corrections —
        // exercises the "next heading or EOF" boundary at end-of-file.
        std::fs::write(
            &path,
            "## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             | Sloan officer | loan officer |",
        )
        .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        append_correction_row(p, "praise overview", "appraisal review").expect("append ok");

        let after = std::fs::read_to_string(&path).expect("read back");
        let rules = parse_rules(&after);
        let heards: Vec<&str> = rules.iter().map(|r| r.heard.as_str()).collect();
        assert!(heards.contains(&"Sloan officer"));
        assert!(heards.contains(&"praise overview"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_rejects_missing_section() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_no_section_test.md");
        std::fs::write(&path, "# Memory\n## Number Rules\n- budget 140,000\n")
            .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        let before = std::fs::read_to_string(&path).expect("read fixture back");
        let err = append_correction_row(p, "heard", "write")
            .expect_err("must error when section is missing");
        assert!(err.contains("Dictation Corrections"));

        // File must be left untouched, not partially written / corrupted.
        let after = std::fs::read_to_string(&path).expect("read after failed append");
        assert_eq!(before, after);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_rejects_section_with_zero_table_rows() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_empty_section_test.md");
        std::fs::write(
            &path,
            "## Dictation Corrections\nNo table here yet.\n## Number Rules\n- x\n",
        )
        .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        let err = append_correction_row(p, "heard", "write")
            .expect_err("must error when section has no table rows at all");
        assert!(err.contains("no table rows"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_rejects_unreadable_path() {
        let err = append_correction_row("Z:/no/such/file.md", "heard", "write")
            .expect_err("must error on unreadable path");
        assert!(err.contains("Could not read"));
    }

    #[test]
    fn append_correction_row_validates_heard_and_write() {
        let dir = std::env::temp_dir();
        let path = dir.join("h2_append_correction_validation_test.md");
        std::fs::write(
            &path,
            "## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             | Sloan officer | loan officer |\n",
        )
        .expect("write fixture");
        let p = path.to_str().expect("utf8 path");

        // Empty / whitespace-only.
        assert!(append_correction_row(p, "", "write").is_err());
        assert!(append_correction_row(p, "heard", "").is_err());
        assert!(append_correction_row(p, "   ", "write").is_err());
        assert!(append_correction_row(p, "heard", "   ").is_err());

        // Contains '|' (would break the table row).
        assert!(append_correction_row(p, "a | b", "write").is_err());
        assert!(append_correction_row(p, "heard", "a | b").is_err());

        // Contains '(' (would become context-conditional / LLM-only, i.e.
        // would silently never apply deterministically).
        assert!(append_correction_row(p, "valley (value context)", "value").is_err());
        assert!(append_correction_row(p, "heard", "value (approx)").is_err());

        // File must be untouched after every rejection above.
        let after = std::fs::read_to_string(&path).expect("read after rejections");
        assert!(!after.contains("| a | b |"));
        assert!(!after.contains("valley (value context)"));
        assert_eq!(
            after,
            "## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             | Sloan officer | loan officer |\n"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Fixture path for the '/' and control-char rejection tests below.
    /// Shared name kept unique per test (via a distinct suffix) so tests
    /// running in parallel don't clash on the same file.
    fn write_fixture(name: &str) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir();
        let path = dir.join(name);
        let contents = "## Dictation Corrections\n\
             | Heard (wrong) | Write (right) |\n\
             |---|---|\n\
             | Sloan officer | loan officer |\n";
        std::fs::write(&path, contents).expect("write fixture");
        (path, contents.to_string())
    }

    #[test]
    fn append_correction_row_rejects_slash_in_write_cell_only() {
        // Finding 1 (review): parse_rules:82 silently skips any row whose
        // WRITE cell contains '/', so accepting it here would write a row
        // that looks taught but never actually fires. Must be rejected.
        let (path, before) = write_fixture("h2_append_correction_slash_write_test.md");
        let p = path.to_str().expect("utf8 path");

        let err = append_correction_row(p, "praise overview", "appraisal review/summary")
            .expect_err("'/' in write must be rejected");
        assert!(err.contains('/'), "error should mention the offending character: {err}");

        // File must be untouched, matching the existing rejection-test
        // before/after byte-equality pattern.
        let after = std::fs::read_to_string(&path).expect("read after rejection");
        assert_eq!(before, after);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_accepts_slash_in_heard_cell() {
        // Critical asymmetry (review finding 1): parse_rules:85 splits the
        // HEARD cell on " / " to let one row teach multiple heard-variants,
        // so '/' there is meaningful and must NOT be rejected. This is a
        // positive test guarding against a future refactor accidentally
        // making the '/' check symmetric.
        let (path, _before) = write_fixture("h2_append_correction_slash_heard_test.md");
        let p = path.to_str().expect("utf8 path");

        append_correction_row(p, "foo / phoo", "bar").expect("'/' in heard must be accepted");

        let after = std::fs::read_to_string(&path).expect("read back");
        assert!(after.contains("| foo / phoo | bar |"));

        // Both heard-variants parsed out of the '/'-joined cell and both
        // apply the same replacement.
        let rules = parse_rules(&after);
        assert_eq!(apply("i said foo just now", &rules), "i said bar just now");
        assert_eq!(apply("i said phoo just now", &rules), "i said bar just now");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_rejects_newline_in_write_cell() {
        // Finding 2 (review): an embedded '\n' in write would splice the
        // row across two physical lines via format!()+join("\n"), corrupting
        // the Markdown table (e.g. injecting a fake "## " heading line).
        let (path, before) = write_fixture("h2_append_correction_newline_write_test.md");
        let p = path.to_str().expect("utf8 path");

        let err = append_correction_row(p, "heard text", "value\n## Fake Heading")
            .expect_err("embedded newline in write must be rejected");
        assert!(
            err.contains("line break"),
            "error should mention line breaks: {err}"
        );

        let after = std::fs::read_to_string(&path).expect("read after rejection");
        assert_eq!(before, after);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_correction_row_rejects_carriage_return_in_heard_cell() {
        // Same class as the newline-in-write test above, but covering the
        // OTHER cell and the OTHER control character ('\r'), since the
        // control-char sweep applies identically to both cells (unlike the
        // '/' check, which is write-only).
        let (path, before) = write_fixture("h2_append_correction_cr_heard_test.md");
        let p = path.to_str().expect("utf8 path");

        let err = append_correction_row(p, "heard\r## Injected", "write")
            .expect_err("embedded carriage return in heard must be rejected");
        assert!(
            err.contains("line break"),
            "error should mention line breaks: {err}"
        );

        let after = std::fs::read_to_string(&path).expect("read after rejection");
        assert_eq!(before, after);

        let _ = std::fs::remove_file(&path);
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
