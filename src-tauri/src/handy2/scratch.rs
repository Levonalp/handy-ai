//! Deterministic "scratch that" / "strike that" self-corrections. If the
//! user talks over their own mistake ("send it Monday, scratch that, send
//! it Tuesday"), drop the abandoned clause and the marker itself rather
//! than paste both the mistake and the correction. Zero-latency, offline,
//! no LLM involved — this runs in the same deterministic pass as filler-word
//! filtering and personal-memory corrections.
//!
//! Deliberately conservative: only the exact markers "scratch that" and
//! "strike that" (case-insensitive, optionally comma-wrapped) trigger a
//! deletion. Fuzzy/paraphrased self-corrections ("delete that", "never
//! mind", "actually no") are out of scope — a false-positive deletion here
//! silently destroys real dictated content, which is worse than leaving a
//! marker phrase unhandled.
//!
//! The marker regex alone is directional-boundary-agnostic on its trailing
//! side (it only ever absorbs an *optional* comma), so by itself it cannot
//! tell a genuine self-correction interjection from an ordinary sentence
//! that happens to contain "scratch/strike that" followed by more of the
//! same clause — "We need to strike that clause from the contract" reads
//! exactly like a marker if the text after "that" is never inspected. Every
//! candidate match is therefore additionally screened by
//! `is_genuine_marker` before it is treated as real; see that function for
//! the trailing-boundary and clause-echo conditions it checks.

use crate::handy2::corrections::capitalize_first;
use regex::Regex;
use std::sync::OnceLock;

/// Matches one marker occurrence together with the punctuation immediately
/// around it that should be deleted along with it: an optional leading
/// `, ` and an optional trailing `,`. `[ \t]*` (not `\s*`) deliberately
/// excludes `\n` from the punctuation the marker itself can absorb — a
/// newline is a sentence boundary in its own right (see
/// `previous_boundary`), not incidental whitespace around the marker.
///
/// This pattern alone over-matches (see the module doc comment): it finds
/// every "scratch/strike that" occurrence regardless of what follows, so
/// every result from it is a *candidate* match that `is_genuine_marker`
/// must still confirm before `apply_scratch_that` acts on it.
///
/// Capture group 1 ends immediately after "that", *before* the trailing
/// `[ \t]*,?` absorption — `is_genuine_marker`'s trailing-boundary check
/// needs to know whether a comma actually followed "that" in the original
/// text, which the full match's end (`m.end()`, group 0) can no longer tell
/// it once that comma has been absorbed into the match itself.
fn marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(,?[ \t]*(?:scratch|strike)[ \t]+that)[ \t]*,?")
            .expect("static scratch-marker pattern is valid regex")
    })
}

/// Matches a bare marker with none of `marker_regex`'s optional
/// comma/whitespace absorption — used only to test whether another marker
/// immediately follows a candidate match (the "back-to-back" trailing
/// condition in `is_genuine_marker`), where absorbing surrounding
/// punctuation would be irrelevant to the question being asked.
fn bare_marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:scratch|strike)[ \t]+that")
            .expect("static bare scratch-marker pattern is valid regex")
    })
}

/// Index just after the last of `. ! ? \n` in `text[..before]`, or `0` if
/// none exists — i.e. the start of the sentence containing `before`. The
/// punctuation mark itself is kept (index points just past it) so the
/// retained prefix still ends with its original terminal punctuation.
fn previous_boundary(text: &str, before: usize) -> usize {
    text[..before]
        .rfind(['.', '!', '?', '\n'])
        .map(|i| i + 1)
        .unwrap_or(0)
}

/// The first maximal run of alphabetic/apostrophe characters found starting
/// at or after `from` in `text` (skipping any leading non-word characters
/// such as whitespace/punctuation), lowercased — or `None` if `text[from..]`
/// has no such run. Apostrophes are included so contractions ("I'll",
/// "don't") compare as a single word rather than splitting at the `'`.
fn first_word_lowercased(text: &str, from: usize) -> Option<String> {
    let rest = &text[from..];
    let start = rest.find(|c: char| c.is_alphabetic())?;
    let word: String = rest[start..]
        .chars()
        .take_while(|c| c.is_alphabetic() || *c == '\'')
        .collect();
    Some(word.to_lowercase())
}

/// True when `text[word_end..]` — the text immediately after "that" itself,
/// *before* `marker_regex`'s trailing `[ \t]*,?` has had a chance to absorb
/// anything — is shaped like the boundary after a genuine self-correction
/// interjection rather than an ordinary clause continuing past "that": empty
/// or whitespace-only (end of utterance), starts with `,` (a comma really
/// did follow "that" in the source text), starts with whitespace then an
/// uppercase letter (shaped like the start of a new sentence), or another
/// marker occurs immediately next (covers "scratch that, scratch that,"
/// with no intervening content). Mirrors `previous_boundary`'s role but as
/// a yes/no gate rather than an index, since the trailing side only ever
/// needs to decide whether to trust the match — the deletion span itself is
/// unchanged either way.
///
/// Must be called with `word_end` = capture group 1's end (right after
/// "that"), not the full match's end: once the full match has absorbed a
/// trailing comma into itself, that comma is no longer present in
/// `text[full_match_end..]` for this function to see, which would make an
/// absorbed-comma marker like "Scratch that, use the letter." (full match
/// `"Scratch that,"`) indistinguishable from a genuinely comma-less one.
fn has_trailing_boundary(text: &str, word_end: usize) -> bool {
    let tail = &text[word_end..];
    let trimmed = tail.trim_start_matches([' ', '\t']);
    if trimmed.is_empty() || trimmed.starts_with(',') {
        return true;
    }
    if trimmed.starts_with(|c: char| c.is_uppercase()) {
        return true;
    }
    // Back-to-back marker: allow an already-unabsorbed leading comma before
    // it too (e.g. the second marker in "scratch that, scratch that, ...").
    let after_comma = trimmed.trim_start_matches(',').trim_start_matches([' ', '\t']);
    bare_marker_regex()
        .find(after_comma)
        .is_some_and(|m| m.start() == 0)
}

/// Fallback for a comma-less marker whose trailing text does *not* look
/// sentence-boundary-shaped (`has_trailing_boundary` returned `false`):
/// still treat it as genuine if the word immediately after "that"
/// case-insensitively repeats the first word of the marker's own containing
/// clause (per `previous_boundary`). Real self-corrections overwhelmingly
/// restate the same leading verb while replacing a trailing detail ("Call
/// them now, scratch that, call them later" / "Send it Monday, scratch
/// that, send it Tuesday") — exactly the shape STT punctuation-dropping
/// produces when the comma is lost but the words are transcribed correctly.
/// An ordinary sentence using "scratch/strike that" as a verb phrase
/// ("strike that clause from the contract", "scratch that itch") has no
/// reason to echo the clause's own opening word right after "that", so this
/// check does not reopen the over-matching this module exists to prevent —
/// it only ever narrows an already-conservative marker pattern further, on
/// a small, bounded, exact-word-equality condition rather than an
/// open-ended vocabulary/grammar heuristic.
///
/// `word_end` is the same "right after that, pre-absorption" position
/// `has_trailing_boundary` uses; `marker_start` is the full candidate
/// match's start (where its own optional leading comma/space may begin).
fn is_repeated_clause_verb(text: &str, marker_start: usize, word_end: usize) -> bool {
    let clause_start = previous_boundary(text, marker_start);
    let Some(clause_word) = first_word_lowercased(text, clause_start) else {
        return false;
    };
    // Guard against the marker matching its own leading word when it sits at
    // the very start of the string (clause_start == marker_start == 0, so
    // "clause_start" would otherwise point at the marker's own "scratch"/
    // "strike" rather than a real preceding clause).
    if clause_start >= marker_start {
        return false;
    }
    first_word_lowercased(text, word_end).as_deref() == Some(clause_word.as_str())
}

/// True when a candidate `marker_regex` match should actually be treated as
/// a self-correction marker: either its trailing context is
/// sentence-boundary-shaped (`has_trailing_boundary`), or — for the
/// comma-dropped case that boundary check alone cannot cover — the word
/// right after it echoes its own clause's opening word
/// (`is_repeated_clause_verb`). Both checks only ever *reject* a candidate
/// match; neither can cause a match `marker_regex` didn't already find, so
/// this function can only make `apply_scratch_that` more conservative than
/// the bare regex, never less.
///
/// `marker_start` is the full match's start; `word_end` is capture group
/// 1's end (right after "that", before trailing-comma absorption) — see
/// `has_trailing_boundary`'s doc comment for why that distinction matters.
fn is_genuine_marker(text: &str, marker_start: usize, word_end: usize) -> bool {
    has_trailing_boundary(text, word_end) || is_repeated_clause_verb(text, marker_start, word_end)
}

/// Capitalize the first alphabetic character found while skipping any
/// leading whitespace in `s`, leaving that whitespace and everything else
/// untouched. Used at a splice point, which `previous_boundary` guarantees
/// is always either the start of the utterance or immediately after a
/// sentence-ending mark — i.e. always a sentence-initial position, so
/// lowercase there (an artifact of the deleted clause reading naturally
/// mid-sentence before the splice) must be corrected the same way
/// `corrections::apply`'s `sentence_initial` case does.
fn capitalize_after_leading_whitespace(s: &str) -> String {
    let Some(idx) = s.find(|c: char| !c.is_whitespace()) else {
        return s.to_string();
    };
    let (lead, rest) = s.split_at(idx);
    format!("{lead}{}", capitalize_first(rest))
}

/// Drop each "scratch that" / "strike that" self-correction and the clause
/// it abandons: for every *genuine* marker (see `is_genuine_marker`;
/// candidate matches that read as ordinary sentence content rather than a
/// self-correction interjection are left untouched), processed left to
/// right and iteratively so multiple corrections in one utterance all
/// apply, delete from the start of its containing sentence through the
/// marker plus its immediately surrounding comma/space, then fix up
/// capitalization at the splice.
pub fn apply_scratch_that(text: &str) -> String {
    let re = marker_regex();
    let mut out = text.to_string();
    loop {
        // `captures_iter` (not a single `find`) for two reasons: (1) the
        // first candidate match isn't necessarily genuine — e.g. in "We
        // need to strike that clause from the contract, scratch that, use
        // the old one." the first "strike that" is ordinary content and
        // must be skipped in favor of the real "scratch that" marker later
        // in the string, rather than stopping the whole pass just because
        // *a* match exists somewhere; (2) `is_genuine_marker` needs group
        // 1's end (right after "that", before trailing-comma absorption),
        // which a plain `find` over group 0 can't expose.
        let found = re.captures_iter(&out).find_map(|caps| {
            let whole = caps.get(0).expect("group 0 always present");
            let word = caps.get(1).expect("group 1 always present (not optional)");
            is_genuine_marker(&out, whole.start(), word.end())
                .then(|| (whole.start(), whole.end()))
        });
        let Some((match_start, match_end)) = found else {
            break;
        };
        let boundary = previous_boundary(&out, match_start);
        let prefix = &out[..boundary];
        let suffix = &out[match_end..];
        let spliced = format!("{prefix}{}", capitalize_after_leading_whitespace(suffix));
        // Only ever bites when boundary == 0 (deletion reached the true
        // start of the utterance): the marker's own leading whitespace was
        // already consumed by the regex, so a lone leading space can only
        // appear here from the suffix side of a start-of-string splice.
        out = spliced.trim_start().to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_that_drops_previous_clause() {
        assert_eq!(
            apply_scratch_that("Send it Monday, scratch that, send it Tuesday."),
            "Send it Tuesday."
        );
        assert_eq!(
            apply_scratch_that("The panel meets Friday. Email Kumar, strike that, email Andres."),
            "The panel meets Friday. Email Andres."
        );
        assert_eq!(apply_scratch_that("No markers here."), "No markers here.");
        // Marker at start: nothing before it to drop — just remove the marker.
        assert_eq!(
            apply_scratch_that("Scratch that, use the new letter."),
            "Use the new letter."
        );
    }

    /// The brief's own test never exercises two markers in one utterance,
    /// but the Interfaces line explicitly requires "Iterative (multiple
    /// markers)" behavior. Three clauses stacked with two scratch-markers
    /// and no terminal punctuation anywhere: each marker should collapse
    /// back to the utterance start (boundary stays 0 every time, since
    /// nothing in this run-on sentence is `.!?\n`), leaving only the final,
    /// uncorrected clause.
    #[test]
    fn scratch_that_handles_multiple_markers_in_one_utterance() {
        assert_eq!(
            apply_scratch_that(
                "Call Sam Monday, scratch that, call Sam Tuesday, scratch that, call Sam Wednesday."
            ),
            "Call Sam Wednesday."
        );
    }

    /// A second multi-marker shape: markers separated by a real sentence
    /// boundary, so each deletion's boundary differs (0, then mid-string
    /// after a period) rather than collapsing to the same spot twice —
    /// exercises `previous_boundary` recomputing correctly on the
    /// already-spliced string, not just re-finding the same index.
    #[test]
    fn scratch_that_handles_markers_across_separate_sentences() {
        assert_eq!(
            apply_scratch_that(
                "Book Kumar, scratch that, book Andres. Send Monday, strike that, send Tuesday."
            ),
            "Book Andres. Send Tuesday."
        );
    }

    #[test]
    fn scratch_that_is_case_insensitive_and_handles_strike_variant() {
        assert_eq!(
            apply_scratch_that("Use plan A, SCRATCH THAT, use plan B."),
            "Use plan B."
        );
        assert_eq!(
            apply_scratch_that("Use plan A, Strike That, use plan B."),
            "Use plan B."
        );
    }

    #[test]
    fn scratch_that_without_surrounding_commas_still_matches() {
        // STT punctuation is inconsistent; the marker itself (not the
        // commas) is what must be exact.
        assert_eq!(
            apply_scratch_that("Call them now scratch that call them later."),
            "Call them later."
        );
    }

    #[test]
    fn scratch_that_does_not_fire_on_paraphrased_corrections() {
        // Conservative by design: "never mind" / "delete that" are out of
        // scope per the brief. A false-positive deletion here would
        // silently destroy real dictated content.
        assert_eq!(
            apply_scratch_that("Send it Monday, never mind, send it Tuesday."),
            "Send it Monday, never mind, send it Tuesday."
        );
        assert_eq!(
            apply_scratch_that("Send it Monday, delete that, send it Tuesday."),
            "Send it Monday, delete that, send it Tuesday."
        );
    }

    /// Task E2 review finding (Critical): the pre-fix regex had no
    /// constraint on what follows "that", so it fired on ordinary sentences
    /// where "scratch/strike that" is an ordinary verb + demonstrative-pronoun
    /// phrase rather than a self-correction interjection — silently
    /// destroying real dictated content (exactly the risk the module's own
    /// doc comment already warns about). Each case here is comma-less on
    /// both sides of the marker, so `has_trailing_boundary` alone rejects
    /// it (no comma/EOS/capital-letter/back-to-back-marker follows "that"),
    /// and `is_repeated_clause_verb` also correctly declines to rescue it
    /// (the word right after "that" doesn't echo the clause's opening word).
    #[test]
    fn scratch_that_does_not_fire_on_ordinary_verb_object_usage() {
        // The review's own exact repro: "strike that clause" is a normal
        // verb phrase ("strike [that clause] from the contract"), not a
        // self-correction marker. Must return the input unchanged.
        assert_eq!(
            apply_scratch_that("We need to strike that clause from the contract."),
            "We need to strike that clause from the contract."
        );
        // "Scratch that itch" — an even more literal false positive for the
        // word "scratch": a bare imperative verb + demonstrative pronoun +
        // noun object, sentence-initial (no leading clause at all), no
        // comma anywhere. `has_trailing_boundary` rejects on the trailing
        // side (" itch." is lowercase, no comma); there is no leading
        // clause for `is_repeated_clause_verb` to echo against either.
        assert_eq!(apply_scratch_that("Scratch that itch."), "Scratch that itch.");
        // "I'll strike that off my list" — comma-less, "that" as object of
        // "strike" again, this time followed by a preposition rather than a
        // noun. Confirms the fix isn't narrowly tuned to only reject
        // noun-shaped continuations.
        assert_eq!(
            apply_scratch_that("I'll strike that off my list."),
            "I'll strike that off my list."
        );
    }

    /// A false positive can appear earlier in an utterance than a genuine
    /// marker later in the *same* utterance — `apply_scratch_that` must
    /// skip the ordinary-usage candidate and still act on the real one,
    /// not stop (or misfire) just because `marker_regex` found *a* match
    /// first. Exercises the `captures_iter().find_map()` skip-and-continue
    /// path, not just the single-candidate cases the other tests cover.
    #[test]
    fn scratch_that_skips_ordinary_usage_and_still_fires_on_a_later_real_marker() {
        assert_eq!(
            apply_scratch_that(
                "We need to strike that clause from the contract, scratch that, use the old one."
            ),
            "Use the old one."
        );
    }

    /// `is_repeated_clause_verb`'s echo fallback exists specifically for
    /// comma-less markers whose trailing word repeats the containing
    /// clause's own opening word (STT dropped the comma but transcribed the
    /// words correctly) — confirms this generalizes past the one verb
    /// ("call") the pre-existing
    /// `scratch_that_without_surrounding_commas_still_matches` test uses,
    /// so the fix isn't coincidentally keyed to that single word.
    #[test]
    fn scratch_that_echo_fallback_generalizes_to_other_repeated_verbs() {
        assert_eq!(
            apply_scratch_that("Email Priya today scratch that email Priya tomorrow."),
            "Email Priya tomorrow."
        );
    }

    /// Regression guard for `is_repeated_clause_verb`'s
    /// `clause_start >= marker_start` check: a marker sitting at the very
    /// start of the string has no real preceding clause, so
    /// `previous_boundary` returns 0 — the same value as `marker_start`
    /// itself. Without the guard, `first_word_lowercased` at that shared
    /// index would read the marker's *own* verb ("scratch") as the "clause
    /// word", which would then spuriously match a same-word continuation
    /// right after the marker and rescue a comma-less, trailing-boundary-less
    /// match that is not a genuine self-correction (there is no earlier
    /// clause to be correcting).
    #[test]
    fn scratch_that_at_string_start_does_not_echo_its_own_verb() {
        assert_eq!(
            apply_scratch_that("Scratch that scratch pad before you leave."),
            "Scratch that scratch pad before you leave."
        );
    }

    #[test]
    fn scratch_that_empty_is_noop() {
        // No marker present ⇒ no splice ever happens, so trim_start (which
        // only runs as part of a splice) never touches the input — matches
        // every other "no markers here" case: input returned verbatim.
        assert_eq!(apply_scratch_that(""), "");
        assert_eq!(apply_scratch_that("   "), "   ");
    }
}
