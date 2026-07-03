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

use crate::handy2::corrections::capitalize_first;
use regex::Regex;
use std::sync::OnceLock;

/// Matches one marker occurrence together with the punctuation immediately
/// around it that should be deleted along with it: an optional leading
/// `, ` and an optional trailing `,`. `[ \t]*` (not `\s*`) deliberately
/// excludes `\n` from the punctuation the marker itself can absorb — a
/// newline is a sentence boundary in its own right (see
/// `previous_boundary`), not incidental whitespace around the marker.
fn marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i),?[ \t]*(?:scratch|strike)[ \t]+that[ \t]*,?")
            .expect("static scratch-marker pattern is valid regex")
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
/// it abandons: for every marker (processed left to right, iteratively, so
/// multiple corrections in one utterance all apply), delete from the start
/// of its containing sentence through the marker plus its immediately
/// surrounding comma/space, then fix up capitalization at the splice.
pub fn apply_scratch_that(text: &str) -> String {
    let re = marker_regex();
    let mut out = text.to_string();
    while let Some(m) = re.find(&out) {
        let boundary = previous_boundary(&out, m.start());
        let prefix = &out[..boundary];
        let suffix = &out[m.end()..];
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

    #[test]
    fn scratch_that_empty_is_noop() {
        // No marker present ⇒ no splice ever happens, so trim_start (which
        // only runs as part of a splice) never touches the input — matches
        // every other "no markers here" case: input returned verbatim.
        assert_eq!(apply_scratch_that(""), "");
        assert_eq!(apply_scratch_that("   "), "   ");
    }
}
