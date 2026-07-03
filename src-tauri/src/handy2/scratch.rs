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
//! `has_trailing_boundary` before it is treated as real; see that function
//! for the trailing-boundary conditions it checks.
//!
//! Deliberately does *not* attempt to rescue comma-less markers whose
//! trailing text lacks a clean boundary (e.g. "Send it Monday scratch that
//! deliver it Tuesday.") by inferring intent from surrounding words. An
//! earlier revision tried exactly that (an "echo the containing clause's
//! opening verb" heuristic) and it backfired: it both introduced a new
//! false-positive deletion on ordinary text and still failed to fire on the
//! most realistic comma-less self-correction shape (a changed verb). A
//! missed marker — the words are left in the output, unrecognized — is
//! always preferable to a wrong deletion, so a comma-less marker with no
//! other trailing boundary signal is left alone rather than guessed at.

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
/// every result from it is a *candidate* match that `has_trailing_boundary`
/// must still confirm before `apply_scratch_that` acts on it.
///
/// Capture group 1 ends immediately after "that", *before* the trailing
/// `[ \t]*,?` absorption — `has_trailing_boundary`'s check needs to know
/// whether a comma actually followed "that" in the original text, which the
/// full match's end (`m.end()`, group 0) can no longer tell it once that
/// comma has been absorbed into the match itself.
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
/// condition in `has_trailing_boundary`), where absorbing surrounding
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
/// it abandons: for every *genuine* marker (see `has_trailing_boundary`;
/// candidate matches that read as ordinary sentence content rather than a
/// self-correction interjection — including any comma-less marker whose
/// trailing text lacks a clean boundary — are left untouched), processed
/// left to right and iteratively so multiple corrections in one utterance
/// all apply, delete from the start of its containing sentence through the
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
        // *a* match exists somewhere; (2) `has_trailing_boundary` needs
        // group 1's end (right after "that", before trailing-comma
        // absorption), which a plain `find` over group 0 can't expose.
        let found = re.captures_iter(&out).find_map(|caps| {
            let whole = caps.get(0).expect("group 0 always present");
            let word = caps.get(1).expect("group 1 always present (not optional)");
            has_trailing_boundary(&out, word.end()).then(|| (whole.start(), whole.end()))
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

    /// Task E2 re-review finding: this test previously asserted that a
    /// comma-less marker with a lowercase continuation ("Call them now
    /// scratch that call them later.") still matched, via a since-removed
    /// echo heuristic (`is_repeated_clause_verb`) that inferred intent from
    /// the trailing word repeating the clause's own opening word. That
    /// heuristic was overfit: it introduced a new false-positive deletion on
    /// ordinary text ("Strike the tent then strike that strike the pegs
    /// firmly." → wrongly destroyed "Strike the tent then") and still failed
    /// to fire on realistic comma-less verb-changing corrections. Removing
    /// it makes this exact input genuinely ambiguous — identical shape to
    /// the "strike that clause from the contract" false positive, no comma,
    /// no capital-letter continuation, no other trailing-boundary signal —
    /// so it is now correctly left unrecognized. Kept (not deleted) as an
    /// explicit "considered and intentionally not treated as a marker" case
    /// per the module's false-negative-over-false-positive doctrine, rather
    /// than being silently forgotten.
    #[test]
    fn scratch_that_without_any_trailing_boundary_signal_is_not_recognized() {
        assert_eq!(
            apply_scratch_that("Call them now scratch that call them later."),
            "Call them now scratch that call them later."
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
    /// both sides of the marker, so `has_trailing_boundary` rejects it (no
    /// comma/EOS/capital-letter/back-to-back-marker follows "that").
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
        // side (" itch." is lowercase, no comma).
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

    /// Task E2 re-review finding: previously verified that the since-removed
    /// `is_repeated_clause_verb` echo fallback generalized past the single
    /// verb ("call") the sibling
    /// `scratch_that_without_any_trailing_boundary_signal_is_not_recognized`
    /// test uses. With that fallback gone, this comma-less, lowercase-
    /// continuation input ("email" after "that") has the same "no trailing
    /// boundary signal" shape as every other ambiguous case in this module
    /// and is now correctly left unrecognized rather than guessed at. Kept
    /// as an explicit second data point (different repeated verb) for the
    /// same "ambiguous, don't touch" behavior, not deleted.
    #[test]
    fn scratch_that_without_trailing_boundary_generalizes_across_repeated_verbs() {
        assert_eq!(
            apply_scratch_that("Email Priya today scratch that email Priya tomorrow."),
            "Email Priya today scratch that email Priya tomorrow."
        );
    }

    /// A marker at the very start of the string, immediately followed by a
    /// lowercase word that happens to repeat the marker's own leading verb
    /// ("Scratch that scratch pad..."): comma-less, no capital-letter
    /// continuation, no EOS, no back-to-back marker, so `has_trailing_boundary`
    /// rejects it like any other ambiguous comma-less case. Kept as a
    /// dedicated regression case for this specific shape (marker's own verb
    /// echoed immediately after it, at string start) since it was previously
    /// the input a since-removed echo heuristic needed a special guard to
    /// avoid misfiring on.
    #[test]
    fn scratch_that_at_string_start_does_not_echo_its_own_verb() {
        assert_eq!(
            apply_scratch_that("Scratch that scratch pad before you leave."),
            "Scratch that scratch pad before you leave."
        );
    }

    /// Task E2 re-review finding (new false positive introduced by the since-
    /// removed `is_repeated_clause_verb` echo fallback): the containing
    /// clause opens with "Strike" and the word immediately after the second
    /// "that" is "strike", so the old echo heuristic matched on the clause's
    /// opening word alone — with no regard for grammatical role — and
    /// wrongly rescued this as a marker, silently deleting "Strike the tent
    /// then" from the output. This is ordinary text: "strike that" here
    /// means "hit that [strike the pegs]", not a self-correction. With the
    /// echo heuristic removed, this comma-less, lowercase-continuation
    /// candidate now correctly fails `has_trailing_boundary` (no comma, no
    /// capital letter, no EOS, no back-to-back marker after "that") and the
    /// whole sentence is left unchanged.
    #[test]
    fn scratch_that_does_not_echo_match_an_unrelated_repeated_verb() {
        assert_eq!(
            apply_scratch_that("Strike the tent then strike that strike the pegs firmly."),
            "Strike the tent then strike that strike the pegs firmly."
        );
    }

    /// Second data point for the same since-removed echo-heuristic false
    /// positive, with "scratch" instead of "strike" as the repeated word —
    /// confirms the fix isn't narrowly tuned to only one of the two marker
    /// verbs. "Scratch that scratch pad" is again ordinary text ("scratch
    /// [that scratch pad]"), not a self-correction; must return unchanged.
    #[test]
    fn scratch_that_does_not_echo_match_an_unrelated_repeated_noun_modifier() {
        assert_eq!(
            apply_scratch_that("Scratch the itch then scratch that scratch pad quickly."),
            "Scratch the itch then scratch that scratch pad quickly."
        );
    }

    /// Documents an accepted trade-off, not a bug to chase: a comma-less
    /// self-correction that changes the clause's *verb* ("Send" →
    /// "deliver") has no trailing-boundary signal for `has_trailing_boundary`
    /// to key off (no comma, no capital-letter continuation, no EOS, no
    /// back-to-back marker after "that") and is therefore left uncorrected —
    /// both clauses and the marker text remain in the output verbatim. This
    /// is the direct cost of removing the echo heuristic: that heuristic
    /// existed to catch shapes like this, but it did so unsafely (see
    /// `scratch_that_does_not_echo_match_an_unrelated_repeated_verb` above)
    /// and, per this exact case, still didn't even succeed at the goal — a
    /// changed verb never echoes the clause's opening word, so the old
    /// heuristic would have left this uncorrected too. Per the module's
    /// documented doctrine (see module doc comment), a missed marker is
    /// always preferable to a wrong deletion, so this remains unrecognized
    /// rather than guessed at.
    #[test]
    fn scratch_that_leaves_comma_less_verb_changing_corrections_uncorrected() {
        assert_eq!(
            apply_scratch_that("Send it Monday scratch that deliver it Tuesday."),
            "Send it Monday scratch that deliver it Tuesday."
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
