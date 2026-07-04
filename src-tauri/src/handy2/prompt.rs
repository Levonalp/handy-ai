//! System prompt builder (Handy 2.0 v1 SPEC §7 template). Pure.

pub const MEMORY_MAX_BYTES: usize = 100 * 1024;

const BASE_TEMPLATE: &str = "You are an expert transcription formatter. Your job is to clean up spoken text while maintaining the exact core message.

RULES:
1. Remove filler words (ums, ahs) and verbal stumbles.
2. Auto-format lists without being asked. When the dictation is three or more parallel items, an explicit enumeration (\"first... second...\", \"one, two, three\"), a set of feedback points, or a series of fields or steps, format them as a clean Markdown list (one item per line) and drop connective filler like \"and then\", \"also\", or \"the next thing is\". Use \"- \" bullets for unordered items; use \"1. \" numbering only for explicitly ordered steps or rankings. Do NOT make a list when the input is only one or two items, a single sentence, narrative prose, or a short approval stamp — keep those as prose.
   Example input: \"tell the team we need three things, first the panel review, second the fee schedule, and third the new engagement letter\"
   Example output:
   We need three things:
   - Panel review
   - Fee schedule
   - New engagement letter
3. Ensure exact spelling and formatting based strictly on the custom user vocabulary and dictionary rules provided below.

USER CUSTOM VOCABULARY & RULES:
";

const NO_MEMORY_FALLBACK: &str =
    "(No custom vocabulary file is configured. Apply the rules above using standard spelling.)";

/// Small local instruct models default to answering a question-shaped
/// dictation conversationally, even when the ROLE section in the injected
/// memory explicitly forbids it (confirmed 2026-07-02: qwen2.5:3b-instruct
/// answered "Are you working?" with "I'm ready and waiting..." despite that
/// exact prohibition, in multiple prompt orderings) — a system-prompt-only
/// instruction isn't a strong enough override for the model's RLHF "be
/// helpful, answer questions" prior. A concrete example of the failure mode,
/// placed at the end of the system prompt (highest-recency context),
/// reliably corrects it. This is layer one of two — see
/// `handy2::wrap_for_reformat` for the user-turn reinforcement that closes
/// the remaining gap (list-formatting and other rules still apply normally
/// through both layers).
const ANTI_ANSWER_EXAMPLE: &str = "\n\nEXAMPLE (critical — follow exactly):\nDictation: \"Are you working?\"\nCorrect output: \"Are you working?\"\nWRONG output (do not do this): \"Yes, I am working.\" or any reply to the question.";

pub struct BuiltPrompt {
    pub system_prompt: String,
    pub truncated: bool,
}

pub fn build(memory: Option<&str>, addendum: Option<&str>) -> BuiltPrompt {
    let mut truncated = false;
    let memory_block = match memory {
        Some(m) if !m.trim().is_empty() => {
            if m.len() > MEMORY_MAX_BYTES {
                truncated = true;
                let mut end = MEMORY_MAX_BYTES;
                while end > 0 && !m.is_char_boundary(end) {
                    end -= 1;
                }
                &m[..end]
            } else {
                m
            }
        }
        _ => NO_MEMORY_FALLBACK,
    };
    let mut system_prompt = format!("{BASE_TEMPLATE}{memory_block}{ANTI_ANSWER_EXAMPLE}");
    if let Some(extra) = addendum {
        if !extra.trim().is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(extra);
        }
    }
    BuiltPrompt {
        system_prompt,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_memory_verbatim() {
        let p = build(Some("- VMS (Valuation Management System)"), None);
        assert!(p
            .system_prompt
            .contains("- VMS (Valuation Management System)"));
        assert!(!p.truncated);
    }

    #[test]
    fn missing_memory_uses_fallback() {
        let p = build(None, None);
        assert!(p.system_prompt.contains("No custom vocabulary file"));
    }

    #[test]
    fn addendum_appends_after_blank_line() {
        let p = build(Some("vocab"), Some("Rewrite professionally."));
        assert!(p.system_prompt.contains("vocab"));
        assert!(p.system_prompt.ends_with("\n\nRewrite professionally."));
    }

    #[test]
    fn anti_answer_example_present() {
        let p = build(None, None);
        assert!(p.system_prompt.contains("Are you working?"));
        assert!(p.system_prompt.contains("WRONG output"));
    }

    #[test]
    fn anti_answer_example_ordered_between_memory_and_addendum() {
        let p = build(Some("vocab-marker-xyz"), Some("addendum-marker-abc"));
        let vocab_pos = p.system_prompt.find("vocab-marker-xyz").unwrap();
        let example_pos = p.system_prompt.find("WRONG output").unwrap();
        let addendum_pos = p.system_prompt.find("addendum-marker-abc").unwrap();
        assert!(vocab_pos < example_pos);
        assert!(example_pos < addendum_pos);
    }

    #[test]
    fn oversized_memory_truncated() {
        let big = "x".repeat(MEMORY_MAX_BYTES + 500);
        let p = build(Some(&big), None);
        assert!(p.truncated);
    }

    #[test]
    fn truncation_respects_utf8() {
        let big = "é".repeat(MEMORY_MAX_BYTES);
        let p = build(Some(&big), None);
        assert!(p.truncated);
    }

    #[test]
    fn template_has_list_guidance() {
        let p = build(None, None);
        assert!(p.system_prompt.contains("Auto-format lists"));
        assert!(p.system_prompt.contains("- Panel review"));
    }
}
