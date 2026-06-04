//! System prompt builder (Handy 2.0 v1 SPEC §7 template). Pure.

pub const MEMORY_MAX_BYTES: usize = 100 * 1024;

const BASE_TEMPLATE: &str = "You are an expert transcription formatter. Your job is to clean up spoken text while maintaining the exact core message.

RULES:
1. Remove filler words (ums, ahs) and verbal stumbles.
2. If the user dictates a sequence of items or steps, automatically format them as a clean Markdown list.
3. Ensure exact spelling and formatting based strictly on the custom user vocabulary and dictionary rules provided below.

USER CUSTOM VOCABULARY & RULES:
";

const NO_MEMORY_FALLBACK: &str =
    "(No custom vocabulary file is configured. Apply the rules above using standard spelling.)";

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
    let mut system_prompt = format!("{BASE_TEMPLATE}{memory_block}");
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
        assert!(p.system_prompt.contains("- VMS (Valuation Management System)"));
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
        assert!(p.system_prompt.contains("vocab\n\nRewrite professionally."));
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
}
