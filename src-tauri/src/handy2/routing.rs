//! Pure hotword router (ported from Handy 2.0 v1). Case-insensitive,
//! prefix-only: a hotword mid-sentence is content, not a command.

use crate::settings::Route;

#[derive(Debug, PartialEq)]
pub struct RouteDecision<'a> {
    pub route: &'a Route,
    pub cleaned_text: String,
}

/// First prefix match wins; falls back to the `trigger: None` default route.
pub fn route<'a>(raw_text: &str, routes: &'a [Route]) -> RouteDecision<'a> {
    let trimmed = raw_text.trim_start();
    let lower = trimmed.to_lowercase();

    for rc in routes {
        if let Some(trigger) = &rc.trigger {
            let t = trigger.to_lowercase();
            if lower.starts_with(&t) {
                let rest = &trimmed[trigger.len().min(trimmed.len())..];
                let cleaned = rest
                    .trim_start_matches([',', '.', ':', ';'])
                    .trim()
                    .to_string();
                return RouteDecision {
                    route: rc,
                    cleaned_text: cleaned,
                };
            }
        }
    }

    let default = routes
        .iter()
        .find(|r| r.trigger.is_none())
        .expect("config invariant: route table always contains a default route");
    RouteDecision {
        route: default,
        cleaned_text: raw_text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::default_h2_routes;

    #[test]
    fn standard_route_when_no_hotword() {
        let r = default_h2_routes();
        let d = route("um so the panel needs three new appraisers", &r);
        assert_eq!(d.route.id, "standard");
        assert_eq!(d.cleaned_text, "um so the panel needs three new appraisers");
    }

    #[test]
    fn polish_route_strips_prefix() {
        let r = default_h2_routes();
        let d = route("Polish command, draft a note to the credit team", &r);
        assert_eq!(d.route.id, "polish");
        assert_eq!(d.cleaned_text, "draft a note to the credit team");
    }

    #[test]
    fn prompt_engineering_route_matches() {
        let r = default_h2_routes();
        let d = route("Prompt engineering command build a React settings page", &r);
        assert_eq!(d.route.id, "prompt_engineering");
        assert_eq!(d.cleaned_text, "build a React settings page");
    }

    #[test]
    fn matching_is_case_insensitive() {
        let r = default_h2_routes();
        let d = route("polish COMMAND send this to underwriting", &r);
        assert_eq!(d.route.id, "polish");
    }

    #[test]
    fn hotword_mid_sentence_is_content() {
        let r = default_h2_routes();
        let d = route("I told them the Polish command thing was a feature", &r);
        assert_eq!(d.route.id, "standard");
    }

    #[test]
    fn hotword_only_yields_empty_cleaned_text() {
        let r = default_h2_routes();
        let d = route("Polish command", &r);
        assert_eq!(d.route.id, "polish");
        assert!(d.cleaned_text.is_empty());
    }
}
