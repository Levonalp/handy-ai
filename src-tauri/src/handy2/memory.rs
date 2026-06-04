//! Fresh read of the Obsidian vocab file (Handy 2.0 v1 FR-4). Absent/unreadable
//! ⇒ base-prompt-only + warning; never blocks formatting.

use std::path::Path;

pub enum MemoryRead {
    Loaded(String),
    Absent { warning: Option<String> },
}

pub fn read(path: Option<&str>) -> MemoryRead {
    let Some(path) = path else {
        return MemoryRead::Absent { warning: None };
    };
    match std::fs::read_to_string(Path::new(path)) {
        Ok(c) => MemoryRead::Loaded(c),
        Err(e) => MemoryRead::Absent {
            warning: Some(format!(
                "Memory file unreadable ({path}): {e}. Proceeding without custom vocabulary."
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_absent_no_warning() {
        assert!(matches!(read(None), MemoryRead::Absent { warning: None }));
    }

    #[test]
    fn missing_is_absent_with_warning() {
        match read(Some("/no/such/file.md")) {
            MemoryRead::Absent { warning: Some(w) } => assert!(w.contains("unreadable")),
            _ => panic!("expected Absent+warning"),
        }
    }

    #[test]
    fn readable_loads() {
        let f = std::env::temp_dir().join("h2_mem_test.md");
        std::fs::write(&f, "# vocab\n- ARV").expect("write");
        assert!(matches!(read(f.to_str()), MemoryRead::Loaded(c) if c.contains("ARV")));
        let _ = std::fs::remove_file(&f);
    }
}
