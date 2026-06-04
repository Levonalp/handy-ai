//! Ollama API key in the OS credential store (Handy 2.0 v1 ADR-001). Only
//! module that touches `keyring`.

use keyring::Entry;

const SERVICE: &str = "com.levon.handy-ai";
const USERNAME: &str = "ollama_api_key";

fn entry() -> Result<Entry, String> {
    Entry::new(SERVICE, USERNAME).map_err(|e| e.to_string())
}

pub fn set_key(key: &str) -> Result<(), String> {
    let key = key.trim();
    if key.is_empty() {
        return Err("API key cannot be empty.".into());
    }
    entry()?.set_password(key).map_err(|e| e.to_string())
}

pub fn get_key() -> Result<String, String> {
    match entry()?.get_password() {
        Ok(k) => Ok(k),
        Err(keyring::Error::NoEntry) => Err("No Ollama API key saved.".into()),
        Err(e) => Err(e.to_string()),
    }
}

pub fn has_key() -> Result<bool, String> {
    match entry()?.get_password() {
        Ok(_) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(e) => Err(e.to_string()),
    }
}

pub fn delete_key() -> Result<(), String> {
    match entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
