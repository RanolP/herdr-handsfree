//! User config: `config.toml` in HERDR_PLUGIN_CONFIG_DIR. Every field is
//! optional; env vars HANDSFREE_MODEL / HANDSFREE_VAD override the file.

use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Qwen3-ASR model size: 0.6B | 1.7B
    pub asr_model: String,
    /// Whisper ggml model size, only used with `--features whisper`:
    /// tiny | base | small | medium | large-v3
    pub whisper_model: String,
    /// Language hint: "auto", a language name like "korean", or an ISO 639-1
    /// code like "ko"
    pub language: String,
    /// Voice activity detector: "silero" | "energy"
    pub vad: String,
    /// Input device to capture from; case-insensitive substring of the device
    /// name. Empty = system default input.
    pub microphone: String,
    /// One-Euro filter: lower = smoother but laggier cursor
    pub smoothing_min_cutoff: f64,
    /// One-Euro filter: higher = snappier during fast gaze moves
    pub smoothing_beta: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            asr_model: "1.7B".to_string(),
            whisper_model: "small".to_string(),
            language: "auto".to_string(),
            vad: "silero".to_string(),
            microphone: String::new(),
            smoothing_min_cutoff: 1.0,
            smoothing_beta: 0.01,
        }
    }
}

pub fn get() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let dir = match std::env::var_os("HERDR_PLUGIN_CONFIG_DIR") {
            Some(d) => std::path::PathBuf::from(d),
            None => {
                let home = std::env::var_os("HOME").expect("HOME not set");
                std::path::PathBuf::from(home).join(".config/herdr-handsfree")
            }
        };
        let path = dir.join("config.toml");
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).unwrap_or_else(|e| {
                eprintln!("invalid {} ({e}); using defaults", path.display());
                Config::default()
            }),
            Err(_) => Config::default(),
        }
    })
}
