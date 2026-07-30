//! Qwen3-ASR transcription (candle, Metal on macOS).

use qwen3_asr::{AsrInference, StreamingOptions, StreamingState, TranscribeOptions, best_device};

pub struct Stt {
    engine: AsrInference,
    language: Option<String>,
}

impl Stt {
    /// Load the model (downloads from HuggingFace on first run).
    pub fn load() -> std::io::Result<Self> {
        let model_id = crate::models::asr_model_id();
        let cache = crate::models::asr_cache_dir()?;
        let engine =
            AsrInference::from_pretrained(&model_id, &cache, best_device()).map_err(|e| {
                std::io::Error::other(format!("qwen3-asr load failed ({model_id}): {e}"))
            })?;
        Ok(Self {
            engine,
            language: language_hint(&crate::config::get().language),
        })
    }

    /// Transcribe mono 16 kHz f32 samples; returns trimmed text ("" for silence).
    pub fn transcribe(&self, samples_16k: &[f32]) -> std::io::Result<String> {
        let mut opts = TranscribeOptions::default();
        if let Some(lang) = &self.language {
            opts = opts.with_language(lang);
        }
        let result = self
            .engine
            .transcribe_samples(samples_16k, opts)
            .map_err(|e| std::io::Error::other(format!("qwen3-asr transcribe: {e}")))?;
        Ok(strip_sep(&result.text))
    }

    /// Incremental transcription of an utterance still being spoken.
    pub fn stream(&self) -> Stream<'_> {
        let mut opts = StreamingOptions::default();
        if let Some(lang) = &self.language {
            opts = opts.with_language(lang);
        }
        Stream {
            engine: &self.engine,
            state: self.engine.init_streaming(opts),
        }
    }
}

pub struct Stream<'a> {
    engine: &'a AsrInference,
    state: StreamingState,
}

impl Stream<'_> {
    /// `feed_audio` buffers against `chunk_size_sec` internally, so chunks of
    /// any length are fine and inference only runs once one has filled.
    pub fn feed(&mut self, samples_16k: &[f32]) -> std::io::Result<()> {
        self.engine
            .feed_audio(&mut self.state, samples_16k)
            .map(|_| ())
            .map_err(|e| std::io::Error::other(format!("qwen3-asr stream feed: {e}")))
    }

    pub fn finish(mut self) -> std::io::Result<String> {
        let result = self
            .engine
            .finish_streaming(&mut self.state)
            .map_err(|e| std::io::Error::other(format!("qwen3-asr stream finish: {e}")))?;
        Ok(strip_sep(&result.text))
    }
}

/// qwen3-asr 0.2.2 only splits off the `<asr_text>` separator on the
/// auto-detect path; with a forced language it leaves the literal token in
/// `text`. Drop it so we never type it into a pane.
fn strip_sep(text: &str) -> String {
    let t = text.trim();
    t.strip_prefix("<asr_text>").unwrap_or(t).trim().to_string()
}

/// Qwen3-ASR wants a language *name* ("korean"), not whisper's ISO 639-1 code.
/// Accept both so an existing config keeps working; `None` = auto-detect.
fn language_hint(configured: &str) -> Option<String> {
    match configured.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "ko" => Some("korean".to_string()),
        "en" => Some("english".to_string()),
        "ja" => Some("japanese".to_string()),
        "zh" => Some("chinese".to_string()),
        name => Some(name.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::{language_hint, strip_sep};

    #[test]
    fn iso_codes_become_names_and_auto_becomes_none() {
        assert_eq!(language_hint("auto"), None);
        assert_eq!(language_hint("  "), None);
        assert_eq!(language_hint("KO"), Some("korean".to_string()));
        assert_eq!(language_hint("Korean"), Some("korean".to_string()));
    }

    #[test]
    fn forced_language_separator_is_dropped() {
        assert_eq!(strip_sep(" <asr_text>안녕하세요 "), "안녕하세요");
        assert_eq!(strip_sep("hello world"), "hello world");
    }
}
