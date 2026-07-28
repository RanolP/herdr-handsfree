//! Whisper transcription (whisper.cpp via whisper-rs, Metal on macOS).

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Stt {
    ctx: WhisperContext,
}

impl Stt {
    /// Load the ggml model (downloads on first run).
    pub fn load() -> std::io::Result<Self> {
        let model = crate::models::ensure_whisper_model()?;
        let ctx = WhisperContext::new_with_params(
            model.to_str().expect("model path is utf-8"),
            WhisperContextParameters::default(),
        )
        .map_err(|e| std::io::Error::other(format!("whisper load failed: {e}")))?;
        Ok(Self { ctx })
    }

    /// Transcribe mono 16 kHz f32 samples; returns trimmed text ("" for silence).
    pub fn transcribe(&self, samples_16k: &[f32]) -> std::io::Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| std::io::Error::other(format!("whisper state: {e}")))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let language = &crate::config::get().language;
        params.set_language(Some(language));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        state
            .full(params, samples_16k)
            .map_err(|e| std::io::Error::other(format!("whisper full: {e}")))?;

        let mut text = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(seg) = state.get_segment(i) {
                if let Ok(s) = seg.to_str_lossy() {
                    text.push_str(&s);
                }
            }
        }
        Ok(text.trim().to_string())
    }
}
