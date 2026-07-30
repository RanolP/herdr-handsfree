//! Speech-to-text. The default backend is Qwen3-ASR on candle (pure Rust,
//! Metal on macOS); `--features whisper` swaps in whisper.cpp, which needs
//! cmake and a C++ toolchain.

#[cfg(not(feature = "whisper"))]
mod qwen;
#[cfg(feature = "whisper")]
mod whisper;

#[cfg(not(feature = "whisper"))]
pub use qwen::{Stream, Stt};
#[cfg(feature = "whisper")]
pub use whisper::{Stream, Stt};

impl Stt {
    /// Load once and keep the engine for the process lifetime. `Dictation` is
    /// dropped on every toggle-off, so without this each toggle-on pays the
    /// full load again (~1.6 s warm for 1.7B, far worse cold). The weights then
    /// stay resident once you dictate at all (~4.5 GB), which is also why nothing
    /// pre-warms this at daemon boot: a gaze-only user never pays for them.
    ///
    /// The failure is stringified because `io::Error` is not `Clone` and every
    /// later caller has to be handed the same one.
    pub fn shared() -> std::io::Result<&'static Stt> {
        static ENGINE: std::sync::OnceLock<Result<Stt, String>> = std::sync::OnceLock::new();
        match ENGINE.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let loaded = Stt::load().map_err(|e| e.to_string());
            // Logged once per process: a second line here means the cache broke.
            eprintln!("STT model loaded in {:.1?}", t0.elapsed());
            loaded
        }) {
            Ok(stt) => Ok(stt),
            Err(e) => Err(std::io::Error::other(e.clone())),
        }
    }
}
