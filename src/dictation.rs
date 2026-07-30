//! Dictation pipeline: mic (cpal) → energy VAD → STT → focused pane.

use crate::{stt::Stt, vad};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

pub struct Dictation {
    stop: Arc<AtomicBool>,
}

impl Dictation {
    /// Start mic capture + transcription. Returns after threads are spawned;
    /// model load happens on the worker thread so toggling stays instant.
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();

        // Capture thread: owns the cpal stream (not Send), forwards mono chunks.
        let stop_capture = stop.clone();
        std::thread::spawn(move || {
            if let Err(e) = capture_loop(&stop_capture, audio_tx) {
                eprintln!("dictation capture error: {e}");
            }
        });

        // Worker thread: VAD + STT + delivery.
        let stop_worker = stop.clone();
        std::thread::spawn(move || worker_loop(&stop_worker, audio_rx));

        Self { stop }
    }
}

impl Drop for Dictation {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// Case-insensitive substring match, so `microphone = "airpods"` picks
/// "Ranolp's AirPods Pro".
fn mic_matches(description: &str, wanted: &str) -> bool {
    description
        .to_lowercase()
        .contains(&wanted.trim().to_lowercase())
}

/// Config `microphone` (or HANDSFREE_MIC) selects the input device; empty or
/// unmatched falls back to the system default.
fn pick_device(host: &cpal::Host) -> Option<cpal::Device> {
    let wanted =
        std::env::var("HANDSFREE_MIC").unwrap_or_else(|_| crate::config::get().microphone.clone());
    if wanted.trim().is_empty() {
        return host.default_input_device();
    }
    let mut seen = Vec::new();
    for device in host.input_devices().ok()? {
        let name = device
            .description()
            .map(|d| d.to_string())
            .unwrap_or_default();
        if mic_matches(&name, &wanted) {
            return Some(device);
        }
        seen.push(name);
    }
    eprintln!(
        "microphone \"{wanted}\" not found; using default. available: {}",
        seen.join(", ")
    );
    host.default_input_device()
}

fn capture_loop(stop: &AtomicBool, tx: mpsc::Sender<Vec<f32>>) -> Result<(), String> {
    let host = cpal::default_host();
    let device = pick_device(&host)
        .ok_or("no input device — check macOS microphone permission for your terminal app")?;
    let config = device
        .default_input_config()
        .map_err(|e| format!("input config: {e}"))?;
    let sample_rate = config.sample_rate() as usize;
    let channels = config.channels() as usize;
    let mic_name = device
        .description()
        .map(|d| d.to_string())
        .unwrap_or_default();
    eprintln!("mic: {mic_name} @ {sample_rate} Hz, {channels}ch");
    // Tell the worker the sample rate via a header chunk.
    let _ = tx.send(vec![sample_rate as f32]);

    let stream = device
        .build_input_stream(
            config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix interleaved channels to mono.
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect();
                let _ = tx.send(mono);
            },
            |e| eprintln!("mic stream error: {e}"),
            None,
        )
        .map_err(|e| format!("build input stream: {e}"))?;
    stream.play().map_err(|e| format!("stream play: {e}"))?;

    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    drop(stream);
    eprintln!("mic capture stopped");
    Ok(())
}

fn worker_loop(stop: &AtomicBool, rx: mpsc::Receiver<Vec<f32>>) {
    // First chunk is the sample-rate header from the capture thread.
    let sample_rate = match rx.recv() {
        Ok(header) if header.len() == 1 => header[0] as usize,
        _ => return,
    };
    let stt = match Stt::shared() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("dictation disabled: {e}");
            return;
        }
    };
    eprintln!("STT ready; listening");

    let mut resampler = vad::Resampler::new(sample_rate);
    let mut vad = vad::Vad::new();
    // The utterance in progress is decoded as it arrives, so the wait after
    // end-of-speech is the VAD hangover rather than hangover + whole-utterance
    // inference. `fed` mirrors the VAD's emitted mark, so the tail between the
    // last drain and completion can still be handed over before finishing.
    let mut stream: Option<crate::stt::Stream<'static>> = None;
    let mut fed = 0usize;
    // Set when streaming broke partway through the current utterance: the live
    // stream is then missing audio, so the utterance is batched at completion.
    let mut fallback = false;
    while !stop.load(Ordering::SeqCst) {
        let chunk = match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let chunk_16k = resampler.push(&chunk);
        let done = vad.push(&chunk_16k);

        if let Some(utterance) = done {
            let text = match stream.take() {
                Some(mut s) if !fallback => s
                    .feed(&utterance[fed.min(utterance.len())..])
                    .and_then(|()| s.finish()),
                _ => stt.transcribe(&utterance),
            };
            fed = 0;
            fallback = false;
            deliver(text);
        } else if !vad.in_speech() {
            // Speech ended without an utterance (under MIN_SPEECH_MS).
            stream = None;
            fed = 0;
            fallback = false;
        }

        if vad.in_speech() && !fallback {
            let pending = vad.drain_speech();
            if !pending.is_empty() {
                let s = stream.get_or_insert_with(|| stt.stream());
                if let Err(e) = s.feed(&pending) {
                    eprintln!("streaming transcription failed ({e}); batching this utterance");
                    fallback = true;
                }
                fed += pending.len();
            }
        }
    }
}

fn deliver(text: std::io::Result<String>) {
    match text {
        Ok(text) if text.is_empty() => {}
        Ok(text) => {
            eprintln!("transcript: {text}");
            if let Err(e) = crate::deliver::deliver(&text) {
                eprintln!("delivery failed: {e}");
            }
        }
        Err(e) => eprintln!("transcription failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::mic_matches;

    #[test]
    fn matches_case_insensitive_substring() {
        assert!(mic_matches("Ranolp's AirPods Pro", " airpods "));
        assert!(!mic_matches("MacBook Pro Microphone", "airpods"));
    }
}
