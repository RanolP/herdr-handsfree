//! Dictation pipeline: mic (cpal) → energy VAD → whisper → focused pane.

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

        // Worker thread: VAD + whisper + delivery.
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

fn capture_loop(stop: &AtomicBool, tx: mpsc::Sender<Vec<f32>>) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
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
    let stt = match Stt::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("dictation disabled: {e}");
            return;
        }
    };
    eprintln!("whisper ready; listening");

    let mut resampler = vad::Resampler::new(sample_rate);
    let mut vad = vad::Vad::new();
    while !stop.load(Ordering::SeqCst) {
        let chunk = match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let chunk_16k = resampler.push(&chunk);
        if let Some(utterance) = vad.push(&chunk_16k) {
            transcribe_and_deliver(&stt, &utterance);
        }
    }
}

/// `utterance` is already mono 16 kHz.
fn transcribe_and_deliver(stt: &Stt, utterance: &[f32]) {
    match stt.transcribe(utterance) {
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
