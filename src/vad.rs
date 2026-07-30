//! Utterance segmentation over a mono 16 kHz stream. The per-frame voiced
//! decision comes from silero VAD when available, else an energy threshold.

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME: usize = crate::silero::CHUNK; // 512 samples = 32 ms
const FRAME_MS: usize = FRAME * 1000 / SAMPLE_RATE;
const PRE_ROLL_MS: usize = 300;
const HANGOVER_MS: usize = 700;
const MIN_SPEECH_MS: usize = 250;
const MAX_UTTERANCE_MS: usize = 30_000;
const RMS_THRESHOLD: f32 = 0.015;
const SILERO_THRESHOLD: f32 = 0.5;

enum Gate {
    Silero(crate::silero::Silero),
    Energy,
}

pub struct Vad {
    gate: Gate,
    frame: Vec<f32>,
    pre_roll: std::collections::VecDeque<Vec<f32>>,
    utterance: Vec<f32>,
    /// How much of `utterance` `drain_speech` has already handed out.
    emitted: usize,
    in_speech: bool,
    speech_ms: usize,
    silence_ms: usize,
}

impl Vad {
    /// Prefer silero (downloads the model on first use); fall back to the
    /// energy threshold when it cannot load. HANDSFREE_VAD=energy forces the
    /// fallback.
    pub fn new() -> Self {
        let choice =
            std::env::var("HANDSFREE_VAD").unwrap_or_else(|_| crate::config::get().vad.clone());
        let gate = if choice == "energy" {
            Gate::Energy
        } else {
            match crate::silero::Silero::load() {
                Ok(s) => Gate::Silero(s),
                Err(e) => {
                    eprintln!("silero VAD unavailable ({e}); using energy threshold");
                    Gate::Energy
                }
            }
        };
        Self {
            gate,
            frame: Vec::new(),
            pre_roll: std::collections::VecDeque::new(),
            utterance: Vec::new(),
            emitted: 0,
            in_speech: false,
            speech_ms: 0,
            silence_ms: 0,
        }
    }

    /// Feed 16 kHz samples; returns any utterance completed within this chunk.
    pub fn push(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let mut done = None;
        for &s in samples {
            self.frame.push(s);
            if self.frame.len() == FRAME {
                let frame = std::mem::take(&mut self.frame);
                if let Some(u) = self.push_frame(frame) {
                    done = Some(u);
                }
            }
        }
        done
    }

    fn is_voiced(&mut self, frame: &[f32]) -> bool {
        match &mut self.gate {
            Gate::Silero(s) => match s.prob(frame) {
                Ok(p) => p > SILERO_THRESHOLD,
                Err(e) => {
                    eprintln!("silero error ({e}); switching to energy threshold");
                    self.gate = Gate::Energy;
                    energy_voiced(frame)
                }
            },
            Gate::Energy => energy_voiced(frame),
        }
    }

    fn push_frame(&mut self, frame: Vec<f32>) -> Option<Vec<f32>> {
        let voiced = self.is_voiced(&frame);

        if !self.in_speech {
            if voiced {
                self.in_speech = true;
                self.speech_ms = FRAME_MS;
                self.silence_ms = 0;
                self.utterance = self.pre_roll.drain(..).flatten().collect();
                self.utterance.extend_from_slice(&frame);
                self.emitted = 0;
            } else {
                self.pre_roll.push_back(frame);
                while self.pre_roll.len() * FRAME_MS > PRE_ROLL_MS {
                    self.pre_roll.pop_front();
                }
            }
            return None;
        }

        self.utterance.extend_from_slice(&frame);
        if voiced {
            self.speech_ms += FRAME_MS;
            self.silence_ms = 0;
        } else {
            self.silence_ms += FRAME_MS;
        }

        let too_long = self.utterance.len() * 1000 / SAMPLE_RATE > MAX_UTTERANCE_MS;
        if self.silence_ms >= HANGOVER_MS || too_long {
            self.in_speech = false;
            if let Gate::Silero(s) = &mut self.gate {
                s.reset();
            }
            let utterance = std::mem::take(&mut self.utterance);
            self.emitted = 0;
            if self.speech_ms >= MIN_SPEECH_MS {
                return Some(utterance);
            }
        }
        None
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Samples of the utterance under way that no earlier call returned, so a
    /// caller can transcribe it while the speaker is still talking instead of
    /// waiting out the hangover.
    pub fn drain_speech(&mut self) -> Vec<f32> {
        let fresh = self.utterance[self.emitted..].to_vec();
        self.emitted = self.utterance.len();
        fresh
    }

    /// Flush a trailing utterance when the stream ends (for offline runs).
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        if self.in_speech && self.speech_ms >= MIN_SPEECH_MS {
            self.in_speech = false;
            self.emitted = 0;
            return Some(std::mem::take(&mut self.utterance));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{FRAME, Gate, Vad};

    fn energy_vad() -> Vad {
        Vad {
            gate: Gate::Energy,
            frame: Vec::new(),
            pre_roll: std::collections::VecDeque::new(),
            utterance: Vec::new(),
            emitted: 0,
            in_speech: false,
            speech_ms: 0,
            silence_ms: 0,
        }
    }

    /// Each drain hands over only what is new, and the pieces plus the tail left
    /// at completion reconstruct exactly the utterance the batch path would get.
    #[test]
    fn drained_pieces_plus_tail_reconstruct_the_utterance() {
        let mut vad = energy_vad();
        let loud = vec![0.5f32; FRAME];
        let quiet = vec![0.0f32; FRAME];

        let mut drained: Vec<f32> = Vec::new();
        let mut utterance = None;
        // 25 frames of speech (~800 ms, over MIN_SPEECH_MS), then silence well
        // past HANGOVER_MS so the utterance closes.
        for i in 0..80 {
            let frame = if i < 25 { &loud } else { &quiet };
            if let Some(u) = vad.push(frame) {
                utterance = Some(u);
                break;
            }
            if vad.in_speech() {
                drained.extend(vad.drain_speech());
            }
        }
        let utterance = utterance.expect("utterance should complete");
        assert!(!drained.is_empty(), "nothing was drained mid-utterance");
        assert!(drained.len() <= utterance.len());
        assert_eq!(drained, utterance[..drained.len()]);

        // The next utterance drains from its own start, not the old offset.
        assert_eq!(vad.drain_speech().len(), 0);
        vad.push(&loud);
        assert!(vad.in_speech());
        assert!(vad.drain_speech().len() >= FRAME);
    }
}

fn energy_voiced(frame: &[f32]) -> bool {
    let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
    rms > RMS_THRESHOLD
}

/// Streaming linear resampler to 16 kHz, carrying position across chunks.
pub struct Resampler {
    from_rate: usize,
    pos: f64,
    last: f32,
    have_last: bool,
}

impl Resampler {
    pub fn new(from_rate: usize) -> Self {
        Self {
            from_rate,
            pos: 0.0,
            last: 0.0,
            have_last: false,
        }
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.from_rate == SAMPLE_RATE {
            return samples.to_vec();
        }
        let step = self.from_rate as f64 / SAMPLE_RATE as f64;
        let mut out = Vec::with_capacity((samples.len() as f64 / step) as usize + 2);
        // Virtual input index 0 is `self.last` (previous chunk's tail).
        let ext_len = samples.len() + usize::from(self.have_last);
        let get = |i: usize| -> f32 {
            if self.have_last {
                if i == 0 { self.last } else { samples[i - 1] }
            } else {
                samples[i]
            }
        };
        while self.pos + 1.0 < ext_len as f64 {
            let idx = self.pos as usize;
            let frac = (self.pos - idx as f64) as f32;
            let a = get(idx);
            let b = get(idx + 1);
            out.push(a + (b - a) * frac);
            self.pos += step;
        }
        // Rebase position onto the next chunk, keeping the current tail.
        self.pos -= (ext_len - 1) as f64;
        if let Some(&tail) = samples.last() {
            self.last = tail;
            self.have_last = true;
        }
        out
    }
}
