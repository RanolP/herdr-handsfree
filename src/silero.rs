//! Silero VAD v5 (ONNX): speech probability per 512-sample chunk @ 16 kHz.

use ort::session::Session;
use ort::value::Tensor;

pub const CHUNK: usize = 512;
/// The model input is the previous chunk's 64-sample tail + the new chunk.
const CONTEXT: usize = 64;

pub struct Silero {
    session: Session,
    state: Vec<f32>,   // [2, 1, 128] LSTM state carried across chunks
    context: Vec<f32>, // last CONTEXT samples of the previous chunk
}

impl Silero {
    pub fn load() -> std::io::Result<Self> {
        let path = crate::models::ensure_silero_model()?;
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&path))
            .map_err(|e| std::io::Error::other(format!("silero load: {e}")))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * 128],
            context: vec![0.0; CONTEXT],
        })
    }

    /// Speech probability (0..1) for one 512-sample 16 kHz chunk.
    pub fn prob(&mut self, chunk: &[f32]) -> Result<f32, String> {
        debug_assert_eq!(chunk.len(), CHUNK);
        let mut samples = Vec::with_capacity(CONTEXT + CHUNK);
        samples.extend_from_slice(&self.context);
        samples.extend_from_slice(chunk);
        self.context.copy_from_slice(&samples[samples.len() - CONTEXT..]);
        let input = Tensor::from_array((vec![1i64, (CONTEXT + CHUNK) as i64], samples))
            .map_err(|e| e.to_string())?;
        let state = Tensor::from_array((vec![2i64, 1, 128], self.state.clone()))
            .map_err(|e| e.to_string())?;
        let sr = Tensor::from_array(((), vec![16_000i64])).map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])
            .map_err(|e| e.to_string())?;
        let prob = outputs["output"]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?
            .iter()
            .copied()
            .next()
            .ok_or("empty vad output")?;
        let new_state = outputs["stateN"]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        self.state = new_state.iter().copied().collect();
        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
        self.context.iter_mut().for_each(|v| *v = 0.0);
    }
}
