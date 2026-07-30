//! FaceMesh + Iris landmark inference (MediaPipe models exported to ONNX,
//! run via onnxruntime). Conventions follow the ailia-models reference:
//! 192×192 face crop and 64×64 eye crops, pixels normalized to [-1, 1],
//! left-eye crop horizontally flipped for the iris model.

use crate::gaze::mapping::FEATURE_DIM;
use ort::session::Session;
use ort::value::Tensor;

const MESH_RES: usize = 192;
const EYE_RES: usize = 64;

/// MediaPipe FaceMesh eye-contour landmark indices (subject's left / right).
const EYE_LEFT_CONTOUR: [usize; 16] = [
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
];
const EYE_RIGHT_CONTOUR: [usize; 16] = [
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
];

/// Square face region of interest in frame coordinates.
#[derive(Clone, Copy)]
pub struct Roi {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

pub struct GazeNet {
    facemesh: Session,
    iris: Session,
    /// Tracked face ROI from the previous frame; None → cold start.
    roi: Option<Roi>,
}

impl GazeNet {
    pub fn load() -> std::io::Result<Self> {
        let (facemesh_path, iris_path) = crate::models::ensure_gaze_models()?;
        let load = |p: &std::path::Path| {
            Session::builder()
                .and_then(|mut b| b.commit_from_file(p))
                .map_err(|e| std::io::Error::other(format!("onnx load {}: {e}", p.display())))
        };
        Ok(Self {
            facemesh: load(&facemesh_path)?,
            iris: load(&iris_path)?,
            roi: None,
        })
    }

    /// Run the full pipeline on an RGB frame; returns the gaze feature vector
    /// or None when no face is confidently visible.
    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Option<[f64; FEATURE_DIM]> {
        let roi = self.roi.unwrap_or_else(|| center_square(w, h));
        let crop = bilinear_crop(rgb, w, h, roi, MESH_RES);

        let (landmarks, score) = self.run_facemesh(&crop)?;
        if sigmoid(score) < 0.4 {
            self.roi = None;
            return None;
        }

        // Eye centers in 192-crop coordinates (needed for the iris crops).
        let eye_l = contour_mean(&landmarks, &EYE_LEFT_CONTOUR);
        let eye_r = contour_mean(&landmarks, &EYE_RIGHT_CONTOUR);

        let iris_l = self.run_iris(&crop, eye_l, true)?;
        let iris_r = self.run_iris(&crop, eye_r, false)?;

        // Track the ROI for the next frame from the landmark bounding box.
        self.roi = Some(next_roi(&landmarks, roi, w, h));

        // Map crop coords → frame coords for scale-invariant features.
        let to_frame = |p: [f32; 2]| {
            [
                roi.x + p[0] * roi.size / MESH_RES as f32,
                roi.y + p[1] * roi.size / MESH_RES as f32,
            ]
        };
        let (el, er) = (to_frame(eye_l), to_frame(eye_r));
        let (il, ir) = (to_frame(iris_l), to_frame(iris_r));
        let inter = ((el[0] - er[0]).powi(2) + (el[1] - er[1]).powi(2)).sqrt();
        if inter < 1.0 {
            return None;
        }
        let cx = (el[0] + er[0]) / 2.0;
        let cy = (el[1] + er[1]) / 2.0;
        Some([
            ((ir[0] - er[0]) / inter) as f64,
            ((ir[1] - er[1]) / inter) as f64,
            ((il[0] - el[0]) / inter) as f64,
            ((il[1] - el[1]) / inter) as f64,
            (cx / w as f32 - 0.5) as f64,
            (cy / h as f32 - 0.5) as f64,
            (inter / w as f32) as f64,
            (el[1] - er[1]).atan2(el[0] - er[0]) as f64,
        ])
    }

    /// Returns (1404 landmarks in 192-crop coords, raw confidence).
    fn run_facemesh(&mut self, crop: &[f32]) -> Option<(Vec<f32>, f32)> {
        let input = Tensor::from_array((
            vec![1i64, 3, MESH_RES as i64, MESH_RES as i64],
            crop.to_vec(),
        ))
        .ok()?;
        let outputs = self.facemesh.run(ort::inputs![input]).ok()?;
        let mut landmarks = None;
        let mut score = None;
        for (_, value) in outputs.iter() {
            if let Ok(arr) = value.try_extract_array::<f32>() {
                match arr.len() {
                    1404 => landmarks = Some(arr.iter().copied().collect::<Vec<f32>>()),
                    1 => score = Some(arr.iter().copied().next()?),
                    _ => {}
                }
            }
        }
        Some((landmarks?, score?))
    }

    /// Iris center in 192-crop coordinates for one eye.
    fn run_iris(&mut self, crop: &[f32], eye_center: [f32; 2], flip: bool) -> Option<[f32; 2]> {
        let x0 = (eye_center[0].round() as i32 - (EYE_RES / 2) as i32)
            .clamp(0, (MESH_RES - EYE_RES) as i32) as usize;
        let y0 = (eye_center[1].round() as i32 - (EYE_RES / 2) as i32)
            .clamp(0, (MESH_RES - EYE_RES) as i32) as usize;

        let mut eye = vec![0.0f32; 3 * EYE_RES * EYE_RES];
        for c in 0..3 {
            for y in 0..EYE_RES {
                for x in 0..EYE_RES {
                    let sx = if flip { x0 + EYE_RES - 1 - x } else { x0 + x };
                    eye[(c * EYE_RES + y) * EYE_RES + x] =
                        crop[(c * MESH_RES + y0 + y) * MESH_RES + sx];
                }
            }
        }
        let input =
            Tensor::from_array((vec![1i64, 3, EYE_RES as i64, EYE_RES as i64], eye)).ok()?;
        let outputs = self.iris.run(ort::inputs![input]).ok()?;
        let mut iris = None;
        for (_, value) in outputs.iter() {
            if let Ok(arr) = value.try_extract_array::<f32>() {
                if arr.len() == 15 {
                    iris = Some(arr.iter().copied().collect::<Vec<f32>>());
                }
            }
        }
        let iris = iris?;
        // First of the 5 iris points is the pupil center, in eye-crop coords.
        let (mut ix, iy) = (iris[0], iris[1]);
        if flip {
            ix = EYE_RES as f32 - 1.0 - ix;
        }
        Some([x0 as f32 + ix, y0 as f32 + iy])
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn contour_mean(landmarks: &[f32], indices: &[usize]) -> [f32; 2] {
    let mut sum = [0.0f32; 2];
    for &i in indices {
        sum[0] += landmarks[i * 3];
        sum[1] += landmarks[i * 3 + 1];
    }
    [sum[0] / indices.len() as f32, sum[1] / indices.len() as f32]
}

fn center_square(w: usize, h: usize) -> Roi {
    let size = w.min(h) as f32;
    Roi {
        x: (w as f32 - size) / 2.0,
        y: (h as f32 - size) / 2.0,
        size,
    }
}

/// Next-frame ROI: landmark bounding box expanded 1.7×, squared, clamped.
fn next_roi(landmarks: &[f32], roi: Roi, w: usize, h: usize) -> Roi {
    let scale = roi.size / MESH_RES as f32;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for lm in landmarks.chunks(3) {
        let x = roi.x + lm[0] * scale;
        let y = roi.y + lm[1] * scale;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let size = ((max_x - min_x).max(max_y - min_y) * 1.7).clamp(64.0, w.min(h) as f32);
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    Roi {
        x: (cx - size / 2.0).clamp(0.0, w as f32 - size),
        y: (cy - size / 2.0).clamp(0.0, h as f32 - size),
        size,
    }
}

/// Crop `roi` from the RGB frame and bilinear-resize to `out`×`out`,
/// normalized to [-1, 1], CHW layout.
fn bilinear_crop(rgb: &[u8], w: usize, h: usize, roi: Roi, out: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; 3 * out * out];
    let step = roi.size / out as f32;
    for oy in 0..out {
        let sy = (roi.y + (oy as f32 + 0.5) * step - 0.5).clamp(0.0, h as f32 - 1.0);
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = sy - y0 as f32;
        for ox in 0..out {
            let sx = (roi.x + (ox as f32 + 0.5) * step - 0.5).clamp(0.0, w as f32 - 1.0);
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = sx - x0 as f32;
            for c in 0..3 {
                let p = |x: usize, y: usize| rgb[(y * w + x) * 3 + c] as f32;
                let v = p(x0, y0) * (1.0 - fx) * (1.0 - fy)
                    + p(x1, y0) * fx * (1.0 - fy)
                    + p(x0, y1) * (1.0 - fx) * fy
                    + p(x1, y1) * fx * fy;
                buf[(c * out + oy) * out + ox] = v / 127.5 - 1.0;
            }
        }
    }
    buf
}
