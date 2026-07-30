//! Gaze pipeline: webcam → FaceMesh/Iris landmarks → calibrated affine
//! mapping → One-Euro-smoothed CGEvent cursor moves.

pub mod landmarks;
pub mod mapping;
pub mod mouse;

use mapping::{FEATURE_DIM, Mapping, OneEuro};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub struct Shared {
    /// Recent feature vectors, for calibration capture.
    pub recent: Mutex<VecDeque<[f64; FEATURE_DIM]>>,
    pub mapping: RwLock<Option<Mapping>>,
    /// Move the cursor only when true (gaze toggled on, not calibrating).
    pub move_enabled: AtomicBool,
    stop: AtomicBool,
}

pub struct Gaze {
    pub shared: Arc<Shared>,
    /// 3×3 calibration samples collected so far: (features, screen point).
    pub samples: Mutex<Vec<([f64; FEATURE_DIM], [f64; 2])>>,
}

fn mapping_path() -> std::path::PathBuf {
    crate::control::state_dir().join("gaze-mapping.json")
}

impl Gaze {
    /// Spawn the camera + inference thread. Returns immediately.
    pub fn start(move_enabled: bool) -> Self {
        let mapping = std::fs::read(mapping_path())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let shared = Arc::new(Shared {
            recent: Mutex::new(VecDeque::new()),
            mapping: RwLock::new(mapping),
            move_enabled: AtomicBool::new(move_enabled),
            stop: AtomicBool::new(false),
        });
        let worker = shared.clone();
        std::thread::spawn(move || {
            if let Err(e) = camera_loop(&worker) {
                eprintln!("gaze pipeline error: {e}");
            }
        });
        Self {
            shared,
            samples: Mutex::new(Vec::new()),
        }
    }

    pub fn is_calibrated(&self) -> bool {
        self.shared.mapping.read().unwrap().is_some()
    }

    /// Mean of the freshest feature vectors; None until the camera warms up.
    pub fn capture_features(&self) -> Option<[f64; FEATURE_DIM]> {
        let recent = self.shared.recent.lock().unwrap();
        if recent.len() < 5 {
            return None;
        }
        let mut mean = [0.0; FEATURE_DIM];
        for f in recent.iter() {
            for (m, v) in mean.iter_mut().zip(f) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= recent.len() as f64;
        }
        Some(mean)
    }

    /// Fit the mapping from collected samples, persist it, activate it.
    /// Returns the rms residual in pixels.
    pub fn finish_calibration(&self) -> Result<f64, String> {
        let samples = self.samples.lock().unwrap();
        let mapping = Mapping::fit(&samples).ok_or("not enough calibration samples")?;
        let rms = mapping.rms_error(&samples);
        std::fs::write(
            mapping_path(),
            serde_json::to_vec(&mapping).expect("serialize mapping"),
        )
        .map_err(|e| format!("cannot save mapping: {e}"))?;
        *self.shared.mapping.write().unwrap() = Some(mapping);
        Ok(rms)
    }
}

impl Drop for Gaze {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
    }
}

/// The 9 calibration targets: a 3×3 grid at 10% / 50% / 90% of the display.
pub fn calibration_point(index: usize) -> [f64; 2] {
    let (ox, oy, w, h) = mouse::main_display_bounds();
    let frac = [0.1, 0.5, 0.9];
    [ox + w * frac[index % 3], oy + h * frac[index / 3]]
}

fn camera_loop(shared: &Shared) -> Result<(), String> {
    let mut net = landmarks::GazeNet::load().map_err(|e| e.to_string())?;

    let index = CameraIndex::Index(0);
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, format).map_err(|e| {
        format!("camera open failed ({e}) — check macOS camera permission for your terminal app")
    })?;
    camera
        .open_stream()
        .map_err(|e| format!("camera stream: {e}"))?;
    eprintln!("gaze: camera open, models loaded");

    let cfg = crate::config::get();
    let mut filter_x = OneEuro::new(cfg.smoothing_min_cutoff, cfg.smoothing_beta);
    let mut filter_y = OneEuro::new(cfg.smoothing_min_cutoff, cfg.smoothing_beta);
    let mut last = std::time::Instant::now();

    while !shared.stop.load(Ordering::SeqCst) {
        let frame = match camera.frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("camera frame: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };
        let img = match frame.decode_image::<RgbFormat>() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("frame decode: {e}");
                continue;
            }
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.into_raw();

        let Some(features) = net.process(&rgb, w, h) else {
            filter_x.reset();
            filter_y.reset();
            continue;
        };
        {
            let mut recent = shared.recent.lock().unwrap();
            recent.push_back(features);
            while recent.len() > 15 {
                recent.pop_front();
            }
        }
        if shared.move_enabled.load(Ordering::SeqCst) {
            if let Some(mapping) = shared.mapping.read().unwrap().as_ref() {
                let [x, y] = mapping.apply(&features);
                let dt = last.elapsed().as_secs_f64().max(1e-3);
                let sx = filter_x.filter(x, dt);
                let sy = filter_y.filter(y, dt);
                if let Err(e) = mouse::move_cursor(sx, sy) {
                    eprintln!("cursor move: {e}");
                }
            }
        }
        last = std::time::Instant::now();
    }
    eprintln!("gaze: camera stopped");
    Ok(())
}

use nokhwa::Camera;
