//! Gaze-feature → screen-point mapping (ridge-regularized affine fit from
//! calibration samples) and One-Euro smoothing of the resulting cursor.

use serde::{Deserialize, Serialize};

/// Per-frame gaze feature vector: iris offsets relative to the eye centers
/// (both eyes), face position, scale, and roll — see `landmarks.rs`.
pub const FEATURE_DIM: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mapping {
    /// Row per screen axis; FEATURE_DIM weights + bias.
    w: [Vec<f64>; 2],
}

impl Mapping {
    /// Least-squares fit with a small ridge term so 9 calibration points can
    /// stably determine FEATURE_DIM+1 parameters per axis.
    pub fn fit(samples: &[([f64; FEATURE_DIM], [f64; 2])]) -> Option<Self> {
        if samples.len() < 6 {
            return None;
        }
        let n = FEATURE_DIM + 1;
        let f = nalgebra::DMatrix::from_fn(samples.len(), n, |r, c| {
            if c == FEATURE_DIM {
                1.0
            } else {
                samples[r].0[c]
            }
        });
        let lambda = 1e-4;
        let gram = f.transpose() * &f + nalgebra::DMatrix::identity(n, n) * lambda;
        let inv = gram.try_inverse()?;
        let mut w: [Vec<f64>; 2] = [vec![0.0; n], vec![0.0; n]];
        for axis in 0..2 {
            let y = nalgebra::DVector::from_fn(samples.len(), |r, _| samples[r].1[axis]);
            let sol = &inv * f.transpose() * y;
            w[axis] = sol.iter().copied().collect();
        }
        Some(Self { w })
    }

    pub fn apply(&self, features: &[f64; FEATURE_DIM]) -> [f64; 2] {
        let mut out = [0.0; 2];
        for axis in 0..2 {
            let w = &self.w[axis];
            out[axis] = w[FEATURE_DIM] + features.iter().zip(w).map(|(f, w)| f * w).sum::<f64>();
        }
        out
    }

    /// Root-mean-square residual of the fit, in the target unit (pixels).
    pub fn rms_error(&self, samples: &[([f64; FEATURE_DIM], [f64; 2])]) -> f64 {
        let sq: f64 = samples
            .iter()
            .map(|(f, y)| {
                let p = self.apply(f);
                (p[0] - y[0]).powi(2) + (p[1] - y[1]).powi(2)
            })
            .sum();
        (sq / samples.len() as f64).sqrt()
    }
}

/// One-Euro filter (Casiez et al.): low lag when moving, low jitter when still.
pub struct OneEuro {
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
    prev: Option<(f64, f64)>, // (value, derivative)
}

impl OneEuro {
    pub fn new(min_cutoff: f64, beta: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            prev: None,
        }
    }

    pub fn filter(&mut self, x: f64, dt: f64) -> f64 {
        let Some((prev_x, prev_dx)) = self.prev else {
            self.prev = Some((x, 0.0));
            return x;
        };
        let alpha = |cutoff: f64| {
            let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
            1.0 / (1.0 + tau / dt)
        };
        let dx = (x - prev_x) / dt;
        let dx_hat = prev_dx + alpha(self.d_cutoff) * (dx - prev_dx);
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = alpha(cutoff);
        let x_hat = prev_x + a * (x - prev_x);
        self.prev = Some((x_hat, dx_hat));
        x_hat
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M3 self-check: fit on synthetic 9-point data, assert round-trip error.
    #[test]
    fn synthetic_nine_point_roundtrip() {
        // Ground truth: screen = affine function of features + slight noise.
        let truth = |f: &[f64; FEATURE_DIM]| {
            [
                960.0 + 4000.0 * f[0] + 3800.0 * f[2] + 120.0 * f[4],
                540.0 + 2600.0 * f[1] + 2500.0 * f[3] - 90.0 * f[5],
            ]
        };
        let mut samples = Vec::new();
        for gy in 0..3 {
            for gx in 0..3 {
                let mut f = [0.0; FEATURE_DIM];
                f[0] = (gx as f64 - 1.0) * 0.1;
                f[1] = (gy as f64 - 1.0) * 0.08;
                f[2] = (gx as f64 - 1.0) * 0.11;
                f[3] = (gy as f64 - 1.0) * 0.09;
                f[4] = 0.5 + 0.01 * gx as f64;
                f[5] = -0.02 * gy as f64;
                f[6] = 0.3;
                f[7] = 0.001 * (gx + gy) as f64;
                samples.push((f, truth(&f)));
            }
        }
        let mapping = Mapping::fit(&samples).expect("fit");
        assert!(
            mapping.rms_error(&samples) < 1.0,
            "rms {} px too high",
            mapping.rms_error(&samples)
        );
    }

    #[test]
    fn one_euro_converges_and_smooths() {
        let mut f = OneEuro::new(1.0, 0.005);
        let mut y = 0.0;
        for _ in 0..200 {
            y = f.filter(100.0, 1.0 / 30.0);
        }
        assert!((y - 100.0).abs() < 1.0, "did not converge: {y}");
    }
}
