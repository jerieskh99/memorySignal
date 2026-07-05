//! Family B / spatial-shift: did content shift within the page (memmove), via FFT.

use crate::metrics::common::xcorr_metrics;

#[derive(Clone, Default, Debug)]
pub struct Spatial {
    pub cross_corr_lag: f32, // lag (signed) of max raw circular cross-correlation
    pub phase_corr: f32,     // peak of the phase-correlation (normalised cross-power)
    pub byte_rotation: f32,  // lag (signed) of the phase-correlation peak
}

pub fn compute(p: &[u8], q: &[u8], speed: u8) -> Spatial {
    if speed >= 2 {
        // FFT cross-correlation (2 forward + 2 inverse transforms) -- part of the heavy
        // 12, dropped at speed >= 2; all three lags emit 0.
        return Spatial::default();
    }
    let (cross_corr_lag, phase_corr, byte_rotation) = xcorr_metrics(p, q);
    Spatial {
        cross_corr_lag,
        phase_corr,
        byte_rotation,
    }
}
