// Contour tracing module
// TODO: Implement contour extraction and smoothing

use anyhow::Result;
use tooltrace_common::{Contour, Point2DMm};

pub fn trace_contour(
    _segmented_image: &[u8],
    _pixel_to_mm: f64,
) -> Result<Contour> {
    // TODO: Extract contour from segmented image
    // TODO: Convert pixel coordinates to mm
    // TODO: Smooth contour
    unimplemented!("Contour tracing not yet implemented")
}
