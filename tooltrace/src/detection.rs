// AprilTag detection module
// TODO: Implement AprilTag detection using the apriltag crate

use anyhow::Result;

pub struct TagDetection {
    pub id: u32,
    pub center: (f64, f64),
    pub corners: [(f64, f64); 4],
}

pub fn detect_apriltags(_image_path: &str) -> Result<Vec<TagDetection>> {
    // TODO: Implement using apriltag crate
    unimplemented!("AprilTag detection not yet implemented")
}
