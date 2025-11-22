use tooltrace_common::{PaperSize, AprilTagConfig};

/// Calculate marker positions for the four corners
/// Returns (x, y) positions in millimeters from bottom-left origin
/// Order: [top-left, top-right, bottom-right, bottom-left]
pub fn calculate_marker_positions(
    paper_size: PaperSize,
    tag_config: &AprilTagConfig,
) -> [(f64, f64); 4] {
    let (width_mm, height_mm) = paper_size.dimensions_mm();
    let margin = 15.0; // 15mm margin from edges
    let tag_size = tag_config.size_mm;

    [
        // Top-left (ID 0)
        (margin, height_mm - margin - tag_size),
        // Top-right (ID 1)
        (width_mm - margin - tag_size, height_mm - margin - tag_size),
        // Bottom-right (ID 2)
        (width_mm - margin - tag_size, margin),
        // Bottom-left (ID 3)
        (margin, margin),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_positions_a4() {
        let config = AprilTagConfig::default();
        let positions = calculate_marker_positions(PaperSize::A4, &config);

        // A4 is 210x297mm, margin 15mm, tag 50mm
        assert_eq!(positions[0], (15.0, 232.0)); // top-left
        assert_eq!(positions[1], (145.0, 232.0)); // top-right
        assert_eq!(positions[2], (145.0, 15.0)); // bottom-right
        assert_eq!(positions[3], (15.0, 15.0)); // bottom-left
    }
}
