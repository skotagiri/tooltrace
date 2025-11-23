use serde::{Deserialize, Serialize};
use std::fmt;

/// Paper size variants with dimensions in millimeters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PaperSize {
    /// A4: 210 × 297 mm
    A4,
    /// US Letter: 215.9 × 279.4 mm (8.5 × 11 inches)
    Letter,
    /// A3: 297 × 420 mm
    A3,
}

impl PaperSize {
    /// Returns (width, height) in millimeters
    pub fn dimensions_mm(&self) -> (f64, f64) {
        match self {
            PaperSize::A4 => (210.0, 297.0),
            PaperSize::Letter => (215.9, 279.4),
            PaperSize::A3 => (297.0, 420.0),
        }
    }

    /// Returns (width, height) in points (1/72 inch)
    /// Used for PDF generation
    pub fn dimensions_pt(&self) -> (f64, f64) {
        let (w_mm, h_mm) = self.dimensions_mm();
        // Convert mm to points: 1 mm = 2.83465 pt
        (w_mm * 2.83465, h_mm * 2.83465)
    }
}

impl fmt::Display for PaperSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaperSize::A4 => write!(f, "A4 (210×297mm)"),
            PaperSize::Letter => write!(f, "Letter (8.5×11in)"),
            PaperSize::A3 => write!(f, "A3 (297×420mm)"),
        }
    }
}

/// AprilTag configuration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AprilTagConfig {
    /// Tag family (e.g., "36h11")
    pub family: TagFamily,
    /// Physical size of the tag in millimeters (outer black square)
    pub size_mm: f64,
    /// IDs of the 4 corner tags [top-left, top-right, bottom-right, bottom-left]
    pub corner_ids: [u32; 4],
}

impl AprilTagConfig {
    /// Get tag configuration for a specific paper size
    ///
    /// Tag ID scheme:
    /// - A4: IDs 0-3 (top-left, top-right, bottom-right, bottom-left)
    /// - Letter: IDs 4-7
    /// - A3: IDs 8-11
    ///
    /// This allows automatic paper size detection from detected tags
    pub fn for_paper_size(paper_size: PaperSize, tag_size_mm: f64) -> Self {
        let corner_ids = match paper_size {
            PaperSize::A4 => [0, 1, 2, 3],
            PaperSize::Letter => [4, 5, 6, 7],
            PaperSize::A3 => [8, 9, 10, 11],
        };

        Self {
            family: TagFamily::Tag36h11,
            size_mm: tag_size_mm,
            corner_ids,
        }
    }

    /// Detect paper size from corner tag IDs
    /// Returns Some(PaperSize) if all 4 tags match a known pattern
    pub fn detect_paper_size(tag_ids: &[u32]) -> Option<PaperSize> {
        if tag_ids.len() != 4 {
            return None;
        }

        // Check if all IDs are in the same range
        let min_id = *tag_ids.iter().min()?;
        let max_id = *tag_ids.iter().max()?;

        match (min_id, max_id) {
            (0, 3) => Some(PaperSize::A4),
            (4, 7) => Some(PaperSize::Letter),
            (8, 11) => Some(PaperSize::A3),
            _ => None,
        }
    }
}

impl Default for AprilTagConfig {
    fn default() -> Self {
        Self::for_paper_size(PaperSize::A4, 50.0)
    }
}

/// AprilTag family
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TagFamily {
    Tag36h11,
    Tag25h9,
    Tag16h5,
}

impl fmt::Display for TagFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TagFamily::Tag36h11 => write!(f, "36h11"),
            TagFamily::Tag25h9 => write!(f, "25h9"),
            TagFamily::Tag16h5 => write!(f, "16h5"),
        }
    }
}

/// 2D point in millimeters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2DMm {
    pub x: f64,
    pub y: f64,
}

impl Point2DMm {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Contour represented as a sequence of points in millimeters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contour {
    pub points: Vec<Point2DMm>,
    /// Whether the contour is closed
    pub closed: bool,
}

/// Output format for traced vectors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Svg,
    Dxf,
    Both,
}
