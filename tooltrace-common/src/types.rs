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

impl Default for AprilTagConfig {
    fn default() -> Self {
        Self {
            family: TagFamily::Tag36h11,
            size_mm: 50.0, // 50mm tag size
            corner_ids: [0, 1, 2, 3],
        }
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
