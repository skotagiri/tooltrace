// SVG export module
// Exports traced contours to SVG format with millimeter coordinates

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use tooltrace_common::Contour;

/// Export contours to SVG file
/// Coordinates are in millimeters
pub fn export_svg(contours: &[Contour], output_path: &str) -> Result<()> {
    let mut file = File::create(output_path)?;

    // Calculate bounding box
    let (min_x, min_y, max_x, max_y) = calculate_bounds(contours);
    let width = max_x - min_x;
    let height = max_y - min_y;

    // SVG header with viewBox in mm
    writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(file, r#"<svg xmlns="http://www.w3.org/2000/svg" version="1.1""#)?;
    writeln!(file, r#"     width="{}mm" height="{}mm""#, width, height)?;
    writeln!(file, r#"     viewBox="{} {} {} {}">"#, min_x, min_y, width, height)?;
    writeln!(file)?;

    // Add metadata
    writeln!(file, r#"  <title>ToolTrace - Object Outline</title>"#)?;
    writeln!(file, r#"  <desc>Traced object contour from calibration paper photo. Units: millimeters</desc>"#)?;
    writeln!(file)?;

    // Export each contour as a path
    for (idx, contour) in contours.iter().enumerate() {
        if contour.points.is_empty() {
            continue;
        }

        write!(file, r#"  <path id="contour-{}" "#, idx)?;
        write!(file, r#"stroke="black" stroke-width="0.1" fill="none" "#)?;
        write!(file, r#"d=""#)?;

        // Move to first point
        let first = &contour.points[0];
        write!(file, "M {:.3},{:.3} ", first.x, first.y)?;

        // Line to subsequent points
        for pt in &contour.points[1..] {
            write!(file, "L {:.3},{:.3} ", pt.x, pt.y)?;
        }

        // Close path if contour is closed
        if contour.closed {
            write!(file, "Z")?;
        }

        writeln!(file, r#"" />"#)?;
    }

    writeln!(file)?;
    writeln!(file, "</svg>")?;

    println!("Exported {} contour(s) to SVG: {}", contours.len(), output_path);
    println!("  Bounds: {:.1}mm × {:.1}mm", width, height);

    Ok(())
}

/// Calculate bounding box for all contours
fn calculate_bounds(contours: &[Contour]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for contour in contours {
        for pt in &contour.points {
            min_x = min_x.min(pt.x);
            min_y = min_y.min(pt.y);
            max_x = max_x.max(pt.x);
            max_y = max_y.max(pt.y);
        }
    }

    // Add small margin if bounds are valid
    if min_x < f64::MAX {
        let margin = 1.0; // 1mm margin
        (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    } else {
        (0.0, 0.0, 100.0, 100.0) // Default if no points
    }
}
