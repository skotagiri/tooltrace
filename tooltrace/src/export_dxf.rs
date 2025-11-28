// DXF export module
// Exports traced contours to DXF format (AutoCAD/Fusion 360 compatible)

use anyhow::Result;
use dxf::{Drawing, entities::*, Color, LwPolylineVertex};
use dxf::enums::AcadVersion;
use tooltrace_common::Contour;

/// Export contours to DXF file
/// Coordinates are in millimeters
pub fn export_dxf(contours: &[Contour], output_path: &str) -> Result<()> {
    let mut drawing = Drawing::new();
    drawing.header.version = AcadVersion::R2010;

    // Calculate bounding box for reference
    let (min_x, min_y, max_x, max_y) = calculate_bounds(contours);
    let width = max_x - min_x;
    let height = max_y - min_y;

    // Export each contour as a polyline
    for (idx, contour) in contours.iter().enumerate() {
        if contour.points.is_empty() {
            continue;
        }

        // Create a LWPOLYLINE (lightweight polyline) for 2D contours
        let mut polyline = LwPolyline::default();

        // Add vertices
        for pt in &contour.points {
            polyline.vertices.push(LwPolylineVertex {
                x: pt.x,
                y: pt.y,
                .. Default::default()
            });
        }

        // Close the polyline if contour is closed
        polyline.set_is_closed(contour.closed);

        // Create entity with common properties
        let mut common = EntityCommon::default();
        common.layer = format!("CONTOUR-{}", idx);
        common.color = Color::from_index(7); // White/Black (default)

        let entity = Entity {
            common,
            specific: EntityType::LwPolyline(polyline),
        };

        // Add entity to drawing
        drawing.add_entity(entity);
    }

    // Save the DXF file
    drawing.save_file(output_path)?;

    println!("Exported {} contour(s) to DXF: {}", contours.len(), output_path);
    println!("  Bounds: {:.1}mm × {:.1}mm", width, height);
    println!("  Units: millimeters");

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

    if min_x < f64::MAX {
        (min_x, min_y, max_x, max_y)
    } else {
        (0.0, 0.0, 100.0, 100.0) // Default if no points
    }
}
