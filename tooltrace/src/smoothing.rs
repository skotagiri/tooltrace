// Contour smoothing module
// Implements spline fitting to create gentle, low-frequency curves

use anyhow::Result;
use tooltrace_common::{Contour, Point2DMm};
use image::RgbImage;

/// Apply gentle spline smoothing to contours
/// Reduces the number of points and fits a smooth spline through them
pub fn apply_spline_smoothing(
    contours: Vec<Contour>,
    decimation_factor: usize,
    debug_path: Option<&str>,
    source_image: Option<&RgbImage>,
) -> Result<Vec<Contour>> {
    println!("Applying spline smoothing with decimation factor {}...", decimation_factor);

    let mut smoothed_contours = Vec::new();

    for (idx, contour) in contours.iter().enumerate() {
        if contour.points.len() < 4 {
            println!("  Contour {}: too few points ({}), skipping smoothing", idx, contour.points.len());
            smoothed_contours.push(contour.clone());
            continue;
        }

        println!("  Contour {}: {} points -> smoothing...", idx, contour.points.len());

        // Step 1: Decimate the contour to reduce high-frequency noise
        let decimated = decimate_contour(&contour.points, decimation_factor);
        println!("    After decimation: {} points", decimated.len());

        // Step 2: Apply Catmull-Rom spline interpolation for smooth curves
        let smoothed = catmull_rom_spline(&decimated, contour.closed);
        println!("    After spline: {} points", smoothed.len());

        smoothed_contours.push(Contour {
            points: smoothed,
            closed: contour.closed,
        });
    }

    // Save debug visualization if requested
    if let Some(path) = debug_path {
        if let Some(img) = source_image {
            save_smoothing_debug(&contours, &smoothed_contours, img, path)?;
        }
    }

    Ok(smoothed_contours)
}

/// Decimate a contour by keeping every Nth point
/// This reduces high-frequency detail and creates a low-frequency representation
fn decimate_contour(points: &[Point2DMm], factor: usize) -> Vec<Point2DMm> {
    if factor <= 1 {
        return points.to_vec();
    }

    let mut decimated = Vec::new();

    for (i, point) in points.iter().enumerate() {
        if i % factor == 0 {
            decimated.push(*point);
        }
    }

    // Ensure we include the last point if not already included
    if !decimated.is_empty() && decimated.len() * factor < points.len() {
        if let Some(last) = points.last() {
            if decimated.last() != Some(last) {
                decimated.push(*last);
            }
        }
    }

    decimated
}

/// Apply Catmull-Rom spline interpolation to create smooth curves
/// This creates a gentle spline that passes through all control points
fn catmull_rom_spline(control_points: &[Point2DMm], closed: bool) -> Vec<Point2DMm> {
    if control_points.len() < 4 {
        return control_points.to_vec();
    }

    let mut result = Vec::new();
    let segments_per_span = 10; // Number of interpolated points between control points

    let n = control_points.len();

    for i in 0..n {
        // Get the four control points for this segment
        let p0 = if closed {
            control_points[(i + n - 1) % n]
        } else if i == 0 {
            control_points[0] // Use first point twice at the start
        } else {
            control_points[i - 1]
        };

        let p1 = control_points[i];

        let p2 = if closed {
            control_points[(i + 1) % n]
        } else if i == n - 1 {
            break; // Stop at the last point for open curves
        } else {
            control_points[i + 1]
        };

        let p3 = if closed {
            control_points[(i + 2) % n]
        } else if i >= n - 2 {
            control_points[n - 1] // Use last point twice at the end
        } else {
            control_points[i + 2]
        };

        // Skip the last segment for closed curves to avoid duplication
        if closed && i == n - 1 {
            break;
        }

        // Interpolate between p1 and p2
        for j in 0..segments_per_span {
            let t = j as f64 / segments_per_span as f64;
            let point = catmull_rom_point(p0, p1, p2, p3, t);
            result.push(point);
        }
    }

    // Add the last control point for open curves
    if !closed && !control_points.is_empty() {
        result.push(*control_points.last().unwrap());
    }

    result
}

/// Calculate a point on a Catmull-Rom spline
/// t is in range [0, 1] representing position between p1 and p2
fn catmull_rom_point(
    p0: Point2DMm,
    p1: Point2DMm,
    p2: Point2DMm,
    p3: Point2DMm,
    t: f64,
) -> Point2DMm {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom basis functions
    let x = 0.5 * (
        (2.0 * p1.x) +
        (-p0.x + p2.x) * t +
        (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2 +
        (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
    );

    let y = 0.5 * (
        (2.0 * p1.y) +
        (-p0.y + p2.y) * t +
        (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2 +
        (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
    );

    Point2DMm { x, y }
}

/// Save debug image showing original and smoothed contours
fn save_smoothing_debug(
    original_contours: &[Contour],
    smoothed_contours: &[Contour],
    source_image: &RgbImage,
    path: &str,
) -> Result<()> {
    use opencv::core::{Mat, Point, Scalar, Vector};
    use opencv::prelude::MatTraitConst;
    use opencv::imgcodecs;

    // Convert source image to OpenCV Mat
    let img_data: Vec<u8> = source_image.as_raw().clone();
    let mat = Mat::from_slice(&img_data)?;
    let reshaped = mat.reshape(3, source_image.height() as i32)?;
    let mut debug_img = reshaped.try_clone()?;

    // Convert pixel coordinates to image coordinates (contours are in pixels at this point)
    let dpi = 300.0;
    let pixels_per_mm = dpi / 25.4;

    // Draw original contours in red (thin)
    for contour in original_contours {
        let mut opencv_contour: Vector<Point> = Vector::new();
        for pt in &contour.points {
            // Convert from pixels to image coordinates
            opencv_contour.push(Point::new(pt.x as i32, pt.y as i32));
        }

        let mut contour_vec: Vector<Vector<Point>> = Vector::new();
        contour_vec.push(opencv_contour);

        opencv::imgproc::draw_contours(
            &mut debug_img,
            &contour_vec,
            0,
            Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
            1,
            opencv::imgproc::LINE_8,
            &Mat::default(),
            i32::MAX,
            Point::new(0, 0),
        )?;
    }

    // Draw smoothed contours in green (thick)
    for contour in smoothed_contours {
        let mut opencv_contour: Vector<Point> = Vector::new();
        for pt in &contour.points {
            opencv_contour.push(Point::new(pt.x as i32, pt.y as i32));
        }

        let mut contour_vec: Vector<Vector<Point>> = Vector::new();
        contour_vec.push(opencv_contour);

        opencv::imgproc::draw_contours(
            &mut debug_img,
            &contour_vec,
            0,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
            3,
            opencv::imgproc::LINE_8,
            &Mat::default(),
            i32::MAX,
            Point::new(0, 0),
        )?;
    }

    imgcodecs::imwrite(path, &debug_img, &Vector::new())?;
    println!("Saved smoothing debug image to: {}", path);
    println!("  Red: Original contours, Green: Smoothed contours");

    Ok(())
}
