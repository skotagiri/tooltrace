// Object segmentation module
// Implements background subtraction and edge detection to find object contours

use anyhow::Result;
use image::RgbImage;
use opencv::{
    core::{Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT, AlgorithmHint},
    imgcodecs,
    imgproc::{self, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL},
    prelude::{MatTraitConstManual, MatTrait},
};
use tooltrace_common::{Contour, Point2DMm};

/// Segment objects from the flattened paper image
/// Uses edge detection and contour finding to extract object outlines
/// Excludes AprilTag regions and returns only outer boundary contours for tool storage plate generation
pub fn segment_object(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)], // (x, y, width, height) in pixels at 300 DPI
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    println!("Segmenting objects with {} AprilTag exclusion region(s)...", tag_regions.len());

    // Convert image to OpenCV Mat
    let img_data: Vec<u8> = flattened_image.as_raw().clone();
    let mat = Mat::from_slice(&img_data)?;
    let mat = mat.reshape(3, flattened_image.height() as i32)?;

    // Convert to grayscale
    let mut gray = Mat::default();
    opencv::imgproc::cvt_color(&mat, &mut gray, opencv::imgproc::COLOR_RGB2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // Mask out AprilTag regions by drawing filled white rectangles (paper background color)
    for (x, y, w, h) in tag_regions {
        let rect = opencv::core::Rect::new(*x, *y, *w, *h);
        opencv::imgproc::rectangle(
            &mut gray,
            rect,
            Scalar::all(255.0), // White
            -1, // Filled
            opencv::imgproc::LINE_8,
            0,
        )?;

        println!("  Masking AprilTag region at ({}, {}) size {}x{}", x, y, w, h);
    }

    // Apply Gaussian blur to reduce noise
    let mut blurred = Mat::default();
    opencv::imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        opencv::core::Size::new(5, 5),
        1.5,
        1.5,
        BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Apply Canny edge detection with MORE SENSITIVE thresholds for complete tool outlines
    // Lower thresholds (20, 60) instead of (50, 150) to capture subtle edges
    let mut edges = Mat::default();
    opencv::imgproc::canny(&blurred, &mut edges, 20.0, 60.0, 3, false)?;

    println!("Applied sensitive Canny edge detection (20, 60 thresholds)");

    // Dilate edges to close small gaps
    let kernel = opencv::imgproc::get_structuring_element(
        opencv::imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        opencv::core::Point::new(-1, -1),
    )?;
    let mut dilated = Mat::default();
    opencv::imgproc::dilate(
        &edges,
        &mut dilated,
        &kernel,
        opencv::core::Point::new(-1, -1),
        2,
        BORDER_DEFAULT,
        Scalar::default(),
    )?;

    // Find contours
    let mut contours = Vector::<Vector<Point>>::new();
    opencv::imgproc::find_contours(
        &dilated,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    println!("Found {} contours", contours.len());

    // Filter contours by area (remove very small noise)
    // Lower threshold to capture smaller tools
    let min_area = 500.0; // Minimum area in pixels (~3mm² at 300 DPI)
    let mut filtered_contours = Vec::new();

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = opencv::imgproc::contour_area(&contour, false)?;

        if area > min_area {
            // Calculate bounding box to check if it overlaps with AprilTag regions
            let bounding_rect = opencv::imgproc::bounding_rect(&contour)?;

            // Check if contour overlaps significantly with any tag region
            let mut overlaps_tag = false;
            for (tx, ty, tw, th) in tag_regions {
                let tag_rect = opencv::core::Rect::new(*tx, *ty, *tw, *th);
                let intersection = bounding_rect & tag_rect;

                // If intersection area is > 10% of contour bounding box, it's likely a tag edge
                let intersection_area = (intersection.width * intersection.height) as f64;
                let contour_bbox_area = (bounding_rect.width * bounding_rect.height) as f64;

                if intersection_area > contour_bbox_area * 0.1 {
                    overlaps_tag = true;
                    break;
                }
            }

            if !overlaps_tag {
                println!("  Contour {}: area = {:.1} pixels ({:.1}mm²)", i, area, area / (11.811 * 11.811));

                // Convert OpenCV contour to our Contour type (in pixels for now)
                let mut points = Vec::new();
                for j in 0..contour.len() {
                    let pt = contour.get(j)?;
                    points.push(Point2DMm {
                        x: pt.x as f64,
                        y: pt.y as f64,
                    });
                }

                filtered_contours.push(Contour {
                    points,
                    closed: true, // RETR_EXTERNAL always gives closed outer boundaries
                });
            } else {
                println!("  Contour {}: EXCLUDED (overlaps with AprilTag region)", i);
            }
        }
    }

    println!("Filtered to {} tool contours (area > {} pixels, excluding tags)", filtered_contours.len(), min_area);

    // Save debug visualization if requested
    if let Some(path) = debug_path {
        let mut debug_img = mat.try_clone()?;
        let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = opencv::imgproc::contour_area(&contour, false)?;
            if area > min_area {
                let mut contour_vec: Vector<Vector<Point>> = Vector::new();
                contour_vec.push(contour);
                opencv::imgproc::draw_contours(
                    &mut debug_img,
                    &contour_vec,
                    0,
                    color,
                    2,
                    opencv::imgproc::LINE_8,
                    &Mat::default(),
                    i32::MAX,
                    Point::new(0, 0),
                )?;
            }
        }

        imgcodecs::imwrite(path, &debug_img, &Vector::new())?;
        println!("Saved contour debug image to: {}", path);
    }

    Ok(filtered_contours)
}

/// Convert contours from pixels to millimeters
/// The flattened image is at 300 DPI
pub fn pixels_to_mm(contours: Vec<Contour>, dpi: f64) -> Vec<Contour> {
    let pixels_per_mm = dpi / 25.4; // 300 DPI = 11.811 pixels/mm

    contours.into_iter().map(|contour| {
        let points = contour.points.into_iter().map(|pt| {
            Point2DMm {
                x: pt.x / pixels_per_mm,
                y: pt.y / pixels_per_mm,
            }
        }).collect();

        Contour {
            points,
            closed: contour.closed,
        }
    }).collect()
}
