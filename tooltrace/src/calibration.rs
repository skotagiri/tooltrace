// Camera calibration and perspective correction module
// Uses OpenCV for homography calculation and perspective warping

use anyhow::{Context, Result, bail};
use image::{RgbImage, Rgb};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut};
use ab_glyph::{FontRef, PxScale};
use tooltrace_common::PaperSize;
use crate::detection::TagDetection;
use opencv::{
    core::{Mat, Point2f, Size, Vector, BORDER_CONSTANT, Scalar},
    imgproc::{warp_perspective, INTER_LINEAR},
    calib3d::find_homography,
    prelude::{MatTraitConst, MatTraitConstManual},
};


pub struct CameraCalibration {
    pub homography_mat: Mat,  // OpenCV Mat for homography
    pub output_width: u32,
    pub output_height: u32,
    pub rotated_image: Option<RgbImage>,
}

/// Calculate calibration from detected tags
/// Requires exactly 4 corner tags with consecutive IDs (0-3, 4-7, or 8-11)
pub fn calculate_calibration(
    detections: &[TagDetection],
    tag_size_mm: f64,
    input_image: &RgbImage,
    debug_prefix: Option<&str>,
) -> Result<CameraCalibration> {
    // Find the 4 corner tags
    if detections.len() < 4 {
        bail!("Need at least 4 corner tags for calibration, found only {}", detections.len());
    }

    // Detect paper size from tag IDs
    let tag_ids: Vec<u32> = detections.iter().map(|d| d.id).collect();
    let paper_size = tooltrace_common::AprilTagConfig::detect_paper_size(&tag_ids)
        .context("Could not determine paper size from detected tags. Expected 4 consecutive IDs (0-3 for A4, 4-7 for Letter, or 8-11 for A3)")?;

    println!("Detected paper size: {}", paper_size);

    // Find tags by ID and sort them
    // Expected order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    let base_id = match paper_size {
        PaperSize::A4 => 0,
        PaperSize::Letter => 4,
        PaperSize::A3 => 8,
    };

    let mut corner_tags = vec![None; 4];
    for det in detections {
        let idx = det.id as i32 - base_id as i32;
        if idx >= 0 && idx < 4 {
            corner_tags[idx as usize] = Some(det);
        }
    }

    // Verify we have all 4 corners
    for i in 0..4 {
        if corner_tags[i].is_none() {
            bail!("Missing corner tag ID {}", base_id + i as u32);
        }
    }

    // Unwrap the corner tags
    let corner_tags: Vec<&TagDetection> = corner_tags.into_iter().map(|t| t.unwrap()).collect();

    // Identify corners by GEOMETRIC position (not TAG ID, as paper generation may have bugs)
    // In image coordinates: X increases right, Y increases downward
    // So: top-left = min(X+Y), top-right = max(X-Y), bottom-right = max(X+Y), bottom-left = min(X-Y)

    let top_left_tag = corner_tags.iter()
        .min_by(|a, b| {
            let a_score = a.center.0 + a.center.1;
            let b_score = b.center.0 + b.center.1;
            a_score.partial_cmp(&b_score).unwrap()
        })
        .unwrap();

    let top_right_tag = corner_tags.iter()
        .max_by(|a, b| {
            let a_score = a.center.0 - a.center.1;
            let b_score = b.center.0 - b.center.1;
            a_score.partial_cmp(&b_score).unwrap()
        })
        .unwrap();

    let bottom_right_tag = corner_tags.iter()
        .max_by(|a, b| {
            let a_score = a.center.0 + a.center.1;
            let b_score = b.center.0 + b.center.1;
            a_score.partial_cmp(&b_score).unwrap()
        })
        .unwrap();

    let bottom_left_tag = corner_tags.iter()
        .min_by(|a, b| {
            let a_score = a.center.0 - a.center.1;
            let b_score = b.center.0 - b.center.1;
            a_score.partial_cmp(&b_score).unwrap()
        })
        .unwrap();

    println!("Corner assignments by GEOMETRIC position (in image coordinates):");
    println!("  Top-left: ID {} at ({:.1}, {:.1})", top_left_tag.id, top_left_tag.center.0, top_left_tag.center.1);
    println!("  Top-right: ID {} at ({:.1}, {:.1})", top_right_tag.id, top_right_tag.center.0, top_right_tag.center.1);
    println!("  Bottom-right: ID {} at ({:.1}, {:.1})", bottom_right_tag.id, bottom_right_tag.center.0, bottom_right_tag.center.1);
    println!("  Bottom-left: ID {} at ({:.1}, {:.1})", bottom_left_tag.id, bottom_left_tag.center.0, bottom_left_tag.center.1);

    // Detect paper rotation based on tag IDs
    // Expected IDs: base_id+0 (TL), base_id+1 (TR), base_id+2 (BR), base_id+3 (BL)
    let rotation = detect_paper_rotation(
        top_left_tag.id, top_right_tag.id, bottom_right_tag.id, bottom_left_tag.id, base_id
    );

    println!("\nDetected paper rotation: {}", match rotation {
        0 => "0° (upright)",
        1 => "90° counter-clockwise (or 270° clockwise)",
        2 => "180° (upside down)",
        3 => "270° counter-clockwise (or 90° clockwise)",
        _ => "unknown"
    });

    // Remap corners based on rotation
    // Map geometric positions to actual paper corners by finding where each tag ID is
    // Goal: Find which geometric position contains Tag base_id+0, base_id+1, base_id+2, base_id+3
    let (actual_tl, actual_tr, actual_br, actual_bl) = match rotation {
        0 => (top_left_tag, top_right_tag, bottom_right_tag, bottom_left_tag), // No rotation
        1 => (bottom_left_tag, top_left_tag, top_right_tag, bottom_right_tag), // 90° CCW: TL has Tag 5, TR has Tag 6, BR has Tag 7, BL has Tag 4
        2 => (bottom_right_tag, bottom_left_tag, top_left_tag, top_right_tag), // 180°: TL has Tag 6, TR has Tag 7, BR has Tag 4, BL has Tag 5
        3 => (top_right_tag, bottom_right_tag, bottom_left_tag, top_left_tag), // 270° CCW: TL has Tag 7, TR has Tag 4, BR has Tag 5, BL has Tag 6
        _ => bail!("Invalid rotation detected"),
    };

    println!("\nCorrected corner assignments (actual paper positions):");
    println!("  Paper top-left (ID {}): at image ({:.1}, {:.1})", actual_tl.id, actual_tl.center.0, actual_tl.center.1);
    println!("  Paper top-right (ID {}): at image ({:.1}, {:.1})", actual_tr.id, actual_tr.center.0, actual_tr.center.1);
    println!("  Paper bottom-right (ID {}): at image ({:.1}, {:.1})", actual_br.id, actual_br.center.0, actual_br.center.1);
    println!("  Paper bottom-left (ID {}): at image ({:.1}, {:.1})", actual_bl.id, actual_bl.center.0, actual_bl.center.1);

    // Now use the corrected corners for the rest of the calibration
    let top_left_tag = actual_tl;
    let top_right_tag = actual_tr;
    let bottom_right_tag = actual_br;
    let bottom_left_tag = actual_bl;

    // Use the OUTER CORNERS of each tag for accurate paper boundary detection
    // AprilTag corners are ordered: [0]=bottom-left, [1]=bottom-right, [2]=top-right, [3]=top-left
    // in the tag's local coordinate system

    // For top-left tag, use corner closest to top-left (min x+y)
    let tl_corner = top_left_tag.corners.iter()
        .min_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap())
        .unwrap();

    // For top-right tag, use corner closest to top-right (max x, min y)
    let tr_corner = top_right_tag.corners.iter()
        .max_by(|a, b| (a.0 - a.1).partial_cmp(&(b.0 - b.1)).unwrap())
        .unwrap();

    // For bottom-right tag, use corner closest to bottom-right (max x+y)
    let br_corner = bottom_right_tag.corners.iter()
        .max_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap())
        .unwrap();

    // For bottom-left tag, use corner closest to bottom-left (min x, max y)
    let bl_corner = bottom_left_tag.corners.iter()
        .min_by(|a, b| (a.0 - a.1).partial_cmp(&(b.0 - b.1)).unwrap())
        .unwrap();

    // Use DIRECT perspective correction from original image
    // Skip rotation - let homography handle all the transformation
    println!("\nUsing direct perspective correction (no pre-rotation)");

    let working_image = input_image.clone();
    let crop_x = 0;
    let crop_y = 0;
    let cropped_image = working_image.clone();

    // Use the original detected tags (no rotation needed)
    let rot_top_left_tag = top_left_tag;
    let rot_top_right_tag = top_right_tag;
    let rot_bottom_right_tag = bottom_right_tag;
    let rot_bottom_left_tag = bottom_left_tag;

    let rotated_corners = (*tl_corner, *tr_corner, *br_corner, *bl_corner);

    // Use proper computer vision approach: map tag CENTERS to their known positions
    // Working entirely in pixels at 300 DPI

    let (width_inches, height_inches) = paper_size.dimensions_inches();
    let dpi = 300.0;
    let margin_mm = 15.0;  // Distance from paper edge to tag outer corner
    let margin_inches = margin_mm / 25.4;  // Convert mm to inches

    // Tag centers are at margin + tag_size/2 from paper edges (in inches)
    let tag_size_inches = tag_size_mm / 25.4;
    let tag_center_offset_inches = margin_inches + tag_size_inches / 2.0;
    let tag_center_offset_pixels = tag_center_offset_inches * dpi;

    // Output dimensions at 300 DPI
    let output_width = (width_inches * dpi) as u32;
    let output_height = (height_inches * dpi) as u32;

    println!("Output image size: {}x{} pixels at {} DPI", output_width, output_height, dpi);

    // Get tag centers in cropped image coordinates (source points in pixels)
    let src_points = vec![
        (rot_top_left_tag.center.0 - crop_x as f64, rot_top_left_tag.center.1 - crop_y as f64),
        (rot_top_right_tag.center.0 - crop_x as f64, rot_top_right_tag.center.1 - crop_y as f64),
        (rot_bottom_right_tag.center.0 - crop_x as f64, rot_bottom_right_tag.center.1 - crop_y as f64),
        (rot_bottom_left_tag.center.0 - crop_x as f64, rot_bottom_left_tag.center.1 - crop_y as f64),
    ];

    // Also store tag corners for visualization
    let tag_corners_in_crop = vec![
        (rotated_corners.0.0 - crop_x as f64, rotated_corners.0.1 - crop_y as f64),
        (rotated_corners.1.0 - crop_x as f64, rotated_corners.1.1 - crop_y as f64),
        (rotated_corners.2.0 - crop_x as f64, rotated_corners.2.1 - crop_y as f64),
        (rotated_corners.3.0 - crop_x as f64, rotated_corners.3.1 - crop_y as f64),
    ];

    // Destination points are tag centers in output image coordinates (pixels at 300 DPI)
    // Tags are positioned at margin + tag_size/2 from each edge
    let dst_points = vec![
        // Top-left tag center
        (tag_center_offset_pixels, tag_center_offset_pixels),
        // Top-right tag center
        (output_width as f64 - tag_center_offset_pixels, tag_center_offset_pixels),
        // Bottom-right tag center
        (output_width as f64 - tag_center_offset_pixels, output_height as f64 - tag_center_offset_pixels),
        // Bottom-left tag center
        (tag_center_offset_pixels, output_height as f64 - tag_center_offset_pixels),
    ];

    println!("Using tag centers for homography calculation (pixels to pixels):");
    println!("  Tag center offset from paper edge: {:.1}mm = {:.1}px at {}DPI",
        margin_mm + tag_size_mm / 2.0, tag_center_offset_pixels, dpi);

    println!("\nSource points (pixels) - tag centers in input image:");
    println!("  Top-left tag (ID {}) center: ({:.1}, {:.1})", rot_top_left_tag.id, src_points[0].0, src_points[0].1);
    println!("  Top-right tag (ID {}) center: ({:.1}, {:.1})", rot_top_right_tag.id, src_points[1].0, src_points[1].1);
    println!("  Bottom-right tag (ID {}) center: ({:.1}, {:.1})", rot_bottom_right_tag.id, src_points[2].0, src_points[2].1);
    println!("  Bottom-left tag (ID {}) center: ({:.1}, {:.1})", rot_bottom_left_tag.id, src_points[3].0, src_points[3].1);

    println!("\nDestination points (pixels at {}DPI) - tag centers in output image:", dpi);
    let corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"];
    for (i, p) in dst_points.iter().enumerate() {
        println!("  {}: ({:.1}, {:.1})", corner_names[i], p.0, p.1);
    }

    // Calculate homography matrix using OpenCV - mapping pixels to pixels
    let homography_mat = compute_homography_opencv(&src_points, &dst_points)?;

    // Create calibration structure
    let calibration = CameraCalibration {
        homography_mat,
        output_width,
        output_height,
        rotated_image: Some(cropped_image.clone()),
    };

    // Save annotated cropped image if debug mode
    if let Some(prefix) = debug_prefix {
        let mut annotated_cropped = cropped_image.clone();
        let tag_centers_array: [(f64, f64); 4] = [
            src_points[0], src_points[1], src_points[2], src_points[3]
        ];
        let tag_corners_array: [(f64, f64); 4] = [
            tag_corners_in_crop[0], tag_corners_in_crop[1],
            tag_corners_in_crop[2], tag_corners_in_crop[3]
        ];
        let dst_points_array: [(f64, f64); 4] = [
            dst_points[0], dst_points[1], dst_points[2], dst_points[3]
        ];

        draw_homography_annotations(
            &mut annotated_cropped,
            &tag_centers_array,
            &tag_corners_array,
            &dst_points_array
        )?;

        let annotated_path = format!("{}_cropped_annotated.jpg", prefix);
        annotated_cropped.save(&annotated_path)?;
        println!("Saved annotated cropped image to: {}", annotated_path);

        // Also flatten the annotated image to see the mapping
        println!("Applying perspective correction to annotated image for debugging...");
        match apply_perspective_correction(&annotated_cropped, &calibration) {
            Ok(flattened_annotated) => {
                let annotated_flattened_path = format!("{}_flattened_annotated.jpg", prefix);
                flattened_annotated.save(&annotated_flattened_path)?;
                println!("Saved flattened annotated image to: {}", annotated_flattened_path);
            }
            Err(e) => {
                println!("Warning: Failed to flatten annotated image: {}", e);
            }
        }
    }

    Ok(calibration)
}

/// Detect paper rotation by comparing tag IDs at geometric positions
/// Returns rotation amount: 0=none, 1=90°CCW, 2=180°, 3=270°CCW
fn detect_paper_rotation(geo_tl_id: u32, geo_tr_id: u32, geo_br_id: u32, geo_bl_id: u32, base_id: u32) -> u32 {
    // Expected IDs for upright paper:
    // TL=base_id+0, TR=base_id+1, BR=base_id+2, BL=base_id+3

    // Check which rotation matches the detected pattern
    if geo_tl_id == base_id && geo_tr_id == base_id + 1 &&
       geo_br_id == base_id + 2 && geo_bl_id == base_id + 3 {
        return 0; // No rotation
    }

    // 90° CCW: what was at TR is now at TL, BR->TR, BL->BR, TL->BL
    if geo_tl_id == base_id + 1 && geo_tr_id == base_id + 2 &&
       geo_br_id == base_id + 3 && geo_bl_id == base_id {
        return 1;
    }

    // 180°: what was at BR is now at TL, BL->TR, TL->BR, TR->BL
    if geo_tl_id == base_id + 2 && geo_tr_id == base_id + 3 &&
       geo_br_id == base_id && geo_bl_id == base_id + 1 {
        return 2;
    }

    // 270° CCW (90° CW): what was at BL is now at TL, TL->TR, TR->BR, BR->BL
    if geo_tl_id == base_id + 3 && geo_tr_id == base_id &&
       geo_br_id == base_id + 1 && geo_bl_id == base_id + 2 {
        return 3;
    }

    // Unknown rotation
    4
}

/// Draw homography mapping annotations on the cropped image
fn draw_homography_annotations(
    img: &mut RgbImage,
    tag_centers: &[(f64, f64); 4],
    tag_corners: &[(f64, f64); 4],
    dst_points: &[(f64, f64); 4],
) -> Result<()> {
    let font = FontRef::try_from_slice(include_bytes!("../../fonts/NotoSans-Regular.ttf"))
        .context("Failed to load font")?;

    let scale_large = PxScale::from(24.0);
    let scale_small = PxScale::from(18.0);

    // Colors
    let tag_center_color = Rgb([255u8, 0u8, 255u8]);     // Magenta for tag centers (mapping points)
    let tag_corner_color = Rgb([0u8, 255u8, 255u8]);     // Cyan for tag corners
    let text_color = Rgb([255u8, 255u8, 255u8]);         // White for text

    // Draw tag centers (the points used for homography)
    for (i, center) in tag_centers.iter().enumerate() {
        let x = center.0 as i32;
        let y = center.1 as i32;

        // Draw crosshair for tag center
        let cross_size = 15;
        draw_line_segment_mut(img,
            ((x - cross_size) as f32, y as f32),
            ((x + cross_size) as f32, y as f32),
            tag_center_color);
        draw_line_segment_mut(img,
            (x as f32, (y - cross_size) as f32),
            (x as f32, (y + cross_size) as f32),
            tag_center_color);

        // Draw filled circle at center
        draw_filled_circle_mut(img, (x, y), 5, tag_center_color);

        // Draw text label with mm coordinates
        let label = format!("({:.1},{:.1})mm", dst_points[i].0, dst_points[i].1);
        let text_x = (x + 20).max(5).min(img.width() as i32 - 120);
        let text_y = (y - 20).max(5).min(img.height() as i32 - 25);
        draw_text_mut(img, text_color, text_x, text_y, scale_small, &font, &label);
    }

    // Draw tag corner quads for reference
    for i in 0..4 {
        let corners_idx = [i, (i + 1) % 4];
        for &idx in &corners_idx {
            let next_idx = (idx + 1) % 4;
            if i == 0 || idx == i {  // Only draw each edge once
                let x1 = tag_corners[idx].0 as f32;
                let y1 = tag_corners[idx].1 as f32;
                let x2 = tag_corners[next_idx].0 as f32;
                let y2 = tag_corners[next_idx].1 as f32;
                draw_line_segment_mut(img, (x1, y1), (x2, y2), tag_corner_color);
            }
        }
    }

    // Add legend
    let legend_x = 10;
    let legend_y = 10;
    draw_text_mut(img, tag_center_color, legend_x, legend_y, scale_large, &font, "+ Tag centers (homography src)");
    draw_text_mut(img, tag_corner_color, legend_x, legend_y + 30, scale_large, &font, "□ Tag boundaries");

    Ok(())
}

/// Compute homography matrix from 4 point correspondences using OpenCV
fn compute_homography_opencv(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Result<Mat> {
    if src.len() != 4 || dst.len() != 4 {
        bail!("Need exactly 4 point correspondences");
    }

    println!("Computing homography using OpenCV...");

    // Convert points to OpenCV format
    let src_points: Vector<Point2f> = src.iter()
        .map(|(x, y)| Point2f::new(*x as f32, *y as f32))
        .collect();

    let dst_points: Vector<Point2f> = dst.iter()
        .map(|(x, y)| Point2f::new(*x as f32, *y as f32))
        .collect();

    // Compute homography using OpenCV with method 0 (normalized DLT)
    // For exactly 4 points, this uses Hartley's normalization for numerical stability
    let homography = find_homography(&src_points, &dst_points, &mut Mat::default(), 0, 3.0)
        .context("Failed to compute homography")?;

    println!("\nHomography matrix:");
    for row in 0..3 {
        print!("  [");
        for col in 0..3 {
            let val: f64 = *homography.at_2d(row, col).context("Failed to read homography value")?;
            print!(" {:12.6}", val);
        }
        println!(" ]");
    }

    // Verify homography
    println!("\nVerifying homography accuracy:");
    for i in 0..4 {
        let x = src[i].0;
        let y = src[i].1;
        let h00: f64 = *homography.at_2d(0, 0)?;
        let h01: f64 = *homography.at_2d(0, 1)?;
        let h02: f64 = *homography.at_2d(0, 2)?;
        let h10: f64 = *homography.at_2d(1, 0)?;
        let h11: f64 = *homography.at_2d(1, 1)?;
        let h12: f64 = *homography.at_2d(1, 2)?;
        let h20: f64 = *homography.at_2d(2, 0)?;
        let h21: f64 = *homography.at_2d(2, 1)?;
        let h22: f64 = *homography.at_2d(2, 2)?;

        let w = h20 * x + h21 * y + h22;
        let mapped_x = (h00 * x + h01 * y + h02) / w;
        let mapped_y = (h10 * x + h11 * y + h12) / w;

        println!("  Point {}: ({:.1}, {:.1})px -> ({:.1}, {:.1})px, expected ({:.1}, {:.1})px, error: ({:.2}, {:.2})px",
            i, x, y, mapped_x, mapped_y, dst[i].0, dst[i].1,
            mapped_x - dst[i].0, mapped_y - dst[i].1);
    }

    Ok(homography)
}

/// Apply perspective correction to warp image to flat paper coordinates using OpenCV
pub fn apply_perspective_correction(
    source_img: &RgbImage,
    calibration: &CameraCalibration,
) -> Result<RgbImage> {
    let (src_width, src_height) = source_img.dimensions();

    println!("Warping image from {}x{} to {}x{} using OpenCV",
        src_width, src_height, calibration.output_width, calibration.output_height);

    // Convert RgbImage to OpenCV Mat
    let src_mat_raw = Mat::from_slice(source_img.as_raw())
        .context("Failed to create Mat from image")?;
    let src_mat = src_mat_raw
        .reshape_nd(3, &[src_height as i32, src_width as i32])
        .context("Failed to reshape Mat")?;

    // Create output Mat
    let mut dst_mat = Mat::default();
    let dst_size = Size::new(calibration.output_width as i32, calibration.output_height as i32);

    // Apply perspective warp
    println!("Applying perspective warp with OpenCV...");
    warp_perspective(
        &src_mat,
        &mut dst_mat,
        &calibration.homography_mat,
        dst_size,
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::default(),
    ).context("Failed to warp perspective")?;

    // Convert back to RgbImage
    let dst_data = dst_mat.data_bytes().context("Failed to get output data")?;
    let output = RgbImage::from_raw(
        calibration.output_width,
        calibration.output_height,
        dst_data.to_vec()
    ).context("Failed to create output image")?;

    println!("Warping complete!");

    Ok(output)
}
