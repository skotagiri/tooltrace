// Camera calibration and perspective correction module
// Implements perspective transformation using nalgebra for homography calculation

use anyhow::{Context, Result, bail};
use nalgebra::{Matrix3, Vector3, DMatrix, SVD};
use image::{RgbImage, Rgb, imageops};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_hollow_circle_mut, draw_text_mut};
use ab_glyph::{FontRef, PxScale};
use tooltrace_common::PaperSize;
use crate::detection::TagDetection;
use std::f64::consts::PI;

pub struct CameraCalibration {
    pub homography: Matrix3<f64>,
    pub inverse_homography: Matrix3<f64>,
    pub pixel_to_mm_scale: f64,
    pub paper_size: PaperSize,
    pub output_width: u32,
    pub output_height: u32,
    pub rotation_angle: f64,
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

    println!("Corner assignments by GEOMETRIC position:");
    println!("  Top-left: ID {} at ({:.1}, {:.1})", top_left_tag.id, top_left_tag.center.0, top_left_tag.center.1);
    println!("  Top-right: ID {} at ({:.1}, {:.1})", top_right_tag.id, top_right_tag.center.0, top_right_tag.center.1);
    println!("  Bottom-right: ID {} at ({:.1}, {:.1})", bottom_right_tag.id, bottom_right_tag.center.0, bottom_right_tag.center.1);
    println!("  Bottom-left: ID {} at ({:.1}, {:.1})", bottom_left_tag.id, bottom_left_tag.center.0, bottom_left_tag.center.1);

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

    let rotation_angle = 0.0;  // No rotation
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
    // This is more robust than trying to estimate paper corners

    let (width_mm, height_mm) = paper_size.dimensions_mm();
    let margin = 15.0;  // Distance from paper edge to tag outer corner

    // Tag centers are at margin + tag_size/2 from paper edges
    let tag_center_offset = margin + tag_size_mm / 2.0;

    // Get tag centers in cropped image coordinates
    let tag_centers_in_crop = vec![
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

    println!("Using tag centers for homography calculation:");
    println!("  Tag size: {:.1}mm, center offset from paper edge: {:.1}mm", tag_size_mm, tag_center_offset);

    // Source points are tag centers in cropped image (pixels)
    let src_points = tag_centers_in_crop.clone();

    // Destination points are tag centers in paper coordinates (mm)
    // Tags are positioned at margin + tag_size/2 from each edge
    let dst_points = vec![
        // Top-left tag center
        (tag_center_offset, tag_center_offset),
        // Top-right tag center
        (width_mm - tag_center_offset, tag_center_offset),
        // Bottom-right tag center
        (width_mm - tag_center_offset, height_mm - tag_center_offset),
        // Bottom-left tag center
        (tag_center_offset, height_mm - tag_center_offset),
    ];

    println!("Source points (pixels) - tag centers in cropped image:");
    println!("  Top-left tag (ID {}) center: ({:.1}, {:.1})", rot_top_left_tag.id, src_points[0].0, src_points[0].1);
    println!("  Top-right tag (ID {}) center: ({:.1}, {:.1})", rot_top_right_tag.id, src_points[1].0, src_points[1].1);
    println!("  Bottom-right tag (ID {}) center: ({:.1}, {:.1})", rot_bottom_right_tag.id, src_points[2].0, src_points[2].1);
    println!("  Bottom-left tag (ID {}) center: ({:.1}, {:.1})", rot_bottom_left_tag.id, src_points[3].0, src_points[3].1);

    println!("Destination points (mm):");
    let corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"];
    for (i, p) in dst_points.iter().enumerate() {
        println!("  {}: ({:.1}, {:.1})", corner_names[i], p.0, p.1);
    }

    // Calculate homography matrix using DLT (Direct Linear Transform)
    let homography = compute_homography(&src_points, &dst_points)?;

    // Calculate inverse for warping
    let inverse_homography = homography.try_inverse()
        .context("Failed to invert homography matrix")?;

    // Calculate pixel-to-mm scale
    // Use average scale from the tag distances
    let scale1 = ((dst_points[1].0 - dst_points[0].0).powi(2) + (dst_points[1].1 - dst_points[0].1).powi(2)).sqrt()
        / ((src_points[1].0 - src_points[0].0).powi(2) + (src_points[1].1 - src_points[0].1).powi(2)).sqrt();
    let scale2 = ((dst_points[2].0 - dst_points[1].0).powi(2) + (dst_points[2].1 - dst_points[1].1).powi(2)).sqrt()
        / ((src_points[2].0 - src_points[1].0).powi(2) + (src_points[2].1 - src_points[1].1).powi(2)).sqrt();
    let pixel_to_mm_scale = (scale1 + scale2) / 2.0;

    println!("Pixel-to-mm scale factor: {:.4} mm/pixel", pixel_to_mm_scale);

    // Output dimensions match paper size
    // Use a resolution of 10 pixels/mm for good quality
    let pixels_per_mm = 10.0;
    let output_width = (width_mm * pixels_per_mm) as u32;
    let output_height = (height_mm * pixels_per_mm) as u32;

    println!("Output image size: {}x{} pixels ({:.1} pixels/mm)", output_width, output_height, pixels_per_mm);

    // Create calibration structure
    let calibration = CameraCalibration {
        homography,
        inverse_homography,
        pixel_to_mm_scale,
        paper_size,
        output_width,
        output_height,
        rotation_angle,
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
    let arrow_color = Rgb([255u8, 255u8, 0u8]);          // Yellow for connecting lines
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

/// Rotate image and transform corner points
fn rotate_image_and_points(
    img: &RgbImage,
    tl: (f64, f64),
    tr: (f64, f64),
    br: (f64, f64),
    bl: (f64, f64),
    angle: f64,
) -> Result<(RgbImage, (f64, f64), (f64, f64), (f64, f64), (f64, f64))> {
    let (width, height) = img.dimensions();
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    // Calculate new image dimensions after rotation
    let cos_a = angle.cos().abs();
    let sin_a = angle.sin().abs();
    let new_width = ((width as f64) * cos_a + (height as f64) * sin_a).ceil() as u32;
    let new_height = ((height as f64) * cos_a + (width as f64) * sin_a).ceil() as u32;

    println!("  Original size: {}x{}, Rotated size: {}x{}", width, height, new_width, new_height);

    // Create output image
    let mut rotated = RgbImage::new(new_width, new_height);

    let new_center_x = new_width as f64 / 2.0;
    let new_center_y = new_height as f64 / 2.0;

    // Rotate image using inverse mapping (backward mapping)
    for y in 0..new_height {
        for x in 0..new_width {
            // Translate to origin
            let x_rel = x as f64 - new_center_x;
            let y_rel = y as f64 - new_center_y;

            // Rotate backwards
            let src_x = x_rel * angle.cos() + y_rel * angle.sin() + center_x;
            let src_y = -x_rel * angle.sin() + y_rel * angle.cos() + center_y;

            // Bilinear interpolation
            if src_x >= 0.0 && src_x < (width - 1) as f64
                && src_y >= 0.0 && src_y < (height - 1) as f64 {

                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = src_x - x0 as f64;
                let fy = src_y - y0 as f64;

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);

                let r = (
                    p00[0] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[0] as f64 * fx * (1.0 - fy) +
                    p01[0] as f64 * (1.0 - fx) * fy +
                    p11[0] as f64 * fx * fy
                ) as u8;

                let g = (
                    p00[1] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[1] as f64 * fx * (1.0 - fy) +
                    p01[1] as f64 * (1.0 - fx) * fy +
                    p11[1] as f64 * fx * fy
                ) as u8;

                let b = (
                    p00[2] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[2] as f64 * fx * (1.0 - fy) +
                    p01[2] as f64 * (1.0 - fx) * fy +
                    p11[2] as f64 * fx * fy
                ) as u8;

                rotated.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        if y % (new_height / 10).max(1) == 0 {
            println!("    Rotation progress: {:.0}%", (y as f64 / new_height as f64) * 100.0);
        }
    }

    // Transform corner points
    let rotate_point = |p: (f64, f64)| -> (f64, f64) {
        let x_rel = p.0 - center_x;
        let y_rel = p.1 - center_y;

        let new_x = x_rel * (-angle).cos() - y_rel * (-angle).sin() + new_center_x;
        let new_y = x_rel * (-angle).sin() + y_rel * (-angle).cos() + new_center_y;

        (new_x, new_y)
    };

    let rotated_tl = rotate_point(tl);
    let rotated_tr = rotate_point(tr);
    let rotated_br = rotate_point(br);
    let rotated_bl = rotate_point(bl);

    println!("  Rotation complete!");

    Ok((rotated, rotated_tl, rotated_tr, rotated_br, rotated_bl))
}

/// Normalize points for better numerical stability in homography computation
/// Returns (normalized_points, transformation_matrix)
fn normalize_points(points: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    // Compute centroid
    let n = points.len() as f64;
    let cx = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let cy = points.iter().map(|(_, y)| y).sum::<f64>() / n;

    // Compute average distance from centroid
    let avg_dist = points.iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .sum::<f64>() / n;

    // Scale so average distance is sqrt(2)
    let scale = if avg_dist > 0.0 { 2.0f64.sqrt() / avg_dist } else { 1.0 };

    // Apply normalization: translate to origin, then scale
    let normalized: Vec<(f64, f64)> = points.iter()
        .map(|(x, y)| ((x - cx) * scale, (y - cy) * scale))
        .collect();

    // Transformation matrix T such that normalized = T * original
    let t = Matrix3::new(
        scale, 0.0,   -scale * cx,
        0.0,   scale, -scale * cy,
        0.0,   0.0,   1.0,
    );

    (normalized, t)
}

/// Compute homography matrix from 4 point correspondences using normalized DLT
fn compute_homography(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Result<Matrix3<f64>> {
    if src.len() != 4 || dst.len() != 4 {
        bail!("Need exactly 4 point correspondences");
    }

    // TRY WITHOUT NORMALIZATION FIRST to debug
    println!("Computing homography WITHOUT normalization for debugging...");

    // Build the A matrix for homogeneous linear system Ah = 0
    // Each point correspondence gives 2 rows, 9 columns
    let mut a = DMatrix::<f64>::zeros(8, 9);

    for i in 0..4 {
        let (x, y) = src[i];
        let (u, v) = dst[i];

        // First row for this correspondence
        a[(i * 2, 0)] = -x;
        a[(i * 2, 1)] = -y;
        a[(i * 2, 2)] = -1.0;
        a[(i * 2, 6)] = x * u;
        a[(i * 2, 7)] = y * u;
        a[(i * 2, 8)] = u;

        // Second row for this correspondence
        a[(i * 2 + 1, 3)] = -x;
        a[(i * 2 + 1, 4)] = -y;
        a[(i * 2 + 1, 5)] = -1.0;
        a[(i * 2 + 1, 6)] = x * v;
        a[(i * 2 + 1, 7)] = y * v;
        a[(i * 2 + 1, 8)] = v;
    }

    // Solve using SVD
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.context("SVD failed to compute V^T")?;

    // Solution is the last column of V (last row of V^T)
    // V^T has size (num_cols x num_cols) = (9 x 9), we want the last row (index 8)
    let num_rows = v_t.nrows();
    let h_vec = v_t.row(num_rows - 1);

    // Reshape into 3x3 matrix
    let h = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2],
        h_vec[3], h_vec[4], h_vec[5],
        h_vec[6], h_vec[7], h_vec[8],
    );

    // Normalize so that h[2,2] = 1
    let h_normalized = h / h[(2, 2)];

    println!("Homography matrix (after denormalization and renormalization):");
    println!("{:.6}", h_normalized);

    // Verify homography by testing the correspondences
    println!("\nVerifying homography accuracy:");
    for i in 0..4 {
        let src_pt = Vector3::new(src[i].0, src[i].1, 1.0);
        let mapped = h_normalized * src_pt;
        let mapped_x = mapped[0] / mapped[2];
        let mapped_y = mapped[1] / mapped[2];
        println!("  Point {}: ({:.1}, {:.1})px -> ({:.1}, {:.1})mm, expected ({:.1}, {:.1})mm, error: ({:.2}, {:.2})mm",
            i, src[i].0, src[i].1, mapped_x, mapped_y, dst[i].0, dst[i].1,
            mapped_x - dst[i].0, mapped_y - dst[i].1);
    }

    Ok(h_normalized)
}

/// Apply perspective correction to warp image to flat paper coordinates
pub fn apply_perspective_correction(
    source_img: &RgbImage,
    calibration: &CameraCalibration,
) -> Result<RgbImage> {
    // Use the provided source image directly
    let img = source_img;
    let (src_width, src_height) = img.dimensions();
    let mut output = RgbImage::new(calibration.output_width, calibration.output_height);

    println!("Warping image from {}x{} to {}x{}",
        src_width, src_height, calibration.output_width, calibration.output_height);

    // Debug: Test a few key output positions to see where they map
    let test_positions = [
        (0, 0, "top-left corner"),
        (calibration.output_width - 1, 0, "top-right corner"),
        (calibration.output_width - 1, calibration.output_height - 1, "bottom-right corner"),
        (0, calibration.output_height - 1, "bottom-left corner"),
        (275, 275, "top-left tag center (~27.5mm, ~27.5mm)"),
    ];

    println!("\nDebug: Testing key output positions:");
    for (x, y, desc) in &test_positions {
        let mm_x = *x as f64 / 10.0;
        let mm_y = *y as f64 / 10.0;
        let src_point = Vector3::new(mm_x, mm_y, 1.0);
        let dst_point = calibration.inverse_homography * src_point;
        let src_x = dst_point[0] / dst_point[2];
        let src_y = dst_point[1] / dst_point[2];
        println!("  Output ({}, {}) = ({:.1}mm, {:.1}mm) [{}] -> Source ({:.1}px, {:.1}px)",
            x, y, mm_x, mm_y, desc, src_x, src_y);
    }
    println!();

    // For each pixel in the output image, find corresponding pixel in source
    for y in 0..calibration.output_height {
        for x in 0..calibration.output_width {
            // Convert output pixel to mm coordinates (at 10 pixels/mm)
            let mm_x = x as f64 / 10.0;
            let mm_y = y as f64 / 10.0;

            // Apply inverse homography to get source pixel coordinates
            let src_point = Vector3::new(mm_x, mm_y, 1.0);
            let dst_point = calibration.inverse_homography * src_point;

            // Normalize homogeneous coordinates
            let src_x = dst_point[0] / dst_point[2];
            let src_y = dst_point[1] / dst_point[2];

            // Bilinear interpolation
            if src_x >= 0.0 && src_x < (src_width - 1) as f64
                && src_y >= 0.0 && src_y < (src_height - 1) as f64 {

                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = src_x - x0 as f64;
                let fy = src_y - y0 as f64;

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);

                let r = (
                    p00[0] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[0] as f64 * fx * (1.0 - fy) +
                    p01[0] as f64 * (1.0 - fx) * fy +
                    p11[0] as f64 * fx * fy
                ) as u8;

                let g = (
                    p00[1] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[1] as f64 * fx * (1.0 - fy) +
                    p01[1] as f64 * (1.0 - fx) * fy +
                    p11[1] as f64 * fx * fy
                ) as u8;

                let b = (
                    p00[2] as f64 * (1.0 - fx) * (1.0 - fy) +
                    p10[2] as f64 * fx * (1.0 - fy) +
                    p01[2] as f64 * (1.0 - fx) * fy +
                    p11[2] as f64 * fx * fy
                ) as u8;

                output.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Progress indicator every 10%
        if y % (calibration.output_height / 10) == 0 {
            println!("  Warping progress: {:.0}%", (y as f64 / calibration.output_height as f64) * 100.0);
        }
    }

    println!("Warping complete!");

    Ok(output)
}
