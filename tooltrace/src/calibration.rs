// Camera calibration and perspective correction module
// Implements perspective transformation using nalgebra for homography calculation
// Uses kornia-rs for image warping

use anyhow::{Context, Result, bail};
use nalgebra::{Matrix3, Vector3};
use image::{RgbImage, Rgb};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut};
use ab_glyph::{FontRef, PxScale};
use tooltrace_common::PaperSize;
use crate::detection::TagDetection;

// Kornia imports for warping
use kornia_image::{Image as KorniaImage, ImageSize};
use kornia_imgproc::warp::warp_perspective;
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_image::allocator::CpuAllocator;

// Homography computation
use homography::{HomographyComputation, geo::Point};

pub struct CameraCalibration {
    pub inverse_homography: Matrix3<f64>,
    pub output_width: u32,
    pub output_height: u32,
    pub rotated_image: Option<RgbImage>,
}

/// Convert image::RgbImage to kornia::Image<f32, 3>
fn rgb_to_kornia(img: &RgbImage) -> Result<KorniaImage<f32, 3, CpuAllocator>> {
    let (width, height) = img.dimensions();
    let size = ImageSize {
        width: width as usize,
        height: height as usize,
    };

    // Convert u8 pixels to f32 (normalized to 0.0-1.0)
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for pixel in img.pixels() {
        data.push(pixel[0] as f32 / 255.0);
        data.push(pixel[1] as f32 / 255.0);
        data.push(pixel[2] as f32 / 255.0);
    }

    KorniaImage::new(size, data, CpuAllocator).context("Failed to create Kornia image")
}

/// Convert kornia::Image<f32, 3> back to image::RgbImage
fn kornia_to_rgb(img: &KorniaImage<f32, 3, CpuAllocator>) -> Result<RgbImage> {
    let width = img.cols() as u32;
    let height = img.rows() as u32;
    let mut rgb_img = RgbImage::new(width, height);

    let data = img.as_slice();
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = (data[idx] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (data[idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (data[idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
            rgb_img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(rgb_img)
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

    // Output dimensions match paper size
    // Use a resolution of 10 pixels/mm for good quality
    let pixels_per_mm = 10.0;
    let output_width = (width_mm * pixels_per_mm) as u32;
    let output_height = (height_mm * pixels_per_mm) as u32;

    println!("Output image size: {}x{} pixels ({:.1} pixels/mm)", output_width, output_height, pixels_per_mm);

    // Create calibration structure
    let calibration = CameraCalibration {
        inverse_homography,
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

/// Compute homography matrix from 4 point correspondences using the homography crate
fn compute_homography(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Result<Matrix3<f64>> {
    if src.len() != 4 || dst.len() != 4 {
        bail!("Need exactly 4 point correspondences");
    }

    println!("Computing homography using homography crate...");

    // Create homography computation instance
    let mut hc = HomographyComputation::new();

    // Add all point correspondences
    for i in 0..4 {
        let src_point = Point::new(src[i].0, src[i].1);
        let dst_point = Point::new(dst[i].0, dst[i].1);
        hc.add_point_correspondence(src_point, dst_point);
        println!("  Point {}: ({:.1}, {:.1})px -> ({:.1}, {:.1})mm",
            i, src[i].0, src[i].1, dst[i].0, dst[i].1);
    }

    // Compute homography
    let restrictions = hc.get_restrictions();
    let solution = restrictions.compute();

    // Extract the matrix (homography crate returns nalgebra Matrix3)
    let h = solution.matrix;

    // Normalize so that h[2,2] = 1
    let h_normalized = h / h[(2, 2)];

    println!("\nHomography matrix:");
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

/// Apply perspective correction to warp image to flat paper coordinates using kornia-rs
pub fn apply_perspective_correction(
    source_img: &RgbImage,
    calibration: &CameraCalibration,
) -> Result<RgbImage> {
    let (src_width, src_height) = source_img.dimensions();

    println!("Warping image from {}x{} to {}x{} using kornia-rs",
        src_width, src_height, calibration.output_width, calibration.output_height);

    // Convert RgbImage to kornia Image
    let src_kornia = rgb_to_kornia(source_img)?;

    // Create output kornia image
    let dst_size = ImageSize {
        width: calibration.output_width as usize,
        height: calibration.output_height as usize,
    };
    let mut dst_kornia = KorniaImage::<f32, 3, CpuAllocator>::from_size_val(dst_size, 0.0, CpuAllocator)
        .context("Failed to create output kornia image")?;

    // Build the complete transformation matrix:
    // We need to map from output pixels -> mm coords -> source pixels
    //
    // Transform chain: dst_pixel -> dst_mm -> src_mm -> src_pixel
    // 1. Scale dst pixels to mm: divide by 10
    // 2. Apply inverse homography: src_mm = H^-1 * dst_mm
    // 3. Keep in pixel space (no scaling back needed as homography already maps to pixels)

    // Create scaling matrix to convert output pixels to mm
    let scale_to_mm = Matrix3::new(
        0.1, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 1.0,
    );

    // Combined transformation: pixel -> mm -> inverse homography
    let combined = calibration.inverse_homography * scale_to_mm;

    // Convert Matrix3<f64> to [f32; 9] for kornia
    let transform: [f32; 9] = [
        combined[(0, 0)] as f32, combined[(0, 1)] as f32, combined[(0, 2)] as f32,
        combined[(1, 0)] as f32, combined[(1, 1)] as f32, combined[(1, 2)] as f32,
        combined[(2, 0)] as f32, combined[(2, 1)] as f32, combined[(2, 2)] as f32,
    ];

    // Apply perspective warp using kornia
    println!("Applying perspective warp with kornia...");
    warp_perspective(
        &src_kornia,
        &mut dst_kornia,
        &transform,
        InterpolationMode::Bilinear,
    ).context("Failed to apply perspective warp")?;

    // Convert back to RgbImage
    let output = kornia_to_rgb(&dst_kornia)?;

    println!("Warping complete!");

    Ok(output)
}
