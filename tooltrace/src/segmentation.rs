// Object segmentation module
// Implements background subtraction and edge detection to find object contours

use anyhow::Result;
use image::RgbImage;
use opencv::{
    core::{Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT, AlgorithmHint},
    imgcodecs,
    imgproc::{CHAIN_APPROX_SIMPLE, RETR_EXTERNAL},
};
use tooltrace_common::{Contour, Point2DMm};

/// Segment objects from the flattened paper image using FastSAM or edge detection fallback
/// Provides accurate instance segmentation regardless of tool color (handles white/dark tools)
/// Excludes AprilTag regions and returns only outer boundary contours for tool storage plate generation
pub fn segment_object(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)], // (x, y, width, height) in pixels at 300 DPI
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    // Try FastSAM first (best for segmenting all objects), fallback to edge detection
    let fastsam_path = "d:/data/FastSam-S.onnx";
    if std::path::Path::new(fastsam_path).exists() {
        println!("Using FastSAM model for segmentation");
        return segment_with_fastsam(flattened_image, fastsam_path, tag_regions, debug_path);
    }

    println!("No ML models found, using edge detection fallback");
    segment_with_edges(flattened_image, tag_regions, debug_path)
}

/// FastSAM based segmentation using ONNX Runtime
/// FastSAM (Fast Segment Anything Model) segments all objects in the image without class labels
fn segment_with_fastsam(
    flattened_image: &RgbImage,
    model_path: &str,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    use ort::session::Session;
    use ort::execution_providers::DirectMLExecutionProvider;
    use ort::value::TensorRef;
    use ndarray::Array;

    println!("Loading FastSAM model with ONNX Runtime from: {}", model_path);

    let img_height = flattened_image.height() as i32;
    let img_width = flattened_image.width() as i32;

    println!("Input image size: {}x{}", img_width, img_height);

    // Create ONNX Runtime session with GPU acceleration
    let mut session = Session::builder()?
        .with_execution_providers([
            DirectMLExecutionProvider::default().build(), // Try DirectML first (any DX12 GPU)
        ])?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    println!("ONNX Runtime session created with GPU acceleration (DirectML)");

    // FastSAM expects 1024x1024 input (or sometimes 640x640, we'll try 1024 first)
    let input_size = 1024;

    // Prepare input: resize and normalize
    let resized = image::imageops::resize(
        flattened_image,
        input_size as u32,
        input_size as u32,
        image::imageops::FilterType::Triangle,
    );

    // Convert to ndarray format: [1, 3, 1024, 1024] with CHW layout
    let mut input_array = Array::zeros((1, 3, input_size, input_size));

    for y in 0..input_size {
        for x in 0..input_size {
            let pixel = resized.get_pixel(x as u32, y as u32);
            input_array[[0, 0, y, x]] = pixel[0] as f32 / 255.0; // R
            input_array[[0, 1, y, x]] = pixel[1] as f32 / 255.0; // G
            input_array[[0, 2, y, x]] = pixel[2] as f32 / 255.0; // B
        }
    }

    println!("Prepared input tensor: {:?}", input_array.shape());

    // Run inference
    println!("Running FastSAM inference with ONNX Runtime...");

    let input_tensor = TensorRef::from_array_view(&input_array)?;
    let outputs = session.run(ort::inputs!["images" => input_tensor])?;

    println!("Got {} output tensors", outputs.len());

    // Debug: print all output shapes
    for (idx, (name, value)) in outputs.iter().enumerate() {
        if let Ok(tensor) = value.try_extract_tensor::<f32>() {
            println!("  Output {} '{}': shape {:?}", idx, name, tensor.0);
        }
    }

    // Process FastSAM outputs
    // Expected: output0 [1, 37, N] where N is number of anchors
    //           output1 [1, 32, H, W] mask prototypes

    // Create debug paths for segmentation overlays
    let contour_debug_path = debug_path.map(|p| {
        p.replace("_contours.jpg", "_fastsam_contours.jpg")
    });
    let masks_debug_path = debug_path.map(|p| {
        p.replace("_contours.jpg", "_fastsam_masks.jpg")
    });

    // Convert source image to OpenCV Mat for debug overlay
    let source_mat = if contour_debug_path.is_some() || masks_debug_path.is_some() {
        let img_data: Vec<u8> = flattened_image.as_raw().clone();
        let mat = Mat::from_slice(&img_data)?;
        Some(mat.reshape(3, img_height)?.try_clone()?)
    } else {
        None
    };

    let contours = process_fastsam_outputs(
        &outputs,
        img_width,
        img_height,
        input_size as i32,
        tag_regions,
        contour_debug_path.as_deref(),
        masks_debug_path.as_deref(),
        source_mat.as_ref(),
    )?;

    Ok(contours)
}

/// Calculate IoU (Intersection over Union) for two boxes
fn calculate_iou(box_a: (f32, f32, f32, f32), box_b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, aw, ah) = box_a;
    let (bx1, by1, bw, bh) = box_b;

    let ax2 = ax1 + aw;
    let ay2 = ay1 + ah;
    let bx2 = bx1 + bw;
    let by2 = by1 + bh;

    let intersect_x1 = ax1.max(bx1);
    let intersect_y1 = ay1.max(by1);
    let intersect_x2 = ax2.min(bx2);
    let intersect_y2 = ay2.min(by2);

    if intersect_x1 >= intersect_x2 || intersect_y1 >= intersect_y2 {
        return 0.0;
    }

    let intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
    let area_a = aw * ah;
    let area_b = bw * bh;
    let union = area_a + area_b - intersection;

    intersection / union
}

/// Edge-based segmentation (fallback)
fn segment_with_edges(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)],
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

/// Process FastSAM ONNX Runtime outputs to extract contours from segmentation masks
fn process_fastsam_outputs(
    outputs: &ort::session::SessionOutputs,
    img_width: i32,
    img_height: i32,
    input_size: i32,
    tag_regions: &[(i32, i32, i32, i32)],
    contour_debug_path: Option<&str>,
    masks_debug_path: Option<&str>,
    source_image: Option<&Mat>,
) -> Result<Vec<Contour>> {
    use ndarray::{ArrayView3, ArrayView4};

    println!("Processing FastSAM ONNX Runtime outputs with mask extraction...");

    // FastSAM outputs:
    // output0: shape [1, 37, N] - detections (4 bbox + 1 obj + 32 mask coeffs)
    // output1: shape [1, 32, H, W] - mask prototypes

    // Tunable parameters for FastSAM segmentation
    let conf_threshold = 0.50f32;      // Confidence threshold: higher = more selective, fewer false positives
    let nms_threshold = 0.45f32;       // NMS IoU threshold
    let mask_threshold = 0.5f32;       // Mask binarization threshold

    // Area-based filtering to remove background and noise
    let total_pixels = (img_width * img_height) as f32;
    let min_area_pixels = total_pixels * 0.0005;  // Min 0.05% of image (filter noise)
    let max_area_pixels = total_pixels * 0.70;    // Max 70% of image (filter paper background)

    println!("Area filtering: min={:.0} pixels, max={:.0} pixels", min_area_pixels, max_area_pixels);

    // Extract detection tensor (output0)
    let det_value = &outputs[0];
    let det_tensor = det_value.try_extract_tensor::<f32>()?;
    let det_shape = det_tensor.0;
    let det_data = det_tensor.1;

    println!("Detections shape: {:?}", det_shape);

    // Extract mask prototypes (output1)
    let proto_value = &outputs[1];
    let proto_tensor = proto_value.try_extract_tensor::<f32>()?;
    let proto_shape = proto_tensor.0;
    let proto_data = proto_tensor.1;

    println!("Mask prototypes shape: {:?}", proto_shape);

    if det_shape.len() < 3 {
        anyhow::bail!("Expected 3D detections tensor, got {:?}", det_shape);
    }

    // Create ArrayViews
    let det_view = ArrayView3::from_shape(
        (det_shape[0] as usize, det_shape[1] as usize, det_shape[2] as usize),
        det_data,
    )?;

    let proto_view = ArrayView4::from_shape(
        (proto_shape[0] as usize, proto_shape[1] as usize, proto_shape[2] as usize, proto_shape[3] as usize),
        proto_data,
    )?;

    let num_detections = det_shape[2] as usize;
    let proto_h = proto_shape[2] as usize;
    let proto_w = proto_shape[3] as usize;

    println!("Processing {} potential detections", num_detections);
    println!("Mask prototype size: {}x{}", proto_h, proto_w);

    // Collect detections above threshold
    struct Detection {
        bbox: (f32, f32, f32, f32), // x, y, w, h in image coordinates
        confidence: f32,
        mask_coeffs: Vec<f32>, // 32 mask coefficients
        det_idx: usize, // Original detection index
    }

    let mut detections_list = Vec::new();

    let x_scale = img_width as f32 / input_size as f32;
    let y_scale = img_height as f32 / input_size as f32;

    // Process each detection
    for i in 0..num_detections {
        // Extract box coordinates [cx, cy, w, h] (indices 0-3)
        let cx = det_view[[0, 0, i]] * x_scale;
        let cy = det_view[[0, 1, i]] * y_scale;
        let w = det_view[[0, 2, i]] * x_scale;
        let h = det_view[[0, 3, i]] * y_scale;

        // Extract objectness score (index 4)
        let objectness = det_view[[0, 4, i]];

        if objectness > conf_threshold {
            // Extract 32 mask coefficients (indices 5-36)
            let mut mask_coeffs = Vec::with_capacity(32);
            for j in 0..32 {
                mask_coeffs.push(det_view[[0, 5 + j, i]]);
            }

            detections_list.push(Detection {
                bbox: (cx, cy, w, h),
                confidence: objectness,
                mask_coeffs,
                det_idx: i,
            });
        }
    }

    println!("Found {} detections above confidence threshold", detections_list.len());

    // Simple NMS implementation
    let mut keep_indices = Vec::new();
    let mut sorted_indices: Vec<usize> = (0..detections_list.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        detections_list[b].confidence.partial_cmp(&detections_list[a].confidence).unwrap()
    });

    let mut suppressed = vec![false; detections_list.len()];

    for &idx in &sorted_indices {
        if suppressed[idx] {
            continue;
        }

        keep_indices.push(idx);

        let det_a = &detections_list[idx];
        let box_a = (
            det_a.bbox.0 - det_a.bbox.2 / 2.0,
            det_a.bbox.1 - det_a.bbox.3 / 2.0,
            det_a.bbox.2,
            det_a.bbox.3,
        );

        for &other_idx in &sorted_indices {
            if other_idx == idx || suppressed[other_idx] {
                continue;
            }

            let det_b = &detections_list[other_idx];
            let box_b = (
                det_b.bbox.0 - det_b.bbox.2 / 2.0,
                det_b.bbox.1 - det_b.bbox.3 / 2.0,
                det_b.bbox.2,
                det_b.bbox.3,
            );

            let iou = calculate_iou(box_a, box_b);
            if iou > nms_threshold {
                suppressed[other_idx] = true;
            }
        }
    }

    println!("After NMS: {} final detections", keep_indices.len());

    // Create debug images if requested
    // 1. Contour overlay on source image
    let mut contour_debug_img = if contour_debug_path.is_some() {
        if let Some(src) = source_image {
            src.try_clone()?
        } else {
            Mat::default()
        }
    } else {
        Mat::default()
    };

    // 2. Mask overlay showing segmentation regions
    let mut masks_debug_img = if masks_debug_path.is_some() {
        if let Some(src) = source_image {
            src.try_clone()?
        } else {
            Mat::default()
        }
    } else {
        Mat::default()
    };

    // Process each kept detection to generate masks and extract contours
    let mut contours_result = Vec::new();
    let mut detection_count = 0;

    for &idx in &keep_indices {
        let det = &detections_list[idx];

        println!("Processing detection {}: conf={:.2}", detection_count, det.confidence);

        // Convert to x1, y1, x2, y2
        let x1 = (det.bbox.0 - det.bbox.2 / 2.0).max(0.0) as i32;
        let y1 = (det.bbox.1 - det.bbox.3 / 2.0).max(0.0) as i32;
        let x2 = (det.bbox.0 + det.bbox.2 / 2.0).min(img_width as f32) as i32;
        let y2 = (det.bbox.1 + det.bbox.3 / 2.0).min(img_height as f32) as i32;

        // Check if overlaps with AprilTag regions
        let mut overlaps_tag = false;
        for (tx, ty, tw, th) in tag_regions {
            let tag_x2 = tx + tw;
            let tag_y2 = ty + th;

            let intersect_x1 = x1.max(*tx);
            let intersect_y1 = y1.max(*ty);
            let intersect_x2 = x2.min(tag_x2);
            let intersect_y2 = y2.min(tag_y2);

            if intersect_x1 < intersect_x2 && intersect_y1 < intersect_y2 {
                let intersection_area = ((intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)) as f64;
                let bbox_area = ((x2 - x1) * (y2 - y1)) as f64;

                if intersection_area > bbox_area * 0.1 {
                    overlaps_tag = true;
                    break;
                }
            }
        }

        if overlaps_tag {
            println!("  Excluded: overlaps with AprilTag region");
            continue;
        }

        // Generate segmentation mask from coefficients and prototypes
        // Mask = sigmoid(sum(coeffs[i] * prototypes[i]))
        let mut mask = vec![0.0f32; proto_h * proto_w];

        for proto_idx in 0..32 {
            let coeff = det.mask_coeffs[proto_idx];
            for y in 0..proto_h {
                for x in 0..proto_w {
                    let proto_val = proto_view[[0, proto_idx, y, x]];
                    mask[y * proto_w + x] += coeff * proto_val;
                }
            }
        }

        // Debug: Check mask values before sigmoid
        let mask_min = mask.iter().copied().fold(f32::INFINITY, f32::min);
        let mask_max = mask.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  Mask before sigmoid: min={:.3}, max={:.3}", mask_min, mask_max);

        // Apply sigmoid activation
        for val in mask.iter_mut() {
            *val = 1.0 / (1.0 + (-*val).exp());
        }

        // Debug: Check mask values after sigmoid
        let sig_min = mask.iter().copied().fold(f32::INFINITY, f32::min);
        let sig_max = mask.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  Mask after sigmoid: min={:.3}, max={:.3}", sig_min, sig_max);

        // Resize mask from proto_h x proto_w to full image size
        let mask_mat_temp = Mat::from_slice(&mask)?;
        let mask_mat = mask_mat_temp.reshape(1, proto_h as i32)?;
        let mut resized_mask = Mat::default();
        opencv::imgproc::resize(
            &mask_mat,
            &mut resized_mask,
            opencv::core::Size::new(img_width, img_height),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        // Debug: Check resized mask values
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        opencv::core::min_max_loc(
            &resized_mask,
            Some(&mut min_val),
            Some(&mut max_val),
            None,
            None,
            &Mat::default(),
        )?;
        println!("  Resized mask: min={:.3}, max={:.3}", min_val, max_val);

        // Threshold mask to create binary mask (values are in [0,1] range)
        let mut binary_mask = Mat::default();
        opencv::imgproc::threshold(
            &resized_mask,
            &mut binary_mask,
            mask_threshold as f64,  // Threshold directly since mask is in [0,1] range
            1.0,                     // Max value is 1.0
            opencv::imgproc::THRESH_BINARY,
        )?;

        // Count pixels above threshold
        let nonzero = opencv::core::count_non_zero(&binary_mask)?;
        println!("  Binary mask: {} pixels above threshold ({}%)",
                 nonzero, (nonzero as f32 * 100.0) / (img_width * img_height) as f32);

        // Convert to 8-bit for contour detection
        let mut mask_8u = Mat::default();
        binary_mask.convert_to(&mut mask_8u, opencv::core::CV_8U, 1.0, 0.0)?;

        // Draw segmentation mask on mask debug image if requested
        if masks_debug_path.is_some() {
            // Generate a unique color for this detection (BGR format)
            let color = Scalar::new(
                ((detection_count * 70) % 256) as f64,
                ((detection_count * 150) % 256) as f64,
                ((detection_count * 230) % 256) as f64,
                0.0,
            );

            // Create a colored overlay where mask is active
            let mut colored_overlay = Mat::default();
            masks_debug_img.try_clone()?.copy_to(&mut colored_overlay)?;

            // Fill mask region with color using copyTo with mask
            let mut color_mat = Mat::new_rows_cols_with_default(
                img_height,
                img_width,
                opencv::core::CV_8UC3,
                color,
            )?;

            // Blend colored region with original image (60% color, 40% original)
            let mut temp = Mat::default();
            opencv::core::add_weighted(
                &color_mat,
                0.6,
                &colored_overlay,
                0.4,
                0.0,
                &mut temp,
                -1,
            )?;

            // Copy only the masked region to the debug image
            temp.copy_to_masked(&mut masks_debug_img, &mask_8u)?;
        }

        // Find contours from binary mask
        let mut mask_contours = Vector::<Vector<Point>>::new();
        opencv::imgproc::find_contours(
            &mask_8u,
            &mut mask_contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        // Find the largest contour (main object)
        let mut largest_contour_idx = None;
        let mut largest_area = 0.0;

        for i in 0..mask_contours.len() {
            let contour = mask_contours.get(i)?;
            let area = opencv::imgproc::contour_area(&contour, false)?;
            if area > largest_area {
                largest_area = area;
                largest_contour_idx = Some(i);
            }
        }

        if let Some(contour_idx) = largest_contour_idx {
            let contour = mask_contours.get(contour_idx)?;

            // Filter by area to remove background (too large) and noise (too small)
            if largest_area < min_area_pixels as f64 {
                println!("  Skipped: area {:.1} pixels too small (min {:.0})", largest_area, min_area_pixels);
                continue;
            }
            if largest_area > max_area_pixels as f64 {
                println!("  Skipped: area {:.1} pixels too large (max {:.0}), likely background", largest_area, max_area_pixels);
                continue;
            }

            // Convert OpenCV contour to our Contour type
            let mut points = Vec::new();
            for j in 0..contour.len() {
                let pt = contour.get(j)?;
                points.push(Point2DMm {
                    x: pt.x as f64,
                    y: pt.y as f64,
                });
            }

            println!("  ✓ Extracted contour with {} points, area={:.1} pixels", points.len(), largest_area);

            contours_result.push(Contour {
                points,
                closed: true,
            });

            // Draw contour on contour debug image
            if contour_debug_path.is_some() {
                // Generate color for this detection
                let color = Scalar::new(
                    ((detection_count * 50) % 255) as f64,
                    ((detection_count * 100) % 255) as f64,
                    ((detection_count * 150) % 255) as f64,
                    0.0,
                );

                let mut contour_vec: Vector<Vector<Point>> = Vector::new();
                contour_vec.push(contour);
                opencv::imgproc::draw_contours(
                    &mut contour_debug_img,
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

            detection_count += 1;
        } else {
            println!("  No valid contour found in mask");
        }
    }

    // Save debug images if requested
    if let Some(path) = contour_debug_path {
        imgcodecs::imwrite(path, &contour_debug_img, &Vector::new())?;
        println!("Saved contour overlay to: {}", path);
    }

    if let Some(path) = masks_debug_path {
        imgcodecs::imwrite(path, &masks_debug_img, &Vector::new())?;
        println!("Saved segmentation masks overlay to: {}", path);
    }

    // Remove contours that are fully contained within other contours
    let filtered_contours = remove_nested_contours(contours_result, img_width, img_height)?;

    println!("Returning {} contours after removing nested contours", filtered_contours.len());

    Ok(filtered_contours)
}

/// Remove contours that are fully contained within other contours
/// Uses OpenCV's pointPolygonTest to check containment
fn remove_nested_contours(
    contours: Vec<Contour>,
    img_width: i32,
    img_height: i32,
) -> Result<Vec<Contour>> {
    use opencv::core::Point;
    use opencv::core::Vector;
    use opencv::imgproc;

    if contours.len() <= 1 {
        return Ok(contours);
    }

    println!("Checking {} contours for containment...", contours.len());

    // Convert our contours to OpenCV format for polygon testing
    let mut opencv_contours: Vec<Vector<Point>> = Vec::new();
    for contour in &contours {
        let mut pts: Vector<Point> = Vector::new();
        for pt in &contour.points {
            pts.push(Point::new(pt.x as i32, pt.y as i32));
        }
        opencv_contours.push(pts);
    }

    // Track which contours are contained within others
    let mut is_contained = vec![false; contours.len()];

    // Check each pair of contours
    for i in 0..contours.len() {
        if is_contained[i] {
            continue; // Skip if already marked as contained
        }

        for j in 0..contours.len() {
            if i == j || is_contained[j] {
                continue;
            }

            // Check if contour i is fully inside contour j
            if is_contour_inside(&opencv_contours[i], &opencv_contours[j])? {
                is_contained[i] = true;
                println!("  Contour {} is contained within contour {}", i, j);
                break;
            }
        }
    }

    // Collect only non-contained contours
    let result: Vec<Contour> = contours
        .into_iter()
        .enumerate()
        .filter_map(|(idx, contour)| {
            if is_contained[idx] {
                None
            } else {
                Some(contour)
            }
        })
        .collect();

    let removed = opencv_contours.len() - result.len();
    if removed > 0 {
        println!("Removed {} nested contour(s)", removed);
    }

    Ok(result)
}

/// Check if contour A is fully contained inside contour B
/// Samples points from contour A and tests if they're all inside B
fn is_contour_inside(contour_a: &Vector<Point>, contour_b: &Vector<Point>) -> Result<bool> {
    use opencv::imgproc;

    if contour_a.is_empty() || contour_b.is_empty() {
        return Ok(false);
    }

    // Sample every Nth point from contour A to test (for efficiency)
    // For small contours, test all points; for large ones, sample
    let sample_step = if contour_a.len() < 50 {
        1
    } else {
        contour_a.len() / 50
    };

    let mut points_tested = 0;
    let mut points_inside = 0;

    for i in (0..contour_a.len()).step_by(sample_step) {
        let pt = contour_a.get(i)?;
        let point_f64 = opencv::core::Point2f::new(pt.x as f32, pt.y as f32);

        // pointPolygonTest returns:
        // > 0 if point is inside
        // < 0 if point is outside
        // = 0 if point is on edge
        let dist = imgproc::point_polygon_test(contour_b, point_f64, false)?;

        points_tested += 1;
        if dist >= 0.0 {
            points_inside += 1;
        }
    }

    // Contour A is inside B if all (or nearly all) sampled points are inside
    // Use 95% threshold to handle edge cases and floating point precision
    let inside_ratio = points_inside as f64 / points_tested as f64;
    Ok(inside_ratio >= 0.95)
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

/// Filter contours to keep only the best object contour using smart scoring
/// Considers area, centrality, compactness, and edge proximity
/// Returns a vector with at most one contour (the best object)
pub fn filter_largest_contour(
    contours: Vec<Contour>,
    debug_path: Option<&str>,
    source_image: &RgbImage,
) -> Result<Vec<Contour>> {
    if contours.is_empty() {
        println!("No contours to filter");
        return Ok(vec![]);
    }

    if contours.len() == 1 {
        println!("Only one contour, keeping it");
        return Ok(contours);
    }

    println!("Filtering {} contours using smart object scoring...", contours.len());

    let img_width = source_image.width() as f64;
    let img_height = source_image.height() as f64;
    let img_center_x = img_width / 2.0;
    let img_center_y = img_height / 2.0;

    // Calculate scores for each contour
    let mut contour_scores: Vec<(usize, f64, ContourMetrics)> = Vec::new();

    for (idx, contour) in contours.iter().enumerate() {
        let metrics = calculate_contour_metrics(&contour.points, img_width, img_height);

        // Calculate composite score
        let area_score = calculate_area_score(metrics.area, img_width * img_height);
        let centrality_score = calculate_centrality_score(
            metrics.centroid_x,
            metrics.centroid_y,
            img_center_x,
            img_center_y,
            img_width,
            img_height
        );
        let compactness_score = metrics.compactness;
        let edge_proximity_score = calculate_edge_proximity_score(
            &metrics.bbox,
            img_width,
            img_height
        );

        // Filter out highly irregular contours (paper edges/creases have very low compactness)
        if metrics.compactness < 0.15 {
            // Skip this contour - too irregular, likely paper edge/crease/shadow
            println!("    REJECTED: compactness too low ({:.3}), likely paper artifact", metrics.compactness);
            continue;
        }

        // Weighted composite score (tuned for tool/object detection)
        // Heavy emphasis on compactness to reject paper edges/creases
        let total_score =
            area_score * 0.15 +           // Area is less important when we have shape info
            centrality_score * 0.25 +     // Centered objects are preferred
            compactness_score * 0.50 +    // STRONG preference for compact shapes (rejects paper edges)
            edge_proximity_score * 0.10;  // Penalize contours near image edges

        contour_scores.push((idx, total_score, metrics));

        println!("  Contour {}: area={:.1}px², center=({:.0},{:.0}), compact={:.3}, edge_dist={:.1}px, SCORE={:.3}",
            idx, metrics.area, metrics.centroid_x, metrics.centroid_y,
            metrics.compactness, edge_proximity_score, total_score);
    }

    // Find the best scoring contour
    let (best_idx, best_score, best_metrics) = contour_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("Best contour: index {} with score {:.3} (area={:.1}px², center=({:.0},{:.0}))",
        best_idx, best_score, best_metrics.area, best_metrics.centroid_x, best_metrics.centroid_y);

    // Save debug visualization if requested
    if let Some(path) = debug_path {
        save_best_contour_debug(&contours, *best_idx, source_image, path)?;
    }

    // Return only the best contour
    Ok(vec![contours[*best_idx].clone()])
}

/// Metrics for evaluating contour quality
#[derive(Clone, Copy)]
struct ContourMetrics {
    area: f64,
    perimeter: f64,
    compactness: f64,  // Circularity measure: 4π * area / perimeter²
    centroid_x: f64,
    centroid_y: f64,
    bbox: (f64, f64, f64, f64),  // min_x, min_y, max_x, max_y
}

/// Calculate comprehensive metrics for a contour
fn calculate_contour_metrics(points: &[Point2DMm], img_width: f64, img_height: f64) -> ContourMetrics {
    if points.len() < 3 {
        return ContourMetrics {
            area: 0.0,
            perimeter: 0.0,
            compactness: 0.0,
            centroid_x: img_width / 2.0,
            centroid_y: img_height / 2.0,
            bbox: (0.0, 0.0, 0.0, 0.0),
        };
    }

    // Calculate area using shoelace formula
    let area = calculate_contour_area(points);

    // Calculate perimeter
    let mut perimeter = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = points[j].x - points[i].x;
        let dy = points[j].y - points[i].y;
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    // Calculate compactness (circularity): 4π * area / perimeter²
    // Perfect circle = 1.0, elongated shapes < 1.0
    let compactness = if perimeter > 0.0 {
        (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter)
    } else {
        0.0
    };

    // Calculate centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for point in points {
        cx += point.x;
        cy += point.y;
    }
    cx /= points.len() as f64;
    cy /= points.len() as f64;

    // Calculate bounding box
    let min_x = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let min_y = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let max_y = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

    ContourMetrics {
        area,
        perimeter,
        compactness,
        centroid_x: cx,
        centroid_y: cy,
        bbox: (min_x, min_y, max_x, max_y),
    }
}

/// Score based on area relative to image size
/// Prefer moderate-sized objects (not too small, not too large like full paper)
fn calculate_area_score(area: f64, image_area: f64) -> f64 {
    let ratio = area / image_area;

    // Ideal object size: 5-40% of image
    // Score peaks around 15-20% of image area
    if ratio < 0.001 {
        0.0  // Too small (noise)
    } else if ratio < 0.05 {
        ratio / 0.05  // Linearly increase from 0 to 1
    } else if ratio <= 0.40 {
        1.0  // Ideal range
    } else if ratio < 0.70 {
        1.0 - (ratio - 0.40) / 0.30  // Penalize large areas (likely paper/background)
    } else {
        0.1  // Heavily penalize very large areas (definitely background)
    }
}

/// Score based on distance from image center
/// Prefer centered objects over edge objects
fn calculate_centrality_score(
    cx: f64,
    cy: f64,
    img_cx: f64,
    img_cy: f64,
    img_width: f64,
    img_height: f64,
) -> f64 {
    let dx = (cx - img_cx).abs();
    let dy = (cy - img_cy).abs();
    let max_dist = ((img_width / 2.0).powi(2) + (img_height / 2.0).powi(2)).sqrt();
    let dist = (dx * dx + dy * dy).sqrt();

    // Score decreases with distance from center
    (1.0 - (dist / max_dist)).max(0.0)
}

/// Score based on proximity to image edges
/// Penalize contours very close to edges (likely paper boundaries)
fn calculate_edge_proximity_score(
    bbox: &(f64, f64, f64, f64),
    img_width: f64,
    img_height: f64,
) -> f64 {
    let (min_x, min_y, max_x, max_y) = *bbox;

    // Calculate minimum distance to any edge
    let dist_left = min_x;
    let dist_top = min_y;
    let dist_right = img_width - max_x;
    let dist_bottom = img_height - max_y;

    let min_edge_dist = dist_left.min(dist_top).min(dist_right).min(dist_bottom);

    // If contour is within 5% of image dimension from edge, heavily penalize
    let edge_threshold = (img_width.min(img_height)) * 0.05;

    if min_edge_dist < edge_threshold {
        0.0  // Very close to edge, likely paper boundary
    } else if min_edge_dist < edge_threshold * 3.0 {
        min_edge_dist / (edge_threshold * 3.0)  // Gradually increase score
    } else {
        1.0  // Far from edges, good
    }
}

/// Calculate the area of a contour using the shoelace formula
fn calculate_contour_area(points: &[Point2DMm]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }

    (area / 2.0).abs()
}

/// Save debug image showing all contours with the best one highlighted
fn save_best_contour_debug(
    contours: &[Contour],
    best_idx: usize,
    source_image: &RgbImage,
    path: &str,
) -> Result<()> {
    use opencv::core::{Mat, Point, Scalar, Vector};
    use opencv::imgcodecs;

    // Convert source image to OpenCV Mat
    let img_data: Vec<u8> = source_image.as_raw().clone();
    let mat = Mat::from_slice(&img_data)?;
    let reshaped = mat.reshape(3, source_image.height() as i32)?;
    let mut debug_img = reshaped.try_clone()?;

    // Draw all contours in gray
    for (idx, contour) in contours.iter().enumerate() {
        let color = if idx == best_idx {
            Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for best
        } else {
            Scalar::new(128.0, 128.0, 128.0, 0.0) // Gray for others
        };

        let thickness = if idx == best_idx { 3 } else { 1 };

        // Convert to OpenCV format
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
            color,
            thickness,
            opencv::imgproc::LINE_8,
            &Mat::default(),
            i32::MAX,
            Point::new(0, 0),
        )?;
    }

    imgcodecs::imwrite(path, &debug_img, &Vector::new())?;
    println!("Saved best contour debug image to: {}", path);

    Ok(())
}
