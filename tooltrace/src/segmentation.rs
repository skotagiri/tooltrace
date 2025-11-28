// Object segmentation module
// Implements background subtraction and edge detection to find object contours

use anyhow::Result;
use image::RgbImage;
use opencv::{
    core::{Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT, AlgorithmHint},
    dnn,
    imgcodecs,
    imgproc::{self, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL},
    prelude::{MatTraitConstManual, MatTrait, NetTrait},
};
use tooltrace_common::{Contour, Point2DMm};

/// Segment objects from the flattened paper image using YOLOv8-seg DNN model
/// Provides accurate instance segmentation regardless of tool color (handles white/dark tools)
/// Excludes AprilTag regions and returns only outer boundary contours for tool storage plate generation
pub fn segment_object(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)], // (x, y, width, height) in pixels at 300 DPI
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    // Try FastSAM first (best for segmenting all objects), then YOLOv8-seg, then edge detection
    let fastsam_path = "d:/data/FastSam-S.onnx";
    if std::path::Path::new(fastsam_path).exists() {
        println!("Using FastSAM model for segmentation");
        return segment_with_fastsam(flattened_image, fastsam_path, tag_regions, debug_path);
    }

    let yolo_path = "d:/data/yolov8n-seg.onnx";
    if std::path::Path::new(yolo_path).exists() {
        println!("Using YOLOv8-seg model for segmentation");
        return segment_with_yolov8(flattened_image, yolo_path, tag_regions, debug_path);
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
    let contours = process_fastsam_outputs(
        &outputs,
        img_width,
        img_height,
        input_size as i32,
        tag_regions,
        debug_path,
    )?;

    Ok(contours)
}

/// YOLOv8-seg based segmentation using ONNX Runtime
fn segment_with_yolov8(
    flattened_image: &RgbImage,
    model_path: &str,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    use ort::session::Session;
    use ort::execution_providers::DirectMLExecutionProvider;
    use ort::value::TensorRef;
    use ndarray::Array;

    println!("Loading YOLOv8-seg model with ONNX Runtime from: {}", model_path);

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

    // YOLOv8 expects 640x640 input
    let input_size = 640;

    // Prepare input: resize and normalize
    let resized = image::imageops::resize(
        flattened_image,
        input_size as u32,
        input_size as u32,
        image::imageops::FilterType::Triangle,
    );

    // Convert to ndarray format: [1, 3, 640, 640] with CHW layout
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
    println!("Running YOLOv8-seg inference with ONNX Runtime...");

    let input_tensor = TensorRef::from_array_view(&input_array)?;
    let outputs = session.run(ort::inputs!["images" => input_tensor])?;

    println!("Got {} output tensors", outputs.len());

    // Debug: print all output shapes
    for (idx, (name, value)) in outputs.iter().enumerate() {
        if let Ok(tensor) = value.try_extract_tensor::<f32>() {
            println!("  Output {} '{}': shape {:?}", idx, name, tensor.0);
        }
    }

    // Process multi-scale YOLOv8-seg outputs from ONNX Runtime
    let contours = process_yolov8_multiscale_outputs(
        &outputs,
        img_width,
        img_height,
        input_size as i32,
        tag_regions,
        debug_path,
    )?;

    Ok(contours)
}

/// Process YOLOv8-seg multi-scale ONNX Runtime outputs to extract contours
fn process_yolov8_multiscale_outputs(
    outputs: &ort::session::SessionOutputs,
    img_width: i32,
    img_height: i32,
    input_size: i32,
    tag_regions: &[(i32, i32, i32, i32)],
    _debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    use ndarray::ArrayView4;

    println!("Processing multi-scale ONNX Runtime outputs...");

    // Multi-scale YOLOv8-seg outputs (without simplify):
    // Output 1: /model.22/Concat_1_output_0 [1, 144, 80, 80] - scale 1 (stride 8)
    // Output 5: /model.22/Concat_2_output_0 [1, 144, 40, 40] - scale 2 (stride 16)
    // Output 6: /model.22/Concat_3_output_0 [1, 144, 20, 20] - scale 3 (stride 32)
    // 144 channels = 4 bbox + 80 classes + 32 mask coeffs + 28 extra (likely anchors/objectness)

    let conf_threshold = 0.25f32;
    let nms_threshold = 0.45f32;

    // Process the three detection scales
    let scale_indices = [1, 5, 6]; // Indices of Concat outputs
    let strides = [8, 16, 32]; // Grid stride for each scale

    // Collect detections above threshold
    struct Detection {
        bbox: (f32, f32, f32, f32), // x, y, w, h in image coordinates
        confidence: f32,
        class_id: usize,
    }

    let mut detections_list = Vec::new();

    let x_scale = img_width as f32 / input_size as f32;
    let y_scale = img_height as f32 / input_size as f32;

    // Process each scale
    for (scale_idx, (&output_idx, &stride)) in scale_indices.iter().zip(strides.iter()).enumerate() {
        // Extract feature map for this scale
        let feature_value = &outputs[output_idx];
        let feature_tensor = feature_value.try_extract_tensor::<f32>()?;
        let feature_shape = feature_tensor.0;
        let feature_data = feature_tensor.1;

        println!("Processing scale {}: shape {:?}, stride {}", scale_idx, feature_shape, stride);

        // Create ArrayView [batch=1, channels=144, height, width]
        let feature_view = ArrayView4::from_shape(
            (
                feature_shape[0] as usize,
                feature_shape[1] as usize,
                feature_shape[2] as usize,
                feature_shape[3] as usize,
            ),
            feature_data,
        )?;

        let grid_h = feature_shape[2] as usize;
        let grid_w = feature_shape[3] as usize;

        // Process each grid cell
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                // Extract bbox regression (first 4 channels)
                // YOLOv8 format: box regression outputs that need to be decoded
                let box_x = feature_view[[0, 0, gy, gx]];
                let box_y = feature_view[[0, 1, gy, gx]];
                let box_w = feature_view[[0, 2, gy, gx]];
                let box_h = feature_view[[0, 3, gy, gx]];

                // Convert grid coordinates to image coordinates
                // YOLOv8 uses distribution focal loss, so outputs are relative distances
                let cx = (gx as f32 + box_x) * stride as f32 * x_scale;
                let cy = (gy as f32 + box_y) * stride as f32 * y_scale;
                let w = box_w * stride as f32 * x_scale;
                let h = box_h * stride as f32 * y_scale;

                // Find max class score (channels 4-83 are class probabilities)
                let mut max_score = 0.0f32;
                let mut max_class_id = 0usize;

                for class_idx in 0..80 {
                    let score = feature_view[[0, 4 + class_idx, gy, gx]];
                    if score > max_score {
                        max_score = score;
                        max_class_id = class_idx;
                    }
                }

                if max_score > conf_threshold {
                    detections_list.push(Detection {
                        bbox: (cx, cy, w, h),
                        confidence: max_score,
                        class_id: max_class_id,
                    });
                }
            }
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

    // Convert to contours
    let mut contours_result = Vec::new();

    for &idx in &keep_indices {
        let det = &detections_list[idx];

        println!("Processing detection: class={}, conf={:.2}", det.class_id, det.confidence);

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

        if !overlaps_tag {
            // Create contour from bounding box
            let points = vec![
                Point2DMm { x: x1 as f64, y: y1 as f64 },
                Point2DMm { x: x2 as f64, y: y1 as f64 },
                Point2DMm { x: x2 as f64, y: y2 as f64 },
                Point2DMm { x: x1 as f64, y: y2 as f64 },
            ];

            contours_result.push(Contour {
                points,
                closed: true,
            });
        } else {
            println!("  Excluded: overlaps with AprilTag region");
        }
    }

    println!("Returning {} contours", contours_result.len());

    Ok(contours_result)
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

/// Process YOLOv8-seg model outputs to extract contours (OpenCV DNN version - deprecated)
#[allow(dead_code)]
fn process_yolov8_outputs(
    outputs: &Vector<Mat>,
    img_width: i32,
    img_height: i32,
    input_size: i32,
    tag_regions: &[(i32, i32, i32, i32)],
    _original_img: &Mat,
    _debug_path: Option<&str>,
) -> Result<Vec<Contour>> {

    // YOLOv8-seg outputs:
    // outputs[0]: shape [1, 116, 8400] - detections (84 box+class + 32 mask proto coeffs)
    // outputs[1]: shape [1, 32, 160, 160] - mask prototypes

    if outputs.len() < 2 {
        anyhow::bail!("YOLOv8-seg model should have 2 outputs, got {}", outputs.len());
    }

    let detections_raw = outputs.get(0)?;
    let mask_protos = outputs.get(1)?;

    println!("Detections shape: {:?}", detections_raw.size()?);
    println!("Mask protos shape: {:?}", mask_protos.size()?);

    // Reshape detections from [1, 116, 8400] to [116, 8400]
    let detections = detections_raw.reshape(1, 116)?;

    let num_detections = detections.cols();
    let conf_threshold = 0.25; // Confidence threshold
    let nms_threshold = 0.45;   // NMS threshold

    println!("Processing {} potential detections", num_detections);

    // Extract boxes, confidences, and class IDs
    let mut boxes = Vector::<opencv::core::Rect>::new();
    let mut confidences = Vector::<f32>::new();
    let mut class_ids = Vector::<i32>::new();
    let mut mask_coeffs_list = Vec::new();

    let x_scale = img_width as f32 / input_size as f32;
    let y_scale = img_height as f32 / input_size as f32;

    // Process each detection
    for i in 0..num_detections {
        // Extract detection data
        let mut detection_col = Mat::default();
        detections.col(i)?.copy_to(&mut detection_col)?;

        // First 4 values: box [cx, cy, w, h]
        let cx = *detection_col.at_2d::<f32>(0, 0)? * x_scale;
        let cy = *detection_col.at_2d::<f32>(1, 0)? * y_scale;
        let w = *detection_col.at_2d::<f32>(2, 0)? * x_scale;
        let h = *detection_col.at_2d::<f32>(3, 0)? * y_scale;

        // Next 80 values: class scores (COCO classes)
        let mut max_score = 0.0f32;
        let mut max_class_id = 0i32;

        for class_idx in 0..80 {
            let score = *detection_col.at_2d::<f32>(4 + class_idx, 0)?;
            if score > max_score {
                max_score = score;
                max_class_id = class_idx as i32;
            }
        }

        if max_score > conf_threshold {
            // Convert to x1, y1, w, h format
            let x1 = (cx - w / 2.0).max(0.0) as i32;
            let y1 = (cy - h / 2.0).max(0.0) as i32;
            let width = w.min(img_width as f32 - x1 as f32) as i32;
            let height = h.min(img_height as f32 - y1 as f32) as i32;

            boxes.push(opencv::core::Rect::new(x1, y1, width, height));
            confidences.push(max_score);
            class_ids.push(max_class_id);

            // Extract 32 mask coefficients (indices 84-115)
            let mut mask_coeffs = Vec::new();
            for j in 0..32 {
                mask_coeffs.push(*detection_col.at_2d::<f32>(84 + j, 0)?);
            }
            mask_coeffs_list.push(mask_coeffs);
        }
    }

    println!("Found {} detections above confidence threshold", boxes.len());

    // Apply NMS
    let mut indices = Vector::<i32>::new();
    dnn::nms_boxes(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;

    println!("After NMS: {} final detections", indices.len());

    // Extract contours for each detection
    let mut contours_result = Vec::new();

    for idx_i in 0..indices.len() {
        let idx = indices.get(idx_i)? as usize;

        println!("Processing detection {}: class={}, conf={:.2}",
                 idx, class_ids.get(idx)?, confidences.get(idx)?);

        // TODO: Generate mask from coefficients and prototypes
        // For now, use bounding box as a simple contour
        let bbox = boxes.get(idx)?;

        // Check if overlaps with AprilTag regions
        let mut overlaps_tag = false;
        for (tx, ty, tw, th) in tag_regions {
            let tag_rect = opencv::core::Rect::new(*tx, *ty, *tw, *th);
            let intersection = bbox & tag_rect;
            let intersection_area = (intersection.width * intersection.height) as f64;
            let bbox_area = (bbox.width * bbox.height) as f64;

            if intersection_area > bbox_area * 0.1 {
                overlaps_tag = true;
                break;
            }
        }

        if !overlaps_tag {
            // Create contour from bounding box (simplified for now)
            let points = vec![
                Point2DMm { x: bbox.x as f64, y: bbox.y as f64 },
                Point2DMm { x: (bbox.x + bbox.width) as f64, y: bbox.y as f64 },
                Point2DMm { x: (bbox.x + bbox.width) as f64, y: (bbox.y + bbox.height) as f64 },
                Point2DMm { x: bbox.x as f64, y: (bbox.y + bbox.height) as f64 },
            ];

            contours_result.push(Contour {
                points,
                closed: true,
            });
        } else {
            println!("  Excluded: overlaps with AprilTag region");
        }
    }

    println!("Returning {} contours", contours_result.len());

    Ok(contours_result)
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

/// Process FastSAM ONNX Runtime outputs to extract contours
fn process_fastsam_outputs(
    outputs: &ort::session::SessionOutputs,
    img_width: i32,
    img_height: i32,
    input_size: i32,
    tag_regions: &[(i32, i32, i32, i32)],
    _debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    use ndarray::ArrayView3;

    println!("Processing FastSAM ONNX Runtime outputs...");

    // FastSAM outputs:
    // output0: shape [1, 37, N] - detections (4 bbox + 1 obj + 32 mask coeffs)
    // output1: shape [1, 32, H, H] - mask prototypes

    let conf_threshold = 0.25f32;
    let nms_threshold = 0.45f32;

    // Extract detection tensor (output0)
    let det_value = &outputs[0];
    let det_tensor = det_value.try_extract_tensor::<f32>()?;
    let det_shape = det_tensor.0;
    let det_data = det_tensor.1;

    println!("Detections shape: {:?}", det_shape);

    if det_shape.len() < 3 {
        anyhow::bail!("Expected 3D detections tensor, got {:?}", det_shape);
    }

    // Create ArrayView from raw tensor [1, 37, N]
    let det_view = ArrayView3::from_shape(
        (det_shape[0] as usize, det_shape[1] as usize, det_shape[2] as usize),
        det_data,
    )?;

    let num_detections = det_shape[2] as usize;

    println!("Processing {} potential detections", num_detections);

    // Collect detections above threshold
    struct Detection {
        bbox: (f32, f32, f32, f32), // x, y, w, h in image coordinates
        confidence: f32,
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
            detections_list.push(Detection {
                bbox: (cx, cy, w, h),
                confidence: objectness,
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

    // Convert to contours
    let mut contours_result = Vec::new();

    for &idx in &keep_indices {
        let det = &detections_list[idx];

        println!("Processing detection: conf={:.2}", det.confidence);

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

        if !overlaps_tag {
            // Create contour from bounding box
            let points = vec![
                Point2DMm { x: x1 as f64, y: y1 as f64 },
                Point2DMm { x: x2 as f64, y: y1 as f64 },
                Point2DMm { x: x2 as f64, y: y2 as f64 },
                Point2DMm { x: x1 as f64, y: y2 as f64 },
            ];

            contours_result.push(Contour {
                points,
                closed: true,
            });
        } else {
            println!("  Excluded: overlaps with AprilTag region");
        }
    }

    println!("Returning {} contours", contours_result.len());

    Ok(contours_result)
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
