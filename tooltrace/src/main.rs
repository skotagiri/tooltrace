use clap::Parser;
use anyhow::Result;
use tooltrace_common::OutputFormat;

mod detection;
mod calibration;
mod segmentation;
mod tracing;
mod export_svg;
mod export_dxf;

/// Analyze photographs of objects on calibration paper and generate vector traces
#[derive(Parser, Debug)]
#[command(name = "tooltrace")]
#[command(about = "Trace objects from calibration paper photos", long_about = None)]
struct Args {
    /// Input image file path
    #[arg(short, long)]
    input: String,

    /// Output file path (without extension)
    #[arg(short, long, default_value = "output")]
    output: String,

    /// Output format
    #[arg(short, long, value_enum, default_value = "both")]
    format: FormatArg,

    /// Enable debug mode (save intermediate visualizations)
    #[arg(short, long)]
    debug: bool,

    /// AprilTag size in millimeters
    #[arg(short, long, default_value = "25.0")]
    tag_size: f64,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum FormatArg {
    Svg,
    Dxf,
    Both,
}

impl From<FormatArg> for OutputFormat {
    fn from(arg: FormatArg) -> Self {
        match arg {
            FormatArg::Svg => OutputFormat::Svg,
            FormatArg::Dxf => OutputFormat::Dxf,
            FormatArg::Both => OutputFormat::Both,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("ToolTrace - Object Tracing Tool");
    println!("================================");
    println!("Input: {}", args.input);
    println!("Output: {}", args.output);
    println!("Format: {:?}", args.format);
    println!("Tag size: {}mm", args.tag_size);
    println!("Debug mode: {}", args.debug);
    println!();

    // Step 1: Detect AprilTags
    println!("Step 1: Detecting AprilTags...");
    let debug_path = if args.debug {
        Some(format!("{}_debug_detection.jpg", args.output))
    } else {
        None
    };
    let detections = detection::detect_apriltags(&args.input, debug_path.as_deref())?;

    if detections.is_empty() {
        anyhow::bail!("No AprilTags detected in the image. Make sure the calibration paper is visible.");
    }

    println!("\nDetected {} tag(s)", detections.len());
    for det in &detections {
        println!("  - ID {}: center at ({:.1}, {:.1})", det.id, det.center.0, det.center.1);
    }

    // Step 2: Calculate perspective transform
    println!("\nStep 2: Calculating calibration and perspective transform...");
    let input_img = image::open(&args.input)?.to_rgb8();
    let debug_prefix = if args.debug { Some(args.output.as_str()) } else { None };
    let calibration = calibration::calculate_calibration(&detections, args.tag_size, &input_img, debug_prefix)?;

    // Save rotated and cropped image if debug mode
    if args.debug {
        if let Some(ref cropped) = calibration.rotated_image {
            let cropped_path = format!("{}_cropped.jpg", args.output);
            cropped.save(&cropped_path)?;
            println!("Saved cropped image to: {}", cropped_path);
        }
    }

    // Step 3: Apply perspective correction to flatten the paper
    println!("\nStep 3: Applying perspective correction...");
    let source_for_correction = calibration.rotated_image.as_ref().unwrap_or(&input_img);
    let flattened = calibration::apply_perspective_correction(source_for_correction, &calibration)?;

    // Save the flattened image
    let flattened_path = format!("{}_flattened.jpg", args.output);
    flattened.save(&flattened_path)?;
    println!("Saved flattened image to: {}", flattened_path);

    // Step 4: Segment objects from the flattened image
    println!("\nStep 4: Segmenting objects...");

    // Calculate AprilTag exclusion regions in flattened image space (at 300 DPI)
    let dpi = 300.0;
    let pixels_per_mm = dpi / 25.4; // 11.811 pixels/mm
    let margin_mm = 15.0; // Tags are 15mm from paper edge
    let tag_size_mm = args.tag_size;

    // Calculate tag regions: [top-left, top-right, bottom-right, bottom-left]
    let tag_regions: Vec<(i32, i32, i32, i32)> = vec![
        // Top-left tag
        (
            (margin_mm * pixels_per_mm) as i32,
            (margin_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Top-right tag
        (
            (calibration.output_width as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (margin_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Bottom-right tag
        (
            (calibration.output_width as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (calibration.output_height as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Bottom-left tag
        (
            (margin_mm * pixels_per_mm) as i32,
            (calibration.output_height as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
    ];

    println!("Calculated {} AprilTag exclusion regions", tag_regions.len());

    let debug_contours_path = if args.debug {
        Some(format!("{}_contours.jpg", args.output))
    } else {
        None
    };
    let contours_pixels = segmentation::segment_object(&flattened, &tag_regions, debug_contours_path.as_deref())?;

    if contours_pixels.is_empty() {
        println!("\nNo contours detected. Try adjusting the object placement or lighting.");
        return Ok(());
    }

    // Step 5: Convert pixel coordinates to millimeters
    println!("\nStep 5: Converting coordinates to millimeters...");
    let dpi = 300.0; // Flattened image is at 300 DPI
    let contours_mm = segmentation::pixels_to_mm(contours_pixels, dpi);

    println!("Converted {} contour(s) to millimeter coordinates", contours_mm.len());

    // Step 6: Export to vector format(s)
    println!("\nStep 6: Exporting to vector format(s)...");
    let format = OutputFormat::from(args.format);

    match format {
        OutputFormat::Svg => {
            let svg_path = format!("{}.svg", args.output);
            export_svg::export_svg(&contours_mm, &svg_path)?;
        }
        OutputFormat::Dxf => {
            let dxf_path = format!("{}.dxf", args.output);
            export_dxf::export_dxf(&contours_mm, &dxf_path)?;
        }
        OutputFormat::Both => {
            let svg_path = format!("{}.svg", args.output);
            let dxf_path = format!("{}.dxf", args.output);
            export_svg::export_svg(&contours_mm, &svg_path)?;
            export_dxf::export_dxf(&contours_mm, &dxf_path)?;
        }
    }

    println!("\n✓ Processing complete!");

    Ok(())
}
