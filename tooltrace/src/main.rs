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

    // TODO: Step 2: Calculate perspective transform
    // TODO: Step 3: Segment object
    // TODO: Step 4: Trace contour
    // TODO: Step 5: Export to vector format(s)

    println!("\nRemaining processing steps not yet implemented.");

    Ok(())
}
