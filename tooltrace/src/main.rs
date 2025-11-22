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
    #[arg(short, long, default_value = "50.0")]
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
    println!("Debug mode: {}", args.debug);
    println!();

    // TODO: Implement processing pipeline
    // 1. Load image
    // 2. Detect AprilTags
    // 3. Calculate perspective transform
    // 4. Segment object
    // 5. Trace contour
    // 6. Export to vector format(s)

    println!("Processing pipeline not yet implemented.");
    println!("This is a placeholder for the tooltrace CLI tool.");

    Ok(())
}
