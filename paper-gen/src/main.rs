use clap::Parser;
use anyhow::Result;
use tooltrace_common::PaperSize;

mod paper_sizes;
mod pdf_generator;
mod marker_placement;

use pdf_generator::PdfGenerator;

/// Generate printable calibration papers with AprilTag fiducial markers
#[derive(Parser, Debug)]
#[command(name = "paper-gen")]
#[command(about = "Generate printable calibration papers with AprilTag markers", long_about = None)]
struct Args {
    /// Output PDF file path
    #[arg(short, long, default_value = "calibration_paper.pdf")]
    output: String,

    /// Paper size
    #[arg(short, long, value_enum, default_value = "a4")]
    size: PaperSizeArg,

    /// Tag size in millimeters (outer black square)
    #[arg(short, long, default_value = "50.0")]
    tag_size: f64,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum PaperSizeArg {
    A4,
    Letter,
    A3,
}

impl From<PaperSizeArg> for PaperSize {
    fn from(arg: PaperSizeArg) -> Self {
        match arg {
            PaperSizeArg::A4 => PaperSize::A4,
            PaperSizeArg::Letter => PaperSize::Letter,
            PaperSizeArg::A3 => PaperSize::A3,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let paper_size: PaperSize = args.size.into();

    println!("Generating calibration paper:");
    println!("  Paper size: {}", paper_size);
    println!("  Tag size: {}mm", args.tag_size);
    println!("  Output: {}", args.output);

    let generator = PdfGenerator::new(paper_size, args.tag_size);
    generator.generate(&args.output)?;

    println!("✓ PDF generated successfully: {}", args.output);
    Ok(())
}
