# ToolTrace

Analyze photographs of objects on specially-marked paper and generate mm-accurate vector traces for CAD/CAM applications.

## Project Status

🚧 **Under Active Development** - Phase 1 Complete

Currently implemented:
- ✓ Project infrastructure and workspace setup
- ✓ CLI argument parsing for both tools
- ✓ Basic PDF generator structure
- ⏳ AprilTag marker generation (in progress)
- ⏳ Image analysis pipeline (planned)
- ⏳ Vector export (planned)

## Overview

ToolTrace consists of two command-line tools:

### 1. `paper-gen` - Calibration Paper Generator

Generates printable PDFs with AprilTag fiducial markers and calibration grids.

**Usage:**
```bash
# Generate A4 calibration paper
cargo run --bin paper-gen

# Generate US Letter with custom tag size
cargo run --bin paper-gen -- --size letter --tag-size 60

# All options
paper-gen [OPTIONS]
  -o, --output <FILE>        Output PDF path [default: calibration_paper.pdf]
  -s, --size <SIZE>          Paper size: a4, letter, a3 [default: a4]
  -t, --tag-size <MM>        Tag size in millimeters [default: 50.0]
```

**Features:**
- Supports A4 (210×297mm), US Letter (8.5×11in), and A3 (297×420mm)
- AprilTag 36h11 markers in corners for perspective detection
- 10mm calibration grid with 1mm subdivisions
- Precise ruler markings for verification

### 2. `tooltrace` - Object Tracing Tool

Analyzes photos and extracts object outlines as vector files.

**Usage:**
```bash
# Trace object and output both SVG and DXF
cargo run --bin tooltrace -- --input photo.jpg

# Output only SVG
cargo run --bin tooltrace -- --input photo.jpg --format svg --output trace

# All options
tooltrace --input <IMAGE> [OPTIONS]
  -i, --input <FILE>         Input image file (required)
  -o, --output <NAME>        Output path without extension [default: output]
  -f, --format <FORMAT>      svg, dxf, or both [default: both]
  -d, --debug                Save intermediate visualizations
  -t, --tag-size <MM>        AprilTag size in millimeters [default: 50.0]
```

**Planned Features:**
- AprilTag detection for perspective correction
- Automatic pixel-to-mm calibration
- Object segmentation and edge detection
- Contour smoothing and optimization
- SVG and DXF export for Fusion 360

## Architecture

```
tooltrace/
├── paper-gen/          # PDF generator binary
│   ├── src/
│   │   ├── main.rs
│   │   ├── pdf_generator.rs
│   │   ├── marker_placement.rs
│   │   └── paper_sizes.rs
│   └── Cargo.toml
│
├── tooltrace/          # Image analysis binary
│   ├── src/
│   │   ├── main.rs
│   │   ├── detection.rs      # AprilTag detection
│   │   ├── calibration.rs    # Perspective correction
│   │   ├── segmentation.rs   # Object extraction
│   │   ├── tracing.rs        # Contour tracing
│   │   ├── export_svg.rs     # SVG export
│   │   └── export_dxf.rs     # DXF export
│   └── Cargo.toml
│
└── tooltrace-common/   # Shared types library
    ├── src/
    │   ├── lib.rs
    │   └── types.rs
    └── Cargo.toml
```

## Technology Stack

- **Language:** Rust 2021 Edition
- **Computer Vision:** kornia-rs (pure Rust CV library)
- **Image Processing:** image + imageproc crates
- **Linear Algebra:** nalgebra
- **PDF Generation:** printpdf
- **Vector Export:** svg + dxf crates
- **CLI:** clap v4 with derive macros

## Building

```bash
# Check all crates compile
cargo check --workspace

# Build both binaries
cargo build --release

# Run tests
cargo test --workspace

# Build documentation
cargo doc --workspace --open
```

Binaries will be in `target/release/`:
- `paper-gen.exe` (or `paper-gen` on Unix)
- `tooltrace.exe` (or `tooltrace` on Unix)

## Development Workflow

1. **Print calibration paper:**
   ```bash
   cargo run --bin paper-gen -- --output cal.pdf
   # Print cal.pdf at actual size (no scaling!)
   ```

2. **Take photo:**
   - Place object on calibration paper
   - Ensure all 4 AprilTag markers are visible
   - Photo can be at an angle (up to ~45°)
   - Use good lighting, avoid shadows

3. **Trace object:**
   ```bash
   cargo run --bin tooltrace -- --input photo.jpg --output part
   # Generates part.svg and part.dxf
   ```

4. **Import to Fusion 360:**
   - Open Fusion 360
   - Insert → Insert DXF
   - Select `part.dxf`
   - Extrude or use as sketch

## Success Criteria

- ±1mm accuracy for objects 50-300mm in size
- Works with smartphone camera photos
- Handles perspective angles up to 45°
- SVG and DXF outputs import correctly into Fusion 360
- Print-accurate PDF generation

## Implementation Progress

See [TODO.md](TODO.md) for detailed task list and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for development history.

**Current Phase:** Phase 1 ✓ Complete
**Next Phase:** Phase 2 - Implement AprilTag generation and PDF rendering

## License

MIT OR Apache-2.0

## Contributing

This project is under active development. Contributions welcome once core functionality is stable.
