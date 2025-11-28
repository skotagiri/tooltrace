# ToolTrace

Analyze photographs of objects on specially-marked paper and generate mm-accurate vector traces for CAD/CAM applications.

## Project Status

🚀 **Production Ready** - Full Pipeline Complete

Currently implemented:
- ✓ Project infrastructure and workspace setup
- ✓ CLI argument parsing for both tools
- ✓ **Paper generator with real AprilTag markers**
- ✓ **Unique tag IDs for automatic paper size detection**
- ✓ **Calibration grid and ruler markings**
- ✓ **AprilTag detection and perspective correction**
- ✓ **FastSAM-based object segmentation (GPU accelerated)**
- ✓ **SVG and DXF vector export**
- ✓ **Nested contour removal and false positive filtering**

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
- ✓ Supports A4 (210×297mm), US Letter (8.5×11in), and A3 (297×420mm)
- ✓ AprilTag 36h11 markers in corners for perspective detection
- ✓ Unique tag IDs per paper size for automatic detection:
  - A4: Tag IDs 0-3
  - US Letter: Tag IDs 4-7
  - A3: Tag IDs 8-11
- ✓ 10mm calibration grid with 1mm subdivisions
- ✓ Precise ruler markings for verification (1mm, 5mm, 10mm ticks)
- ✓ High-quality tag generation (160x160px embedded images)

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

**Implemented Features:**
- ✓ AprilTag detection for perspective correction (OpenCV + apriltag-rust)
- ✓ Automatic pixel-to-mm calibration (300 DPI output)
- ✓ FastSAM-based object segmentation (GPU accelerated with ONNX Runtime + DirectML)
- ✓ Mask-based contour extraction for precise object outlines
- ✓ Nested contour removal and false positive filtering
- ✓ AprilTag region exclusion
- ✓ SVG and DXF export for Fusion 360
- ✓ Debug visualizations (masks, contours, flattened images)

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
- **Computer Vision:** OpenCV 4.x (opencv-rust)
- **AprilTag Detection:** apriltag-rust + OpenCV
- **AI/ML Inference:** ONNX Runtime 2.0 with DirectML GPU acceleration
- **Segmentation Model:** FastSAM (Fast Segment Anything Model)
- **Image Processing:** image + imageproc crates
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

## FastSAM Model Setup

The `tooltrace` tool requires the FastSAM ONNX model for object segmentation:

1. **Download FastSAM checkpoint:**
   ```bash
   # Download FastSAM-s.pt from https://github.com/CASIA-IVA-Lab/FastSAM
   # Save to d:/data/FastSAM-s.pt (Windows) or adjust path as needed
   ```

2. **Convert to ONNX format:**
   ```bash
   # Install ultralytics in Python environment
   pip install torch ultralytics onnx

   # Run conversion script
   python convert_fastsam.py
   ```

   Example conversion script:
   ```python
   from ultralytics import YOLO
   model = YOLO("d:/data/FastSAM-s.pt")
   model.export(
       format="onnx",
       imgsz=1024,
       simplify=True,
       dynamic=False,
       opset=12,
   )
   ```

3. **Place ONNX model:**
   - Save the exported `FastSAM-s.onnx` to `d:/data/FastSAM-s.onnx`
   - Or update the path in `tooltrace/src/segmentation.rs:24`

**Fallback:** If FastSAM model is not found, tooltrace falls back to edge-based segmentation (lower quality).

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

See [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for detailed development history.

**Current Status:** ✅ Production Ready - Full pipeline operational!

### Recent Milestones
- ✅ FastSAM ONNX model integration with GPU acceleration
- ✅ Mask-based contour extraction (not just bounding boxes)
- ✅ False positive filtering with tunable parameters
- ✅ Nested contour removal algorithm
- ✅ Complete perspective correction pipeline
- ✅ SVG and DXF export ready for Fusion 360

### Performance
- **Segmentation:** ~5 seconds per image (GPU accelerated)
- **Accuracy:** Sub-millimeter precision with proper calibration
- **Memory:** ~1-2 GB during inference
- **Output:** Clean vector contours following actual object shapes

## License

MIT OR Apache-2.0

## Contributing

This project is under active development. Contributions welcome once core functionality is stable.
