# ToolTrace Implementation Log

All implementation milestones in reverse chronological order.

---

## 2025-11-22: Phase 2 Complete - AprilTag Generation & PDF Rendering

### AprilTag Generator Implemented
Created a custom AprilTag 36h11 generator from scratch since no Rust generation library exists:
- **Module:** `paper-gen/src/apriltag_generator.rs`
- **Supported IDs:** 0-11 (covers all 3 paper sizes)
- **Structure:** 8x8 grid (outer black border + inner white border + 6x6 data bits)
- **Quality:** 20 pixels per bit = 160x160px images

### Paper Size Detection System
Implemented unique tag ID scheme for automatic paper size detection:
- **A4:** AprilTag IDs 0-3 (top-left, top-right, bottom-right, bottom-left)
- **US Letter:** IDs 4-7
- **A3:** IDs 8-11

This allows the tooltrace analyzer to:
1. Detect which paper size was used from the tag IDs
2. Automatically determine correct pixel-to-mm calibration
3. Validate orientation (all 4 corners must have matching paper size IDs)

### PDF Generator Enhancements
- **Real AprilTags:** Replaced black placeholder squares with actual AprilTag images
- **Image Embedding:** Successfully embedded generated PNG images into PDF
- **Version Compatibility:** Resolved image crate version mismatch (0.24 vs 0.25)
- **Tag Placement:** Correctly positioned tags at 15mm margins in all corners

### Technical Solutions
**Challenge:** No existing Rust crate for AprilTag generation
**Solution:** Implemented custom generator with hardcoded 36h11 bit patterns

**Challenge:** Image crate version mismatch (printpdf uses 0.24, we use 0.25)
**Solution:** Convert image 0.25 → raw bytes → image 0.24 → DynamicImage

**Challenge:** Automatic paper size detection
**Solution:** Unique tag ID ranges per paper size in AprilTagConfig

### Build & Test Results
```bash
✓ cargo build --bin paper-gen (compiles successfully)
✓ Generated test_a4.pdf (127KB, tag IDs 0-3)
✓ Generated test_letter.pdf (126KB, tag IDs 4-7)
✓ Generated test_a3.pdf (137KB, tag IDs 8-11)
```

### Files Modified/Created
**New Files:**
- `paper-gen/src/apriltag_generator.rs` - AprilTag generation with bit patterns

**Modified Files:**
- `tooltrace-common/src/types.rs` - Added `AprilTagConfig::for_paper_size()` and `detect_paper_size()`
- `paper-gen/src/pdf_generator.rs` - Replaced placeholders with real tag embedding
- `paper-gen/src/main.rs` - Added apriltag_generator module
- `Cargo.toml` - Enabled `embedded_images` feature for printpdf

### Code Quality
- ✓ All tests pass in apriltag_generator module
- ✓ No compiler errors, only expected unused function warnings in tooltrace stubs
- ✓ PDFs generate in ~0.3-0.4 seconds

### Phase 2 Status: ✓ COMPLETE
Paper generator is fully functional! Can generate calibration papers for all 3 sizes with real AprilTag markers.

**Ready for Phase 3:** Implement tooltrace image analysis and tag detection

---

## 2025-11-22: Phase 1 Complete - Project Infrastructure

### Workspace Setup Completed
- Created Cargo workspace with three crates:
  - `paper-gen`: Binary crate for PDF generation
  - `tooltrace`: Binary crate for image analysis
  - `tooltrace-common`: Library crate for shared types and utilities
- All dependencies configured and building successfully
- Pure Rust dependency stack chosen to avoid C/C++ build issues on Windows

### Dependency Stack Finalized
**Changed from original plan:**
- Switched from `opencv` + `apriltag` to pure Rust alternatives:
  - `kornia-rs` + `kornia-imgproc` + `kornia-io` for computer vision
  - Avoids Windows build issues with pthread.h dependencies
  - Maintains modern, memory-safe Rust implementation

**Final Dependencies:**
- `clap` v4 - CLI argument parsing
- `image` v0.25 + `imageproc` v0.25 - Image processing
- `kornia-rs` v0.1 - Pure Rust computer vision library
- `nalgebra` v0.33 - Linear algebra for transformations
- `printpdf` v0.7 - PDF generation
- `svg` v0.17 - SVG export
- `dxf` v0.6 - DXF export
- `contour_tracing` v1.0 - Outline extraction
- `anyhow` + `thiserror` - Error handling
- `chrono` - Date/time for paper-gen

### Files Created
**Common Library (tooltrace-common):**
- `src/lib.rs` - Module exports
- `src/types.rs` - Shared type definitions:
  - `PaperSize` enum (A4, Letter, A3) with mm and pt conversions
  - `AprilTagConfig` for tag configuration
  - `TagFamily` enum for tag types
  - `Point2DMm` for mm coordinates
  - `Contour` for traced outlines
  - `OutputFormat` enum for export formats

**Paper Generator (paper-gen):**
- `src/main.rs` - CLI entry point with clap argument parsing
- `src/paper_sizes.rs` - Paper size utilities
- `src/marker_placement.rs` - AprilTag corner position calculations
- `src/pdf_generator.rs` - PDF generation with:
  - Title rendering
  - 10mm calibration grid
  - mm-accurate ruler markings (1mm, 5mm, 10mm ticks)
  - AprilTag placeholder squares (actual tag images TODO)

**Analysis Tool (tooltrace):**
- `src/main.rs` - CLI entry point with full argument parsing
- `src/detection.rs` - AprilTag detection module (stub)
- `src/calibration.rs` - Camera calibration and perspective correction (stub)
- `src/segmentation.rs` - Object segmentation (stub)
- `src/tracing.rs` - Contour tracing (stub)
- `src/export_svg.rs` - SVG export (stub)
- `src/export_dxf.rs` - DXF export (stub)

### Build Verification
- ✓ Workspace compiles successfully: `cargo check --workspace`
- ✓ paper-gen binary runs: `cargo run --bin paper-gen -- --help`
- ✓ tooltrace binary runs: `cargo run --bin tooltrace -- --help`
- Both CLIs have full argument parsing and help text

### CLI Features Implemented
**paper-gen:**
```
Options:
  -o, --output <OUTPUT>      Output PDF file path [default: calibration_paper.pdf]
  -s, --size <SIZE>          Paper size [default: a4] [possible values: a4, letter, a3]
  -t, --tag-size <TAG_SIZE>  Tag size in millimeters [default: 50.0]
```

**tooltrace:**
```
Options:
  -i, --input <INPUT>        Input image file path (required)
  -o, --output <OUTPUT>      Output file path without extension [default: output]
  -f, --format <FORMAT>      Output format [default: both] [possible values: svg, dxf, both]
  -d, --debug                Enable debug mode for visualizations
  -t, --tag-size <TAG_SIZE>  AprilTag size in millimeters [default: 50.0]
```

### Updated Project Configuration
- `.gitignore` configured for:
  - Build artifacts (`/target`, `Cargo.lock`)
  - Output files (`*.pdf`, `*.svg`, `*.dxf`, images)
  - Debug outputs (`debug_*`)
  - IDE and OS files
- Edition changed from 2024 to 2021 for better stability

### Known Issues / Technical Debt
1. AprilTag generation not yet implemented - using black placeholder squares
2. All tooltrace processing modules are stubs (TODO: implement)
3. PDF generator uses basic shapes - needs actual AprilTag patterns

### Phase 1 Status: ✓ COMPLETE
All infrastructure is in place. Project compiles, binaries run, CLI parsing works.

**Ready for Phase 2:** Implement paper generator with real AprilTag images

---

## 2025-11-22: Project Initialization

### Planning Phase Completed
- Defined project architecture with two main components:
  1. **paper-gen**: Printable PDF generator with AprilTag markers
  2. **tooltrace**: CLI tool for image analysis and vector tracing

### Technology Stack Decisions
- **Language**: Rust (edition 2024)
- **Fiducial Markers**: AprilTag 36h11 family (IDs 0-3 for corners)
- **Output Formats**: SVG and DXF (both for Fusion 360 compatibility)
- **Paper Sizes**: A4 (210×297mm), US Letter (8.5×11in), A3 (297×420mm)
- **CV Library**: OpenCV + apriltag crate (battle-tested approach)
- **PDF Generation**: printpdf crate

### Key Dependencies Selected
**Core Libraries:**
- `clap` v4 - CLI argument parsing
- `opencv` + `apriltag` - Computer vision and marker detection
- `image` + `imageproc` - Image processing
- `nalgebra` - Linear algebra for transformations
- `printpdf` - PDF generation
- `svg` - SVG export
- `dxf` - DXF export
- `contour_tracing` - Outline extraction

### Project Structure
```
tooltrace/
├── paper-gen/          (Binary: PDF generator)
├── tooltrace/          (Binary: CLI analysis tool)
└── tooltrace-common/   (Library: Shared types)
```

### Success Criteria Defined
- ±1mm accuracy for objects 50-300mm in size
- Support smartphone camera photos with up to 45° perspective angle
- Fusion 360-compatible vector outputs
- Print-accurate PDF generation

### Next Steps
- Create Cargo workspace configuration
- Set up directory structure
- Configure all dependencies
- Begin Phase 2: Paper generator implementation
