# ToolTrace TODO

## Current Status
**Last Updated:** 2025-11-23

**Latest Update:** Perspective correction is now working correctly! Implemented two-step approach: rotate image first, rerun tag detection on rotated image, then use accurate positions for cropping and homography calculation. All debug images are being saved properly.

## Phase 1: Project Structure Setup ✓ COMPLETE
- [x] Create workspace with two binary crates: `paper-gen` and `tooltrace`
- [x] Set up shared library crate `tooltrace-common` for common types and utilities
- [x] Configure Cargo.toml with all dependencies
- [x] Add .gitignore entries for outputs (PDFs, images, SVGs, DXFs)
- [x] Create TODO.md and IMPLEMENTATION_LOG.md for tracking

## Phase 2: Paper Generator (`paper-gen`) ✓ COMPLETE
- [x] Implement paper size definitions (A4, Letter, A3) with mm dimensions
- [x] Download/generate AprilTag 36h11 family images (IDs 0-3 for corners)
- [x] Create PDF layout engine with precise positioning
- [x] Add calibration grid rendering (10mm squares, 1mm subdivisions)
- [x] Add ruler markings along edges
- [x] Implement CLI for paper size selection
- [x] Test print accuracy with real measurements - Now generating valid PDFs

## Phase 3: CLI Tool - Core Detection (`tooltrace`) ✓ COMPLETE
- [x] Implement image loading and basic validation
- [x] Integrate AprilTag detection to find 4 corner markers
- [x] Calculate rotation angle from detected markers
- [x] Rotate image to make paper upright
- [x] Rerun tag detection on rotated image for accurate positions
- [x] Calculate homography matrix from detected markers
- [x] Implement perspective correction/warping - **NOW WORKING CORRECTLY**
- [x] Calculate pixel-to-mm scale factor using known tag size
- [x] Add debug mode to visualize detected markers
- [x] Save intermediate debug images (rotated, rotated_detection, cropped, flattened)

## Phase 4: CLI Tool - Object Segmentation
- [ ] Implement background subtraction using paper area
- [ ] Apply Canny edge detection
- [ ] Implement contour finding and filtering
- [ ] Select largest/relevant contour as object outline
- [ ] Smooth contour for cleaner output
- [ ] Convert pixel coordinates to mm coordinates
- [ ] Add debug visualization of detected outline

## Phase 5: CLI Tool - Vector Export
- [ ] Implement SVG generation from mm-coordinate contour
- [ ] Implement DXF generation from same contour
- [ ] Add metadata (scale, paper size, timestamp) to outputs
- [ ] Implement CLI flags for format selection (--svg, --dxf, --both)
- [ ] Add coordinate origin options (corner, center)
- [ ] Validate outputs can be imported into Fusion 360

## Phase 6: Testing & Refinement
- [ ] Test with various object types (irregular shapes, tools, parts)
- [ ] Test different camera angles and distances
- [ ] Measure accuracy against known dimensions
- [ ] Optimize segmentation parameters
- [ ] Add error handling and user-friendly messages
- [ ] Create example outputs and documentation

## Phase 7: Documentation
- [ ] Write README.md with usage examples
- [ ] Document calibration requirements
- [ ] Add troubleshooting guide
- [ ] Create example workflow from print → photo → trace

## Notes
- Using AprilTag 36h11 family for markers
- Target accuracy: ±1mm for 50-300mm objects
- Output formats: SVG and DXF (Fusion 360 compatible)
- Paper sizes: A4, US Letter, A3
