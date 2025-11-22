# ToolTrace TODO

## Phase 1: Project Structure Setup
- [ ] Create workspace with two binary crates: `paper-gen` and `tooltrace`
- [ ] Set up shared library crate `tooltrace-common` for common types and utilities
- [ ] Configure Cargo.toml with all dependencies
- [ ] Add .gitignore entries for outputs (PDFs, images, SVGs, DXFs)
- [ ] Create TODO.md and IMPLEMENTATION_LOG.md for tracking

## Phase 2: Paper Generator (`paper-gen`)
- [ ] Implement paper size definitions (A4, Letter, A3) with mm dimensions
- [ ] Download/generate AprilTag 36h11 family images (IDs 0-3 for corners)
- [ ] Create PDF layout engine with precise positioning
- [ ] Add calibration grid rendering (10mm squares, 1mm subdivisions)
- [ ] Add ruler markings along edges
- [ ] Implement CLI for paper size selection
- [ ] Test print accuracy with real measurements

## Phase 3: CLI Tool - Core Detection (`tooltrace`)
- [ ] Implement image loading and basic validation
- [ ] Integrate AprilTag detection to find 4 corner markers
- [ ] Calculate homography matrix from detected markers
- [ ] Implement perspective correction/warping
- [ ] Calculate pixel-to-mm scale factor using known tag size
- [ ] Add debug mode to visualize detected markers

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
