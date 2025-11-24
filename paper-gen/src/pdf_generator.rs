use anyhow::Result;
use printpdf::*;
use printpdf::path::{PaintMode, WindingOrder};
use tooltrace_common::{PaperSize, AprilTagConfig};
use crate::marker_placement::calculate_marker_positions;

pub struct PdfGenerator {
    paper_size: PaperSize,
    tag_config: AprilTagConfig,
}

impl PdfGenerator {
    pub fn new(paper_size: PaperSize, tag_size_mm: f64) -> Self {
        // Get tag configuration specific to this paper size
        let tag_config = AprilTagConfig::for_paper_size(paper_size, tag_size_mm);

        Self {
            paper_size,
            tag_config,
        }
    }

    pub fn generate(&self, output_path: &str) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();
        let (width_mm, height_mm) = (width_mm as f32, height_mm as f32);

        // Create PDF document
        let (doc, page1, layer1) = PdfDocument::new(
            "Calibration Paper",
            Mm(width_mm),
            Mm(height_mm),
            "Layer 1"
        );

        let layer = doc.get_page(page1).get_layer(layer1);

        // Add title text
        //self.add_title(&layer, &doc)?;

        // Draw calibration grid
       // self.draw_grid(&layer)?;

        // Draw ruler markings
        //self.draw_rulers(&layer)?;

        // Add AprilTag images
        self.embed_apriltags(&layer, &doc)?;

        // Save PDF
        doc.save(&mut std::io::BufWriter::new(
            std::fs::File::create(output_path)?
        ))?;

        Ok(())
    }

    fn add_title(&self, layer: &PdfLayerReference, doc: &PdfDocumentReference) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();
        let printable_margin = 15.0; // Match the tag margin

        // Add title at top center, inside printable area
        let font = doc.add_builtin_font(BuiltinFont::Helvetica)?;
        let title = format!("ToolTrace Calibration Paper - {}", self.paper_size);

        layer.use_text(
            title,
            10.0, // Smaller font
            Mm((width_mm / 2.0 - 35.0) as f32),
            Mm((height_mm - printable_margin - 3.0) as f32), // 3mm below top printable edge
            &font
        );

        Ok(())
    }

    fn draw_grid(&self, layer: &PdfLayerReference) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();
        let (width_mm, height_mm) = (width_mm as f32, height_mm as f32);
        let grid_spacing = 10.0_f32; // 10mm major grid
        let margin = 25.0_f32; // 15mm printable + 10mm extra = 25mm from edges

        // Set line style for grid
        layer.set_outline_thickness(0.2);
        layer.set_outline_color(Color::Greyscale(Greyscale::new(0.7, None)));

        // Vertical lines
        let mut x = margin;
        while x < width_mm - margin {
            let line = Line {
                points: vec![
                    (Point::new(Mm(x), Mm(margin)), false),
                    (Point::new(Mm(x), Mm(height_mm - margin)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
            x += grid_spacing;
        }

        // Horizontal lines
        let mut y = margin;
        while y < height_mm - margin {
            let line = Line {
                points: vec![
                    (Point::new(Mm(margin), Mm(y)), false),
                    (Point::new(Mm(width_mm - margin), Mm(y)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
            y += grid_spacing;
        }

        Ok(())
    }

    fn draw_rulers(&self, layer: &PdfLayerReference) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();
        let printable_margin = 15.0; // Start rulers inside printable area
        let ruler_offset = 17.0_f32; // Position ruler 2mm inside printable margin

        // Set line style for rulers
        layer.set_outline_thickness(0.5);
        layer.set_outline_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));

        // Bottom ruler (horizontal) - only draw in printable area
        let start_x = printable_margin as i32;
        let end_x = (width_mm - printable_margin) as i32;

        for i in start_x..=end_x {
            let x = i as f32;
            let tick_height = if i % 10 == 0 {
                3.0_f32 // Major tick every 10mm
            } else if i % 5 == 0 {
                2.0_f32 // Medium tick every 5mm
            } else {
                1.0_f32 // Minor tick every 1mm
            };

            let line = Line {
                points: vec![
                    (Point::new(Mm(x), Mm(ruler_offset)), false),
                    (Point::new(Mm(x), Mm(ruler_offset + tick_height)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
        }

        // Left ruler (vertical) - only draw in printable area
        let start_y = printable_margin as i32;
        let end_y = (height_mm - printable_margin) as i32;

        for i in start_y..=end_y {
            let y = i as f32;
            let tick_width = if i % 10 == 0 {
                3.0_f32
            } else if i % 5 == 0 {
                2.0_f32
            } else {
                1.0_f32
            };

            let line = Line {
                points: vec![
                    (Point::new(Mm(ruler_offset), Mm(y)), false),
                    (Point::new(Mm(ruler_offset + tick_width), Mm(y)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
        }

        Ok(())
    }

    fn embed_apriltags(&self, layer: &PdfLayerReference, _doc: &PdfDocumentReference) -> Result<()> {
        let positions = calculate_marker_positions(self.paper_size, &self.tag_config);
        let tag_size_mm = self.tag_config.size_mm as f32;

        for (i, &(x, y)) in positions.iter().enumerate() {
            let tag_id = self.tag_config.corner_ids[i];
            self.draw_apriltag_vector(layer, tag_id, x as f32, y as f32, tag_size_mm)?;
        }

        Ok(())
    }

    fn draw_apriltag_vector(&self, layer: &PdfLayerReference, tag_id: u32, x_mm: f32, y_mm: f32, tag_size_mm: f32) -> Result<()> {
        // Get the bit pattern for this tag ID
        let bit_pattern = get_tag_pattern(tag_id)?;

        // AprilTag is 10x10 grid (1 white + 1 black border + 6x6 data + 1 black + 1 white border)
        let grid_size = 10;
        let cell_size_mm = tag_size_mm / grid_size as f32;

        // Set drawing style - black fill for tag bits
        layer.set_fill_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));
        layer.set_outline_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));
        layer.set_outline_thickness(0.0);

        // Draw each cell in the 8x8 grid
        for grid_y in 0..grid_size {
            for grid_x in 0..grid_size {
                let is_black = get_bit_value_for_vector(grid_x, grid_y, bit_pattern);

                if is_black {
                    // Calculate position for this cell (top-left corner)
                    let cell_x = x_mm + (grid_x as f32 * cell_size_mm);
                    let cell_y = y_mm + (grid_y as f32 * cell_size_mm);

                    // Draw a filled rectangle for this cell
                    let points = vec![
                        (Point::new(Mm(cell_x), Mm(cell_y)), false),
                        (Point::new(Mm(cell_x + cell_size_mm), Mm(cell_y)), false),
                        (Point::new(Mm(cell_x + cell_size_mm), Mm(cell_y + cell_size_mm)), false),
                        (Point::new(Mm(cell_x), Mm(cell_y + cell_size_mm)), false),
                    ];

                    let polygon = Polygon {
                        rings: vec![points],
                        mode: PaintMode::Fill,
                        winding_order: WindingOrder::NonZero,
                    };

                    layer.add_polygon(polygon);
                }
            }
        }

        Ok(())
    }
}

/// AprilTag 36h11 bit patterns for IDs 0-11 (from official kornia-apriltag library)
const TAG_36H11_PATTERNS: &[(u32, u64)] = &[
    (0, 0x0000000d7e00984b),
    (1, 0x0000000dda664ca7),
    (2, 0x0000000dc4a1c821),
    (3, 0x0000000e17b470e9),
    (4, 0x0000000ef91d01b1),
    (5, 0x0000000f429cdd73),
    (6, 0x000000005da29225),
    (7, 0x00000001106cba43),
    (8, 0x0000000223bed79d),
    (9, 0x000000021f51213c),
    (10, 0x000000033eb19ca6),
    (11, 0x00000003f76eb0f8),
];

/// Get the bit pattern for a tag ID
fn get_tag_pattern(tag_id: u32) -> Result<u64> {
    TAG_36H11_PATTERNS
        .iter()
        .find(|(id, _)| *id == tag_id)
        .map(|(_, pattern)| *pattern)
        .ok_or_else(|| anyhow::anyhow!("Tag ID {} not found (only 0-11 supported)", tag_id))
}

/// Get the bit value at position (x, y) in the 10x10 grid
/// Returns true for black, false for white
fn get_bit_value_for_vector(x: u32, y: u32, bit_pattern: u64) -> bool {
    // Outer WHITE border (row 0, row 9, col 0, col 9)
    if x == 0 || x == 9 || y == 0 || y == 9 {
        return false;  // White
    }

    // Inner BLACK border (row 1, row 8, col 1, col 8)
    if x == 1 || x == 8 || y == 1 || y == 8 {
        return true;  // Black
    }

    // Data area (2-7 in 10x10 grid)
    // The bit coordinates are 1-6, so we need to map:
    // Grid position (2,2) = bit position (1,1)
    // NOTE: AprilTag uses BOTTOM-LEFT origin, so flip Y
    let bit_x = x - 1;  // Convert 10x10 coords (2-7) to bit coords (1-6)
    let bit_y = 7 - (y - 1);  // Flip Y: y=2 -> bit_y=6, y=7 -> bit_y=1

    // Bit position lookup table: (bit_x, bit_y) -> bit_index
    // Based on official AprilTag 36h11 specification
    let bit_index = match (bit_x, bit_y) {
        (1, 1) => 0,  (2, 1) => 1,  (3, 1) => 2,  (4, 1) => 3,  (5, 1) => 4,  (6, 1) => 9,
        (1, 2) => 31, (2, 2) => 5,  (3, 2) => 6,  (4, 2) => 7,  (5, 2) => 14, (6, 2) => 10,
        (1, 3) => 30, (2, 3) => 34, (3, 3) => 8,  (4, 3) => 17, (5, 3) => 15, (6, 3) => 11,
        (1, 4) => 29, (2, 4) => 33, (3, 4) => 35, (4, 4) => 26, (5, 4) => 16, (6, 4) => 12,
        (1, 5) => 28, (2, 5) => 32, (3, 5) => 25, (4, 5) => 24, (5, 5) => 23, (6, 5) => 13,
        (1, 6) => 27, (2, 6) => 22, (3, 6) => 21, (4, 6) => 20, (5, 6) => 19, (6, 6) => 18,
        _ => return false,
    };

    // Extract the bit from the pattern
    // In AprilTag: bit=0 means BLACK, bit=1 means WHITE
    let bit = (bit_pattern >> bit_index) & 1;
    bit == 0  // Return true (black) when bit is 0
}
