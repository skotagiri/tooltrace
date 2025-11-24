use anyhow::Result;
use printpdf::*;
use printpdf::path::{PaintMode, WindingOrder};
use tooltrace_common::{PaperSize, AprilTagConfig};
use crate::marker_placement::calculate_marker_positions;
use crate::apriltag_generator::{get_tag_pattern, get_bit_value};

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

        // Draw each cell in the 10x10 grid
        for grid_y in 0..grid_size {
            for grid_x in 0..grid_size {
                let is_black = get_bit_value(grid_x, grid_y, bit_pattern);

                if is_black {
                    // Calculate position for this cell (top-left corner)
                    // Note: PDF coordinates have Y=0 at bottom, but our grid has Y=0 at top
                    // So we flip the Y coordinate to match PNG generator orientation
                    let cell_x = x_mm + (grid_x as f32 * cell_size_mm);
                    let cell_y = y_mm + ((grid_size - 1 - grid_y) as f32 * cell_size_mm);

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
