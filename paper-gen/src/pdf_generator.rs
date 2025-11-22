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
        let mut tag_config = AprilTagConfig::default();
        tag_config.size_mm = tag_size_mm;

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
        self.add_title(&layer, &doc)?;

        // Draw calibration grid
        self.draw_grid(&layer)?;

        // Draw ruler markings
        self.draw_rulers(&layer)?;

        // Add AprilTag placeholders (will add actual tags later)
        self.draw_tag_placeholders(&layer)?;

        // Save PDF
        doc.save(&mut std::io::BufWriter::new(
            std::fs::File::create(output_path)?
        ))?;

        Ok(())
    }

    fn add_title(&self, layer: &PdfLayerReference, doc: &PdfDocumentReference) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();

        // Add title at top center
        let font = doc.add_builtin_font(BuiltinFont::Helvetica)?;
        let title = format!("ToolTrace Calibration Paper - {}", self.paper_size);

        layer.use_text(
            title,
            12.0,
            Mm((width_mm / 2.0 - 40.0) as f32),
            Mm((height_mm - 5.0) as f32),
            &font
        );

        Ok(())
    }

    fn draw_grid(&self, layer: &PdfLayerReference) -> Result<()> {
        let (width_mm, height_mm) = self.paper_size.dimensions_mm();
        let (width_mm, height_mm) = (width_mm as f32, height_mm as f32);
        let grid_spacing = 10.0_f32; // 10mm major grid
        let margin = 20.0_f32; // Keep grid away from edges

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
        let margin = 5.0_f32;

        // Set line style for rulers
        layer.set_outline_thickness(0.5);
        layer.set_outline_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));

        // Bottom ruler (horizontal)
        for i in 0..=(width_mm as i32) {
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
                    (Point::new(Mm(x), Mm(margin)), false),
                    (Point::new(Mm(x), Mm(margin + tick_height)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
        }

        // Left ruler (vertical)
        for i in 0..=(height_mm as i32) {
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
                    (Point::new(Mm(margin), Mm(y)), false),
                    (Point::new(Mm(margin + tick_width), Mm(y)), false),
                ],
                is_closed: false,
            };
            layer.add_line(line);
        }

        Ok(())
    }

    fn draw_tag_placeholders(&self, layer: &PdfLayerReference) -> Result<()> {
        let positions = calculate_marker_positions(self.paper_size, &self.tag_config);
        let tag_size = self.tag_config.size_mm as f32;

        // Draw black squares as placeholders for AprilTags
        layer.set_fill_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));
        layer.set_outline_color(Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None)));
        layer.set_outline_thickness(1.0);

        for (_i, &(x, y)) in positions.iter().enumerate() {
            let (x, y) = (x as f32, y as f32);
            // Draw outer black square
            let rect = Polygon {
                rings: vec![vec![
                    (Point::new(Mm(x), Mm(y)), false),
                    (Point::new(Mm(x + tag_size), Mm(y)), false),
                    (Point::new(Mm(x + tag_size), Mm(y + tag_size)), false),
                    (Point::new(Mm(x), Mm(y + tag_size)), false),
                ]],
                mode: PaintMode::FillStroke,
                winding_order: WindingOrder::NonZero,
            };
            layer.add_polygon(rect);

            // Add ID label below tag
            // Note: In a real implementation, we would draw the actual AprilTag pattern here
            // For now, just marking position
        }

        Ok(())
    }
}
