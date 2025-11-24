/// Visualize AprilTag 36h11 bit position mapping
/// Generates an annotated image showing which bit index corresponds to each position in the 6x6 data grid

use image::{ImageBuffer, Rgba, RgbaImage};
use imageproc::drawing::{draw_text_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};
use anyhow::Result;

/// Official AprilTag 36h11 bit position mapping from the reference C implementation
const BIT_X: [u32; 36] = [
    1, 2, 3, 4, 5, 2, 3, 4, 3, 6,  // bits 0-9
    6, 6, 6, 6, 5, 5, 5, 4, 6, 5,  // bits 10-19
    4, 3, 2, 5, 4, 3, 4, 1, 1, 1,  // bits 20-29
    1, 1, 2, 2, 2, 3,               // bits 30-35
];

const BIT_Y: [u32; 36] = [
    1, 1, 1, 1, 1, 2, 2, 2, 3, 1,  // bits 0-9
    2, 3, 4, 5, 2, 3, 4, 3, 6, 6,  // bits 10-19
    6, 6, 6, 5, 5, 5, 4, 6, 5, 4,  // bits 20-29
    3, 2, 5, 4, 3, 4,               // bits 30-35
];

fn main() -> Result<()> {
    // Create a reverse mapping: (x, y) -> bit_index
    let mut grid = [[None; 6]; 6];
    for bit_idx in 0..36 {
        let x = (BIT_X[bit_idx] - 1) as usize;  // Convert 1-6 to 0-5
        let y = (BIT_Y[bit_idx] - 1) as usize;
        grid[y][x] = Some(bit_idx);
    }

    // Image parameters
    let cell_size = 120u32;
    let border = 60u32;
    let grid_size = 6;
    let img_width = border * 2 + cell_size * grid_size;
    let img_height = border * 2 + cell_size * grid_size + 100; // Extra space for title

    let mut img: RgbaImage = ImageBuffer::from_pixel(img_width, img_height, Rgba([255, 255, 255, 255]));

    // Load font for text
    let font_data = include_bytes!("../../../assets/DejaVuSans.ttf");
    let font = FontRef::try_from_slice(font_data)
        .expect("Error loading font");

    let title_scale = PxScale::from(36.0);
    let cell_scale = PxScale::from(48.0);
    let label_scale = PxScale::from(24.0);

    // Draw title
    let title = "AprilTag 36h11 Bit Position Mapping";
    draw_text_mut(&mut img, Rgba([0, 0, 0, 255]), 80, 20, title_scale, &font, title);

    // Draw subtitle
    let subtitle = "6x6 Data Grid - Numbers show bit index in 36-bit codeword";
    draw_text_mut(&mut img, Rgba([100, 100, 100, 255]), 60, 60, label_scale, &font, subtitle);

    // Draw grid and bit numbers
    for row in 0..6 {
        for col in 0..6 {
            let x = border + col * cell_size;
            let y = border + 100 + row * cell_size;

            // Draw cell border
            let rect = Rect::at(x as i32, y as i32).of_size(cell_size, cell_size);
            draw_hollow_rect_mut(&mut img, rect, Rgba([0, 0, 0, 255]));

            // Draw bit index
            if let Some(bit_idx) = grid[row as usize][col as usize] {
                let text = format!("{}", bit_idx);
                let text_width = text.len() as u32 * 20;
                let text_x = x + (cell_size - text_width) / 2;
                let text_y = y + cell_size / 2 - 20;

                // Color-code by bit index ranges for better visualization
                let color = if bit_idx < 9 {
                    Rgba([220, 50, 50, 255])  // Red - bits 0-8
                } else if bit_idx < 18 {
                    Rgba([50, 120, 220, 255])  // Blue - bits 9-17
                } else if bit_idx < 27 {
                    Rgba([50, 180, 50, 255])  // Green - bits 18-26
                } else {
                    Rgba([180, 50, 180, 255])  // Purple - bits 27-35
                };

                draw_text_mut(&mut img, color, text_x as i32, text_y as i32, cell_scale, &font, &text);
            }

            // Draw coordinate labels
            if row == 0 {
                let coord_text = format!("x={}", col + 1);
                draw_text_mut(&mut img, Rgba([100, 100, 100, 255]),
                    (x + cell_size / 2 - 20) as i32, (border + 80) as i32, label_scale, &font, &coord_text);
            }
            if col == 0 {
                let coord_text = format!("y={}", row + 1);
                draw_text_mut(&mut img, Rgba([100, 100, 100, 255]),
                    10, (y + cell_size / 2 - 10) as i32, label_scale, &font, &coord_text);
            }
        }
    }

    // Add legend
    let legend_y = img_height - 80;
    draw_text_mut(&mut img, Rgba([220, 50, 50, 255]), border as i32, legend_y as i32, label_scale, &font, "■ Bits 0-8");
    draw_text_mut(&mut img, Rgba([50, 120, 220, 255]), (border + 140) as i32, legend_y as i32, label_scale, &font, "■ Bits 9-17");
    draw_text_mut(&mut img, Rgba([50, 180, 50, 255]), (border + 290) as i32, legend_y as i32, label_scale, &font, "■ Bits 18-26");
    draw_text_mut(&mut img, Rgba([180, 50, 180, 255]), (border + 450) as i32, legend_y as i32, label_scale, &font, "■ Bits 27-35");

    // Save image
    img.save("apriltag_bit_mapping.png")?;
    println!("✓ Generated apriltag_bit_mapping.png");
    println!("\nBit Position Mapping (y increases downward in image coordinates):");
    println!("┌─────┬─────┬─────┬─────┬─────┬─────┐");
    for row in 0..6 {
        print!("│");
        for col in 0..6 {
            if let Some(bit_idx) = grid[row][col] {
                print!(" {:2}  │", bit_idx);
            }
        }
        println!();
        if row < 5 {
            println!("├─────┼─────┼─────┼─────┼─────┼─────┤");
        }
    }
    println!("└─────┴─────┴─────┴─────┴─────┴─────┘");

    Ok(())
}
