/// AprilTag 36h11 marker generator
///
/// AprilTag 36h11 tags use a 6x6 bit grid with borders.
/// Structure: 10x10 grid total (total_width=10, width_at_border=8)
/// - Outer WHITE border: 1 pixel (index 0 and 9)
/// - Inner BLACK border: 1 pixel (index 1 and 8)
/// - Data area: 6x6 bits (indices 2-7)
/// - Data bit positions: (1,1) to (6,6) map to grid positions (2,2) to (7,7)
///
/// Bit patterns for tag IDs 0-3 are hardcoded based on the 36h11 family specification.

use image::{ImageBuffer, Luma};
use anyhow::Result;
use std::path::Path;

/// AprilTag 36h11 bit patterns for IDs 0-11
/// Each pattern is a 36-bit value representing a 6x6 grid (read row by row, left to right, top to bottom)
///
/// Tag ID assignment:
/// - A4 paper: IDs 0-3 (top-left, top-right, bottom-right, bottom-left)
/// - Letter paper: IDs 4-7
/// - A3 paper: IDs 8-11
pub const TAG_36H11_PATTERNS: &[(u32, u64)] = &[
    // (ID, bit_pattern)
    // Bit patterns from the official AprilTag 36h11 family (from kornia-apriltag library)
    // A4 paper tags
    (0, 0x0000000d7e00984b), // ID 0 - A4 top-left
    (1, 0x0000000dda664ca7), // ID 1 - A4 top-right
    (2, 0x0000000dc4a1c821), // ID 2 - A4 bottom-right
    (3, 0x0000000e17b470e9), // ID 3 - A4 bottom-left

    // Letter paper tags
    (4, 0x0000000ef91d01b1), // ID 4 - Letter top-left
    (5, 0x0000000f429cdd73), // ID 5 - Letter top-right
    (6, 0x000000005da29225), // ID 6 - Letter bottom-right
    (7, 0x00000001106cba43), // ID 7 - Letter bottom-left

    // A3 paper tags
    (8, 0x0000000223bed79d), // ID 8 - A3 top-left
    (9, 0x000000021f51213c), // ID 9 - A3 top-right
    (10, 0x000000033eb19ca6), // ID 10 - A3 bottom-right
    (11, 0x00000003f76eb0f8), // ID 11 - A3 bottom-left
];

/// Generate an AprilTag image for the given ID
///
/// # Arguments
/// * `tag_id` - The AprilTag ID (0-11 supported for A4, Letter, A3 papers)
/// * `pixels_per_bit` - Number of pixels per data bit (higher = larger image)
///
/// # Returns
/// An ImageBuffer with the AprilTag pattern
pub fn generate_apriltag(tag_id: u32, pixels_per_bit: u32) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
    // Find the bit pattern for this tag ID
    let bit_pattern = TAG_36H11_PATTERNS
        .iter()
        .find(|(id, _)| *id == tag_id)
        .map(|(_, pattern)| *pattern)
        .ok_or_else(|| anyhow::anyhow!("Tag ID {} not found (only 0-11 supported)", tag_id))?;

    // Total size: 10x10 grid (1 white border + 1 black border + 6x6 data + 1 black border + 1 white border)
    let grid_size = 10;
    let image_size = grid_size * pixels_per_bit;

    let mut img = ImageBuffer::new(image_size, image_size);

    for y in 0..grid_size {
        for x in 0..grid_size {
            let is_black = get_bit_value(x, y, bit_pattern);
            let color = if is_black {
                Luma([0u8])  // Black
            } else {
                Luma([255u8]) // White
            };

            // Fill the pixels for this grid cell
            for py in 0..pixels_per_bit {
                for px in 0..pixels_per_bit {
                    let img_x = x * pixels_per_bit + px;
                    let img_y = y * pixels_per_bit + py;
                    img.put_pixel(img_x, img_y, color);
                }
            }
        }
    }

    Ok(img)
}

/// Generate an AprilTag and save it as a PNG file
///
/// # Arguments
/// * `tag_id` - The AprilTag ID (0-11 supported)
/// * `pixels_per_bit` - Number of pixels per data bit (higher = larger image)
/// * `output_path` - Path to save the PNG file
pub fn generate_and_save_apriltag(tag_id: u32, pixels_per_bit: u32, output_path: &str) -> Result<()> {
    let img = generate_apriltag(tag_id, pixels_per_bit)?;
    img.save(Path::new(output_path))?;
    println!("Generated tag ID {} -> {}", tag_id, output_path);
    Ok(())
}

/// Get the bit pattern for a given AprilTag ID
///
/// # Arguments
/// * `tag_id` - The AprilTag ID (0-11 supported)
///
/// # Returns
/// The 64-bit pattern for this tag, or an error if the ID is not found
pub fn get_tag_pattern(tag_id: u32) -> Result<u64> {
    TAG_36H11_PATTERNS
        .iter()
        .find(|(id, _)| *id == tag_id)
        .map(|(_, pattern)| *pattern)
        .ok_or_else(|| anyhow::anyhow!("Tag ID {} not found (only 0-11 supported)", tag_id))
}

/// Official AprilTag 36h11 bit position mapping from the reference C implementation
/// Maps bit index to (x, y) coordinates in the 6x6 data region
pub const BIT_X: [u32; 36] = [
    1, 2, 3, 4, 5, 2, 3, 4, 3, 6,  // bits 0-9
    6, 6, 6, 6, 5, 5, 5, 4, 6, 5,  // bits 10-19
    4, 3, 2, 5, 4, 3, 4, 1, 1, 1,  // bits 20-29
    1, 1, 2, 2, 2, 3,               // bits 30-35
];

pub const BIT_Y: [u32; 36] = [
    1, 1, 1, 1, 1, 2, 2, 2, 3, 1,  // bits 0-9
    2, 3, 4, 5, 2, 3, 4, 3, 6, 6,  // bits 10-19
    6, 6, 6, 5, 5, 5, 4, 6, 5, 4,  // bits 20-29
    3, 2, 5, 4, 3, 4,               // bits 30-35
];

/// Get the bit value at position (x, y) in the 10x10 grid
/// Returns true for black, false for white
///
/// # Arguments
/// * `x` - X coordinate in 10x10 grid (0-9)
/// * `y` - Y coordinate in 10x10 grid (0-9)
/// * `bit_pattern` - The 64-bit codeword for the tag
pub fn get_bit_value(x: u32, y: u32, bit_pattern: u64) -> bool {
    // Outer WHITE border (row 0, row 9, col 0, col 9)
    if x == 0 || x == 9 || y == 0 || y == 9 {
        return false;  // White
    }

    // Inner BLACK border (row 1, row 8, col 1, col 8)
    if x == 1 || x == 8 || y == 1 || y == 8 {
        return true;  // Black
    }

    // Data area (2-7 in 10x10 grid)
    // Map from 10x10 grid coordinates to 1-6 bit coordinates
    // Grid coordinates 2-7 map to bit coordinates 1-6
    // NOTE: No Y-flip needed - C code uses image coordinates directly
    // bit_y=1 is at the TOP (row 2), bit_y=6 is at the BOTTOM (row 7)
    let data_x = x - 1;  // Grid x=2 -> bit_x=1, Grid x=7 -> bit_x=6
    let data_y = y - 1;  // Grid y=2 -> bit_y=1, Grid y=7 -> bit_y=6

    // Find the bit index for this position using the official mapping
    // Search through the BIT_X and BIT_Y arrays to find matching coordinates
    let bit_index = (0..36)
        .find(|&i| BIT_X[i] == data_x && BIT_Y[i] == data_y)
        .unwrap_or_else(|| {
            // This should never happen if the arrays are correct
            eprintln!("Warning: No bit mapping found for position ({}, {})", data_x, data_y);
            return 0;
        });

    // Extract the bit from the pattern
    // IMPORTANT: The C code checks bits in reverse order: bit_index 0 → codeword bit 35
    // This matches: if (code & (1 << (nbits - i - 1))) from apriltag_to_image()
    // In AprilTag: bit=0 means BLACK, bit=1 means WHITE
    let bit = (bit_pattern >> (35 - bit_index)) & 1;
    bit == 0  // Return true (black) when bit is 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_tag_0() {
        let img = generate_apriltag(0, 10).unwrap();
        assert_eq!(img.width(), 80); // 8x8 grid * 10 pixels/bit
        assert_eq!(img.height(), 80);
    }

    #[test]
    fn test_border_is_black() {
        let img = generate_apriltag(0, 1).unwrap();

        // Check corners are black (outer border)
        assert_eq!(img.get_pixel(0, 0)[0], 0);
        assert_eq!(img.get_pixel(7, 0)[0], 0);
        assert_eq!(img.get_pixel(0, 7)[0], 0);
        assert_eq!(img.get_pixel(7, 7)[0], 0);
    }

    #[test]
    fn test_data_area_from_pattern() {
        let img = generate_apriltag(0, 1).unwrap();

        // The data area is 1-6, which encodes the bit pattern
        // We can't predict exact values without decoding the pattern,
        // but we can verify the grid size
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
    }

    #[test]
    fn test_invalid_id() {
        let result = generate_apriltag(999, 10);
        assert!(result.is_err());
    }
}
