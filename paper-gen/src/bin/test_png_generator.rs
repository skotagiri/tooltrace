/// Test utility to generate AprilTag PNG files for verification

// Include the apriltag_generator module
include!("../apriltag_generator.rs");

fn main() -> Result<()> {
    println!("Generating AprilTag PNG files for IDs 0-5...\n");

    let pixels_per_bit = 5; // 50 pixels per bit = 400x400px image (8x8 grid * 50)

    for tag_id in 0..=5 {
        let filename = format!("test_tag_{:03}.png", tag_id);
        generate_and_save_apriltag(tag_id, pixels_per_bit, &filename)?;
    }

    println!("\nAll tags generated successfully!");
    println!("Compare these PNG files with tags extracted from D:\\data\\downloads.jpg");

    Ok(())
}
