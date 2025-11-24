// Compare our generated tag with the reference
include!("../apriltag_generator.rs");

fn main() -> Result<()> {
    let codeword: u64 = 0x0000000d7e00984b;

    println!("Tag ID 0 - Codeword: 0x{:016x}\n", codeword);

    // Print the 10x10 grid we're generating
    println!("Generated 10x10 grid:");
    println!("  0 1 2 3 4 5 6 7 8 9");
    for y in 0..10 {
        print!("{} ", y);
        for x in 0..10 {
            let is_black = get_bit_value(x, y, codeword);
            print!("{} ", if is_black { "█" } else { "░" });
        }
        println!();
    }

    println!("\n6x6 Data area (indices 2-7):");
    println!("  2 3 4 5 6 7");
    for y in 2..=7 {
        print!("{} ", y);
        for x in 2..=7 {
            let is_black = get_bit_value(x, y, codeword);
            print!("{} ", if is_black { "█" } else { "░" });
        }
        println!();
    }

    // Show the bit values
    println!("\nBit values for each position in 6x6 data:");
    for y in 2..=7 {
        for x in 2..=7 {
            let bit_x = x - 1;
            let bit_y = y - 1;
            let bit_index = match (bit_x, bit_y) {
                (1, 1) => 0,  (2, 1) => 1,  (3, 1) => 2,  (4, 1) => 3,  (5, 1) => 4,  (6, 1) => 9,
                (1, 2) => 31, (2, 2) => 5,  (3, 2) => 6,  (4, 2) => 7,  (5, 2) => 14, (6, 2) => 10,
                (1, 3) => 30, (2, 3) => 34, (3, 3) => 8,  (4, 3) => 17, (5, 3) => 15, (6, 3) => 11,
                (1, 4) => 29, (2, 4) => 33, (3, 4) => 35, (4, 4) => 26, (5, 4) => 16, (6, 4) => 12,
                (1, 5) => 28, (2, 5) => 32, (3, 5) => 25, (4, 5) => 24, (5, 5) => 23, (6, 5) => 13,
                (1, 6) => 27, (2, 6) => 22, (3, 6) => 21, (4, 6) => 20, (5, 6) => 19, (6, 6) => 18,
                _ => 99,
            };
            let bit = (codeword >> bit_index) & 1;
            print!("{}[b{}={},{}] ", if bit == 0 { "█" } else { "░" }, bit_index, bit_x, bit_y);
        }
        println!();
    }

    Ok(())
}
