// Debug tool to visualize tag generation
include!("../apriltag_generator.rs");

fn main() -> Result<()> {
    let tag_id = 0;
    let codeword: u64 = 0x0000000d7e00984b;

    println!("Tag ID: {}", tag_id);
    println!("Codeword: 0x{:016x}", codeword);
    println!("\nBit pattern (LSB first):");
    for i in 0..36 {
        let bit = (codeword >> i) & 1;
        print!("{}", bit);
        if (i + 1) % 6 == 0 {
            println!();
        }
    }

    println!("\n\n8x8 Grid (our current rendering):");
    for y in 0..8 {
        for x in 0..8 {
            let is_black = get_bit_value(x, y, codeword);
            print!("{}", if is_black { "█" } else { "░" });
        }
        println!();
    }

    println!("\n\nBit position mapping:");
    for y in 1..=6 {
        for x in 1..=6 {
            let bit_index = match (x, y) {
                (1, 1) => 0,  (2, 1) => 1,  (3, 1) => 2,  (4, 1) => 3,  (5, 1) => 4,  (6, 1) => 9,
                (1, 2) => 31, (2, 2) => 5,  (3, 2) => 6,  (4, 2) => 7,  (5, 2) => 14, (6, 2) => 10,
                (1, 3) => 30, (2, 3) => 34, (3, 3) => 8,  (4, 3) => 17, (5, 3) => 15, (6, 3) => 11,
                (1, 4) => 29, (2, 4) => 33, (3, 4) => 35, (4, 4) => 26, (5, 4) => 16, (6, 4) => 12,
                (1, 5) => 28, (2, 5) => 32, (3, 5) => 25, (4, 5) => 24, (5, 5) => 23, (6, 5) => 13,
                (1, 6) => 27, (2, 6) => 22, (3, 6) => 21, (4, 6) => 20, (5, 6) => 19, (6, 6) => 18,
                _ => 99,
            };
            print!("{:2} ", bit_index);
        }
        println!();
    }

    Ok(())
}
