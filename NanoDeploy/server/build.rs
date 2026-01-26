use std::process::Command;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=../proto/sequence.fbs");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Invoke flatc
    let status = Command::new("flatc")
        .args(&[
            "--rust",
            "-o", &out_dir,
            "../proto/sequence.fbs",
            "../proto/connection.fbs"
        ])
        .status()
        .expect("Failed to run flatc");

    if !status.success() {
        panic!("flatc failed with status: {}", status);
    }
}
