use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let nanosequence_dir = PathBuf::from(&manifest_dir).parent().unwrap();
    let nanosequence_csrc = nanosequence_dir.join("nanosequence").join("csrc");
    let sequence_dir = nanosequence_csrc.join("sequence");
    let proto_dir = nanosequence_dir.join("proto");

    // Find NanoInfra root
    let nanosequence_root = PathBuf::from(&manifest_dir).parent().unwrap();
    let nanoinfra_dir = nanosequence_root.parent().unwrap();

    // Tell cargo to link against the C++ library
    println!("cargo:rustc-link-lib=dylib=nanosequence");
    println!("cargo:rustc-link-lib=dylib=nanosequence_metrics");

    // Add library search paths
    let build_dir = env::var("OUT_DIR").unwrap();
    let build_path = PathBuf::from(&build_dir).parent().unwrap()
        .parent().unwrap().parent().unwrap();

    // Try to find the library in common locations
    let possible_lib_paths = vec![
        build_path.join("lib"),
        build_path.join("build").join("lib"),
        PathBuf::from("/opt/conda/lib/python3.11/site-packages/nanosequence"),
    ];

    for lib_path in possible_lib_paths {
        if lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
        }
    }

    // Rebuild if C++ headers change
    println!("cargo:rerun-if-changed={}", sequence_dir.join("sequence.h").display());
    println!("cargo:rerun-if-changed={}", sequence_dir.join("serialization.h").display());
    println!("cargo:rerun-if-changed={}", proto_dir.join("sequence.fbs").display());
}
