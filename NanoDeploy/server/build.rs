use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Post-process flatbuffers-generated Rust for flatbuffers 2.x API compatibility.
/// Old flatc emits get_root/get_size_prefixed_root and unguarded unsafe calls.
fn patch_generated_flatbuffers(out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    for filename in &["sequence_generated.rs", "connection_generated.rs"] {
        let path = Path::new(out_dir).join(filename);
        if !path.exists() {
            continue;
        }
        let mut content = fs::read_to_string(&path)?;

        // flatbuffers 2.x: get_root is private module; use root_unchecked (no Verifiable)
        // Code removed because flatc 2.0.8 generates unsafe fn, making these patches erroneous.

        // sequence_generated only: wrap read_scalar_at and emplace_scalar in unsafe (required in flatbuffers 2.x)
        if *filename == "sequence_generated.rs" {
            content = content.replace(
                "flatbuffers::read_scalar_at::<Self>(buf, loc)",
                "unsafe { flatbuffers::read_scalar_at::<Self>(buf, loc) }",
            );
            content = content.replace(
                "flatbuffers::emplace_scalar::<SequenceStatus>(dst, *self)",
                "unsafe { flatbuffers::emplace_scalar::<SequenceStatus>(dst, *self) }",
            );
        }

        fs::write(&path, content)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;

    // Generate FlatBuffers
    // We assume 'flatc' is in CACHE or PATH.
    let fbs_files = ["../proto/sequence.fbs", "../proto/connection.fbs"];

    for fbs in &fbs_files {
        println!("cargo:rerun-if-changed={}", fbs);
        let status = Command::new("flatc")
            .arg("--rust")
            .arg("-o")
            .arg(&out_dir)
            .arg(fbs)
            .status();

        match status {
            Ok(s) => {
                if !s.success() {
                    eprintln!("flatc command failed for {}", fbs);
                }
            }
            Err(e) => {
                eprintln!("Failed to execute flatc: {}", e);
            }
        }
    }

    patch_generated_flatbuffers(&out_dir)?;

    // Generate gRPC/Protobuf
    println!("cargo:rerun-if-changed=../proto/engine_rpc.proto");
    tonic_build::compile_protos("../proto/engine_rpc.proto")?;

    Ok(())
}
