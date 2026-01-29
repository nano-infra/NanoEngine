use std::env;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;

    // Generate FlatBuffers
    // We assume 'flatc' is in CACHE or PATH.
    // Since the user is running flatc manually in the shell, it should be available.
    let fbs_files = ["../proto/sequence.fbs", "../proto/connection.fbs"];

    for fbs in &fbs_files {
        println!("cargo:rerun-if-changed={}", fbs);
        let status = Command::new("flatc")
            .arg("--rust")
            .arg("--gen-object-api")
            .arg("-o")
            .arg(&out_dir)
            .arg(fbs)
            .status();

        match status {
            Ok(s) => {
                if !s.success() {
                    // It's possible flatc is not in path or failed.
                    // But we proceed and let logic fail if files are missing.
                    eprintln!("flatc command failed for {}", fbs);
                }
            }
            Err(e) => {
                eprintln!("Failed to execute flatc: {}", e);
            }
        }
    }

    // Generate gRPC/Protobuf
    println!("cargo:rerun-if-changed=../proto/engine_rpc.proto");
    tonic_build::compile_protos("../proto/engine_rpc.proto")?;

    Ok(())
}
