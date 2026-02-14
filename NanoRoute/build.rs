use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Post-process flatbuffers-generated Rust for flatbuffers 2.x API compatibility.
fn patch_generated_flatbuffers(
    out_dir: &str,
    filename: &str,
    enum_names: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(out_dir).join(filename);
    if !path.exists() {
        return Ok(());
    }

    let mut content = fs::read_to_string(&path)?;
    // Wrap read_scalar_at calls that aren't already inside unsafe blocks
    content = content.replace(
        "flatbuffers::read_scalar_at::<Self>(buf, loc)",
        "unsafe { flatbuffers::read_scalar_at::<Self>(buf, loc) }",
    );
    // Wrap emplace_scalar calls for all enum types
    for enum_name in enum_names {
        let from = format!("flatbuffers::emplace_scalar::<{}>(dst, *self)", enum_name);
        let to = format!(
            "unsafe {{ flatbuffers::emplace_scalar::<{}>(dst, *self) }}",
            enum_name
        );
        content = content.replace(&from, &to);
    }
    fs::write(&path, content)?;

    Ok(())
}

/// Find or build flatc. Returns path to flatc executable.
fn find_or_build_flatc() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // 1. Try flatc from PATH
    if Command::new("flatc").arg("--version").output().is_ok() {
        return Ok(PathBuf::from("flatc"));
    }

    // 2. Build from third_party/flatbuffers
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let flatbuffers_root = Path::new(&manifest_dir).join("../third_party/flatbuffers");
    let out_dir = env::var("OUT_DIR")?;
    let build_dir = Path::new(&out_dir).join("../flatbuffers-build");

    if !flatbuffers_root.join("CMakeLists.txt").exists() {
        return Err("flatc not found in PATH and third_party/flatbuffers missing. Install: apt install flatbuffers-compiler, or run: git submodule update --init".into());
    }

    fs::create_dir_all(&build_dir)?;

    let cmake_status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg(flatbuffers_root.as_os_str())
        .arg("-DFLATBUFFERS_BUILD_TESTS=OFF")
        .arg("-DFLATBUFFERS_BUILD_FLATHASH=OFF")
        .status()?;

    if !cmake_status.success() {
        return Err("cmake configure failed for flatbuffers".into());
    }

    let make_status = Command::new("cmake")
        .current_dir(&build_dir)
        .args(["--build", ".", "--target", "flatc"])
        .status()?;

    if !make_status.success() {
        return Err("flatc build failed".into());
    }

    #[cfg(windows)]
    let flatc_exe = build_dir.join("Debug").join("flatc.exe");
    #[cfg(not(windows))]
    let flatc_exe = build_dir.join("flatc");

    if !flatc_exe.exists() {
        #[cfg(not(windows))]
        let alt = build_dir.join("bin").join("flatc");
        #[cfg(not(windows))]
        if alt.exists() {
            return Ok(alt);
        }
        return Err("flatc binary not found after build".into());
    }

    Ok(flatc_exe)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let flatc = find_or_build_flatc()?;

    let fbs_files = [
        "../NanoSequence/proto/sequence.fbs",
        "../NanoSequence/proto/packet.fbs",
    ];

    for fbs in &fbs_files {
        println!("cargo:rerun-if-changed={}", fbs);
        let status = Command::new(&flatc)
            .arg("--rust")
            .arg("-o")
            .arg(&out_dir)
            .arg(fbs)
            .status()?;

        if !status.success() {
            return Err(format!("flatc failed for {}", fbs).into());
        }
    }

    patch_generated_flatbuffers(&out_dir, "sequence_generated.rs", &["SequenceStatus"])?;
    patch_generated_flatbuffers(&out_dir, "packet_generated.rs", &["Action"])?;

    Ok(())
}
