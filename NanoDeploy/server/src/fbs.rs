#[allow(non_snake_case, unused_imports, dead_code, clippy::all)]
pub mod nanodeploy {
    pub mod fbs {
        include!(concat!(env!("OUT_DIR"), "/sequence_generated.rs"));
    }
}
