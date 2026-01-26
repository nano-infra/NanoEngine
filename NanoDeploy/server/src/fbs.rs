#[allow(non_snake_case, unused_imports, dead_code, clippy::all)]
pub mod nanodeploy {
    pub mod sequence {
        include!(concat!(env!("OUT_DIR"), "/sequence_generated.rs"));
    }
    pub mod connection {
        include!(concat!(env!("OUT_DIR"), "/connection_generated.rs"));
    }
}
