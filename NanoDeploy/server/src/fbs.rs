#[allow(warnings)]
pub mod nanodeploy {
    pub mod sequence {
        include!(concat!(env!("OUT_DIR"), "/sequence_generated.rs"));
    }
    pub mod connection {
        include!(concat!(env!("OUT_DIR"), "/connection_generated.rs"));
    }
}
