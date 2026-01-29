pub mod config;

#[allow(warnings)]
pub mod fbs {
    #[allow(clippy::all)]
    mod sequence_generated {
        include!(concat!(env!("OUT_DIR"), "/sequence_generated.rs"));
    }
    #[allow(clippy::all)]
    mod connection_generated {
        include!(concat!(env!("OUT_DIR"), "/connection_generated.rs"));
    }

    pub use self::connection_generated::nanodeploy::fbs::*;
    pub use self::sequence_generated::nanodeploy::fbs::*;
}

pub mod peer_type;
pub use peer_type::{peer_from_table, PeerT};
pub mod engine_manager;

pub mod engine_rpc {
    tonic::include_proto!("nanodeploy");
}

pub mod engine_adapter;
pub mod http_server;
pub mod tokenizer;
