pub mod config;

pub mod engine_manager;
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

pub mod engine_rpc {
    tonic::include_proto!("nanodeploy");
}

pub mod engine_adapter;
pub mod http_server;
pub mod tokenizer;
