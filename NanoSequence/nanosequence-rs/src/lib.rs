//! NanoSequence Rust bindings
//!
//! This crate provides Rust bindings for the NanoSequence C++ library,
//! allowing Rust code to create and manipulate sequences and serialize/deserialize them.

pub mod ffi;
pub mod sequence;
pub mod serialization;

// Re-export main types for convenience
pub use sequence::{BlockContext, BlockContextSlot, SamplingParams, Sequence, SequenceStatus};
pub use serialization::{deserialize_sequences, serialize_sequences};
