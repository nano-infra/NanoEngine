# nanosequence-rs

Rust bindings for the NanoSequence C++ library.

## Overview

This crate provides Rust bindings for the NanoSequence C++ library, allowing Rust code to:

- Create and manipulate sequences
- Serialize and deserialize sequences
- Access block context information

## Building

First, ensure that the NanoSequence C++ library is built:

```bash
cd /path/to/NanoInfra/NanoSequence
pip install -e .
```

Then build the Rust crate:

```bash
cd nanosequence-rs
cargo build
```

## Usage

### Basic Example

```rust
use nanosequence_rs::{Sequence, BlockContextSlot, SamplingParams};

// Create a sequence
let tokens = vec![1, 2, 3, 4, 5];
let mut seq = Sequence::new(&tokens, None)?;
seq.set_seq_id(123);

// Append a token
seq.append_token(6, BlockContextSlot::Active, None);

// Get sequence information
println!("Sequence ID: {}", seq.seq_id());
println!("Number of tokens: {}", seq.num_tokens());
```

### Serialization Example

```rust
use nanosequence_rs::{Sequence, serialize_sequences, deserialize_sequences};

// Create sequences
let seq1 = Sequence::new(&[1, 2, 3], None)?;
let seq2 = Sequence::new(&[10, 20, 30], None)?;

// Serialize
let sequences = vec![&seq1, &seq2];
let mut buffer = vec![0u8; 1024 * 1024];
let written = serialize_sequences(&mut buffer, &sequences, false)?;

// Deserialize
let deserialized = deserialize_sequences(&buffer[..written])?;
```

## Integration with NanoRoute

The `nanosequence-rs` crate is already integrated into NanoRoute. You can use it via:

```rust
use nanodeploy_server::sequence_utils;

let seq = sequence_utils::create_sequence(&tokens, seq_id)?;
```

## API Documentation

See the [API documentation](https://docs.rs/nanosequence-rs) for detailed information about all available functions and types.
