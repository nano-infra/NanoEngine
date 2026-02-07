//! Utilities for working with sequences using nanosequence-rs

use nanosequence_rs::{BlockContextSlot, Sequence, SequenceStatus};
use std::sync::Arc;

/// Helper function to create a sequence from token IDs
pub fn create_sequence(token_ids: &[i32], seq_id: u64) -> Result<Sequence, String> {
    let mut seq = Sequence::new(token_ids, None)?;
    seq.set_seq_id(seq_id);
    Ok(seq)
}

/// Helper function to serialize sequences for transmission
pub fn serialize_sequences_for_transmission(
    sequences: &[Arc<Sequence>],
    buffer: &mut [u8],
    is_prefill: bool,
) -> Result<usize, String> {
    let seq_refs: Vec<&Sequence> = sequences.iter().map(|s| s.as_ref()).collect();
    nanosequence_rs::serialize_sequences(buffer, &seq_refs, is_prefill)
}

/// Helper function to deserialize sequences from received data
pub fn deserialize_sequences_from_transmission(
    buffer: &[u8],
) -> Result<Vec<Sequence>, String> {
    nanosequence_rs::deserialize_sequences(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_creation() {
        let tokens = vec![1, 2, 3, 4, 5];
        let seq = create_sequence(&tokens, 123).unwrap();
        assert_eq!(seq.seq_id(), 123);
        assert_eq!(seq.num_tokens(), 5);
    }

    #[test]
    fn test_serialize_deserialize() {
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![10, 20, 30];

        let seq1 = Arc::new(create_sequence(&tokens1, 1).unwrap());
        let seq2 = Arc::new(create_sequence(&tokens2, 2).unwrap());

        let sequences = vec![seq1.clone(), seq2.clone()];
        let mut buffer = vec![0u8; 1024 * 1024];

        let written = serialize_sequences_for_transmission(&sequences, &mut buffer, false).unwrap();
        assert!(written > 0);

        let deserialized = deserialize_sequences_from_transmission(&buffer[..written]).unwrap();
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized[0].seq_id(), 1);
        assert_eq!(deserialized[1].seq_id(), 2);
    }
}
