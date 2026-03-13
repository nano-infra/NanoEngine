//! Utilities for working with sequences using flatbuffers directly

use crate::fbs::{self, SequenceStatus};
use flatbuffers::{root, FlatBufferBuilder};

/// Represents a sequence object built from flatbuffers
#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id: u64,
    pub status: SequenceStatus,
    pub last_token: i32,
    pub num_tokens: i32,
    pub num_prompt_tokens: i32,
    pub num_checkpointed_tokens: i32,
    pub num_cached_tokens: i32,
    pub token_ids: Vec<i32>,
    pub temperature: f64,
    pub max_tokens: i32,
    pub ignore_eos: bool,
}

impl Sequence {
    /// Create a new sequence from token IDs with default sampling params
    pub fn new(token_ids: &[i32], seq_id: u64) -> Self {
        Self {
            seq_id,
            status: SequenceStatus::WAITING,
            last_token: *token_ids.last().unwrap_or(&0),
            num_tokens: token_ids.len() as i32,
            num_prompt_tokens: token_ids.len() as i32,
            num_checkpointed_tokens: 0,
            num_cached_tokens: 0,
            token_ids: token_ids.to_vec(),
            temperature: 1.0,
            max_tokens: 1024,
            ignore_eos: false,
        }
    }

    /// Set sequence ID
    pub fn set_seq_id(&mut self, seq_id: u64) {
        self.seq_id = seq_id;
    }

    /// Get sequence ID
    pub fn seq_id(&self) -> u64 {
        self.seq_id
    }

    /// Get number of tokens
    pub fn num_tokens(&self) -> usize {
        self.num_tokens as usize
    }

    /// Convert this sequence into a flatbuffers representation
    fn to_flatbuffer<'a>(
        &self,
        builder: &mut FlatBufferBuilder<'a>,
    ) -> flatbuffers::WIPOffset<fbs::Sequence<'a>> {
        // Create sampling params
        let sampling_params = fbs::SamplingParams::create(
            builder,
            &fbs::SamplingParamsArgs {
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                ignore_eos: self.ignore_eos,
            },
        );

        // Create token_ids vector
        let token_ids_vec = builder.create_vector(&self.token_ids);

        // Create sequence
        fbs::Sequence::create(
            builder,
            &fbs::SequenceArgs {
                seq_id: self.seq_id,
                status: self.status,
                sampling_params: Some(sampling_params),
                last_token: self.last_token,
                num_tokens: self.num_tokens,
                num_prompt_tokens: self.num_prompt_tokens,
                num_checkpointed_tokens: self.num_checkpointed_tokens,
                num_cached_tokens: self.num_cached_tokens,
                token_ids: Some(token_ids_vec),
                slots: None,        // Not used in NanoRoute
                vision_slots: None, // Not used in NanoRoute
            },
        )
    }

    /// Create from flatbuffers representation
    fn from_flatbuffer(seq: fbs::Sequence) -> Result<Self, String> {
        let sampling_params = seq.sampling_params().ok_or("Missing sampling params")?;

        let token_ids = seq.token_ids().ok_or("Missing token_ids")?.iter().collect();

        Ok(Self {
            seq_id: seq.seq_id(),
            status: seq.status(),
            last_token: seq.last_token(),
            num_tokens: seq.num_tokens(),
            num_prompt_tokens: seq.num_prompt_tokens(),
            num_checkpointed_tokens: seq.num_checkpointed_tokens(),
            num_cached_tokens: seq.num_cached_tokens(),
            token_ids,
            temperature: sampling_params.temperature(),
            max_tokens: sampling_params.max_tokens(),
            ignore_eos: sampling_params.ignore_eos(),
        })
    }
}

/// Helper function to create a sequence from token IDs
pub fn create_sequence(token_ids: &[i32], seq_id: u64) -> Result<Sequence, String> {
    Ok(Sequence::new(token_ids, seq_id))
}

/// Helper function to serialize sequences for transmission
pub fn serialize_sequences_for_transmission(
    sequences: &[std::sync::Arc<Sequence>],
    buffer: &mut [u8],
    _is_prefill: bool,
) -> Result<usize, String> {
    let mut builder = FlatBufferBuilder::new();

    // Build sequence list
    let seq_offsets: Vec<_> = sequences
        .iter()
        .map(|seq| seq.to_flatbuffer(&mut builder))
        .collect();

    let sequences_vec = builder.create_vector(&seq_offsets);
    let seq_list = fbs::SequenceList::create(
        &mut builder,
        &fbs::SequenceListArgs {
            sequences: Some(sequences_vec),
        },
    );

    builder.finish(seq_list, None);
    let data = builder.finished_data();

    if data.len() > buffer.len() {
        return Err(format!(
            "Buffer too small: need {} bytes, have {}",
            data.len(),
            buffer.len()
        ));
    }

    buffer[..data.len()].copy_from_slice(data);
    Ok(data.len())
}

/// Helper function to deserialize sequences from received data
pub fn deserialize_sequences_from_transmission(buffer: &[u8]) -> Result<Vec<Sequence>, String> {
    let seq_list = root::<fbs::SequenceList>(buffer)
        .map_err(|e| format!("Failed to parse flatbuffer: {}", e))?;

    let sequences = seq_list
        .sequences()
        .ok_or("Missing sequences in SequenceList")?;

    sequences
        .iter()
        .map(|seq| Sequence::from_flatbuffer(seq))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        assert_eq!(deserialized[0].num_tokens(), 3);
        assert_eq!(deserialized[1].num_tokens(), 3);
    }
}
