use crate::ffi::*;
use crate::sequence::Sequence;
use std::ptr;

/// Serialize a list of sequences to a buffer
///
/// # Arguments
/// * `buffer` - The buffer to write serialized data to
/// * `sequences` - The sequences to serialize
/// * `is_prefill` - Whether this is a prefill phase (affects serialization format)
///
/// # Returns
/// The number of bytes written, or an error message
pub fn serialize_sequences(
    buffer: &mut [u8],
    sequences: &[&Sequence],
    is_prefill: bool,
) -> Result<usize, String> {
    if sequences.is_empty() {
        return Ok(0);
    }

    let data_ptr = buffer.as_mut_ptr() as uintptr_t;
    let buffer_size = buffer.len();

    let seq_ptrs: Vec<NanosequenceSequence> = sequences.iter().map(|s| s.as_ptr()).collect();

    let written = unsafe {
        nanosequence_serialize_sequences(
            data_ptr,
            buffer_size,
            seq_ptrs.as_ptr(),
            seq_ptrs.len(),
            is_prefill,
        )
    };

    if written == 0 && !sequences.is_empty() {
        Err("Serialization failed or buffer too small".to_string())
    } else {
        Ok(written)
    }
}

/// Deserialize sequences from a buffer
///
/// # Arguments
/// * `buffer` - The buffer containing serialized data
///
/// # Returns
/// A vector of deserialized sequences
pub fn deserialize_sequences(buffer: &[u8]) -> Result<Vec<Sequence>, String> {
    if buffer.is_empty() {
        return Ok(Vec::new());
    }

    let data_ptr = buffer.as_ptr() as uintptr_t;
    let data_len = buffer.len();
    let mut count = 0usize;

    let seq_ptrs = unsafe {
        nanosequence_deserialize_sequences(data_ptr, data_len, &mut count as *mut size_t)
    };

    if seq_ptrs.is_null() {
        if count == 0 {
            return Ok(Vec::new());
        } else {
            return Err("Deserialization failed".to_string());
        }
    }

    let mut sequences = Vec::with_capacity(count);

    unsafe {
        for i in 0..count {
            let seq_ptr = *seq_ptrs.add(i);
            if !seq_ptr.is_null() {
                sequences.push(Sequence { inner: seq_ptr });
            }
        }

        // Free the array of pointers (but not the sequences themselves)
        nanosequence_sequences_free(seq_ptrs, count);
    }

    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{BlockContextSlot, SamplingParams, Sequence};

    #[test]
    fn test_serialize_deserialize() {
        // Create a test sequence
        let tokens = vec![1, 2, 3, 4, 5];
        let mut seq1 = Sequence::new(&tokens, None).unwrap();
        seq1.set_seq_id(123);

        let mut seq2 = Sequence::new(&[10, 20, 30], None).unwrap();
        seq2.set_seq_id(456);

        // Serialize
        let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer
        let sequences = vec![&seq1, &seq2];
        let written = serialize_sequences(&mut buffer, &sequences, false).unwrap();
        assert!(written > 0);

        // Deserialize
        let deserialized = deserialize_sequences(&buffer[..written]).unwrap();
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized[0].seq_id(), 123);
        assert_eq!(deserialized[1].seq_id(), 456);
    }
}
