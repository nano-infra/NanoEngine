use super::metadata::MigrateSequenceView;
use super::wire::{
    bytes_arg, decode_binary, encode_binary, encode_wire, run_result_tuple,
    sequence_to_migrate_wire, sequence_to_wire, wire_to_sequence, WireBatch, WireMigrateSequence,
    WireRunResult, WireSequence,
};
use crate::sequence::Sequence;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
#[pyo3(signature = (seqs, is_prefill))]
pub(super) fn serialize_run_batch(
    py: Python<'_>,
    seqs: &Bound<'_, PyAny>,
    is_prefill: bool,
) -> PyResult<PyObject> {
    let seq: Vec<Py<Sequence>> = seqs.extract()?;
    if seq.is_empty() {
        return encode_wire(py, &WireBatch::empty(is_prefill));
    }

    let mut input_ids = Vec::new();
    let mut positions = Vec::new();
    let mut seq_lens = Vec::new();
    let mut block_tables = Vec::new();
    let mut temperatures = Vec::new();
    let mut state_slots = Vec::new();
    let mut compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>> =
        std::collections::HashMap::new();
    let mut hisparse_slots = Vec::new();

    for item in seq {
        let item = item.borrow(py);
        let tokens: Vec<i64> = item.token_ids.iter().copied().map(i64::from).collect();
        let prompt_len = (item.num_prompt_tokens.max(0) as usize).min(tokens.len());
        let block_table = item.active_block_table.clone();
        let compressed_tables = item.active_compressed_block_tables.clone();
        let temperature = item.sampling_params.temperature as f32;

        if is_prefill {
            let start = (item.num_cached_tokens.max(0) as usize).min(tokens.len());
            let chunk_end = item.num_tokens.max(0) as usize;
            let end = chunk_end.min(prompt_len).min(tokens.len()).max(start);
            let slice = &tokens[start..end];
            seq_lens.push(slice.len() as i32);
            for (offset, token) in slice.iter().enumerate() {
                input_ids.push(*token);
                positions.push((start + offset) as i64);
            }
        } else {
            let token = i64::from(item.last_token);
            input_ids.push(token);
            positions.push(tokens.len().saturating_sub(1) as i64);
            seq_lens.push(1);
        }
        block_tables.push(block_table);
        temperatures.push(temperature);
        state_slots.push(i64::from(item.active_state_slot));
        hisparse_slots.push(i64::from(item.active_hisparse_slot));
        for (ratio, blocks) in compressed_tables {
            let rows = compressed_block_tables.entry(ratio).or_default();
            while rows.len() + 1 < seq_lens.len() {
                rows.push(Vec::new());
            }
            rows.push(blocks);
        }
        for rows in compressed_block_tables.values_mut() {
            while rows.len() < seq_lens.len() {
                rows.push(Vec::new());
            }
        }
    }

    encode_wire(
        py,
        &WireBatch {
            is_prefill,
            input_ids,
            positions,
            seq_lens,
            block_tables,
            temperatures,
            state_slots,
            compressed_block_tables,
            hisparse_slots,
            is_dummy: false,
        },
    )
}

#[pyfunction]
#[pyo3(signature = (_engine_id, _num_kvcache_blocks, is_prefill))]
pub(super) fn serialize_dummy_run_batch(
    py: Python<'_>,
    _engine_id: &str,
    _num_kvcache_blocks: i64,
    is_prefill: bool,
) -> PyResult<PyObject> {
    encode_wire(py, &WireBatch::dummy(is_prefill))
}

#[pyfunction]
#[pyo3(signature = (token_ids, logprobs = None, server_handler_ns = 0))]
pub(super) fn encode_run_result(
    py: Python<'_>,
    token_ids: &Bound<'_, PyAny>,
    logprobs: Option<&Bound<'_, PyAny>>,
    server_handler_ns: u64,
) -> PyResult<PyObject> {
    let payload = WireRunResult {
        server_handler_ns,
        token_ids: token_ids.extract::<Vec<Vec<i64>>>()?,
        logprobs: match logprobs {
            Some(value) if !value.is_none() => Some(value.extract::<Vec<Vec<f32>>>()?),
            _ => None,
        },
    };
    let bytes = encode_binary(&payload, "run result")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
pub(super) fn decode_run_result(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let data = bytes_arg(data)?;
    let result: WireRunResult = decode_binary(&data, "run result")?;
    run_result_tuple(py, result)
}

#[pyfunction]
#[pyo3(signature = (data_ptr, buffer_size, seqs, _is_prefill))]
pub(super) fn serialize(
    py: Python<'_>,
    data_ptr: usize,
    buffer_size: usize,
    seqs: &Bound<'_, PyAny>,
    _is_prefill: bool,
) -> PyResult<usize> {
    if data_ptr == 0 {
        return Err(PyValueError::new_err("serialize data_ptr is null"));
    }
    let seqs: Vec<Py<Sequence>> = seqs.extract()?;
    let mut wire = Vec::with_capacity(seqs.len());
    for seq in seqs {
        wire.push(sequence_to_wire(py, &seq));
    }
    let bytes = encode_binary(&wire, "sequences")?;
    if bytes.len() > buffer_size {
        return Err(PyValueError::new_err(format!(
            "serialize buffer too small: need {} bytes, got {}",
            bytes.len(),
            buffer_size
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr as *mut u8, bytes.len());
    }
    Ok(bytes.len())
}

#[pyfunction]
#[pyo3(signature = (data_ptr, data_len))]
pub(super) fn deserialize(
    py: Python<'_>,
    data_ptr: usize,
    data_len: usize,
) -> PyResult<Vec<Py<Sequence>>> {
    if data_ptr == 0 {
        return Err(PyValueError::new_err("deserialize data_ptr is null"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) };
    let wire: Vec<WireSequence> = decode_binary(bytes, "sequences")?;
    wire.into_iter()
        .map(|seq| wire_to_sequence(py, seq))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (seqs))]
pub(super) fn serialize_migrate_batch(
    py: Python<'_>,
    seqs: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let seqs: Vec<Py<Sequence>> = seqs.extract()?;
    let mut wire = Vec::with_capacity(seqs.len());
    for seq in seqs {
        wire.push(sequence_to_migrate_wire(py, &seq));
    }
    let bytes = encode_binary(&wire, "migrate batch")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn parse_migrate_batch(data: &Bound<'_, PyAny>) -> PyResult<Vec<MigrateSequenceView>> {
    let data = bytes_arg(data)?;
    let wire: Vec<WireMigrateSequence> = decode_binary(&data, "migrate batch")?;
    Ok(wire.into_iter().map(Into::into).collect())
}
