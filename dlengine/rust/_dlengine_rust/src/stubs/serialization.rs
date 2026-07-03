use super::metadata::MigrateSequenceView;
use super::wire::{
    bytes_arg, decode_binary, encode_binary, encode_wire, run_result_tuple, WireAddRequest,
    WireBatch, WireMigrateSequence, WireMigrationRequest, WireRunResult, WireSamplingParams,
    WireVisionSlot,
};
use crate::sequence::SamplingParams;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
#[pyo3(signature = (seq_id, prompt_token_ids, sampling_params, affinity_key = 0, vision_slots = None))]
pub(super) fn encode_add_request(
    py: Python<'_>,
    seq_id: u64,
    prompt_token_ids: Vec<i32>,
    sampling_params: Py<SamplingParams>,
    affinity_key: u64,
    vision_slots: Option<Vec<(String, i32, i32, i32, i32)>>,
) -> PyResult<PyObject> {
    let sampling_params = sampling_params.borrow(py);
    let request = WireAddRequest {
        seq_id,
        prompt_token_ids,
        sampling_params: WireSamplingParams::from(&*sampling_params),
        affinity_key,
        vision_slots: vision_slots
            .unwrap_or_default()
            .into_iter()
            .map(
                |(encoder_engine_id, slot_idx, num_tokens, hidden_size, max_tokens_per_slot)| {
                    WireVisionSlot {
                        encoder_engine_id,
                        slot_idx,
                        num_tokens,
                        hidden_size,
                        max_tokens_per_slot,
                    }
                },
            )
            .collect(),
    };
    let bytes = encode_binary(&vec![request], "add request")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn decode_migration_metadata(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let data = bytes_arg(data)?;
    let request: WireMigrationRequest = decode_binary(&data, "migration request")?;
    let first_token = request
        .token_ids
        .get(request.num_prompt_tokens.max(0) as usize..)
        .and_then(|tokens| tokens.last())
        .copied()
        .unwrap_or(request.last_token);
    Ok((request.seq_id, first_token)
        .into_pyobject(py)?
        .unbind()
        .into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn parse_migrate_batch(data: &Bound<'_, PyAny>) -> PyResult<Vec<MigrateSequenceView>> {
    let data = bytes_arg(data)?;
    let wire: Vec<WireMigrateSequence> = decode_binary(&data, "migrate batch")?;
    Ok(wire.into_iter().map(Into::into).collect())
}
