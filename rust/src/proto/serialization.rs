use super::metadata::{FreeSequences, FreeVisionSlots, Packet};
use super::wire::{
    bytes_arg, decode_binary, encode_binary, WireFreeSequences, WireFreeVisionSlots, WirePacket,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
#[pyo3(signature = (action, payload))]
pub(super) fn encode_packet(
    py: Python<'_>,
    action: i32,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload = bytes_arg(payload)?;
    let wire = WirePacket { action, payload };
    let bytes = encode_binary(&wire, "packet")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn decode_packet(data: &Bound<'_, PyAny>) -> PyResult<Packet> {
    let data = bytes_arg(data)?;
    let wire: WirePacket = decode_binary(&data, "packet")?;
    Ok(wire.into())
}

#[pyfunction]
#[pyo3(signature = (seq_ids, source_engine_id = ""))]
pub(super) fn encode_free_sequences(
    py: Python<'_>,
    seq_ids: Vec<u64>,
    source_engine_id: &str,
) -> PyResult<PyObject> {
    let wire = WireFreeSequences {
        seq_ids,
        source_engine_id: source_engine_id.to_string(),
    };
    let bytes = encode_binary(&wire, "free sequences")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn decode_free_sequences(data: &Bound<'_, PyAny>) -> PyResult<FreeSequences> {
    let data = bytes_arg(data)?;
    let wire: WireFreeSequences = decode_binary(&data, "free sequences")?;
    Ok(wire.into())
}

#[pyfunction]
#[pyo3(signature = (encoder_engine_id, slot_indices, source_engine_id = ""))]
pub(super) fn encode_free_vision_slots(
    py: Python<'_>,
    encoder_engine_id: &str,
    slot_indices: Vec<i32>,
    source_engine_id: &str,
) -> PyResult<PyObject> {
    let wire = WireFreeVisionSlots {
        encoder_engine_id: encoder_engine_id.to_string(),
        slot_indices,
        source_engine_id: source_engine_id.to_string(),
    };
    let bytes = encode_binary(&wire, "free vision slots")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(super) fn decode_free_vision_slots(data: &Bound<'_, PyAny>) -> PyResult<FreeVisionSlots> {
    let data = bytes_arg(data)?;
    let wire: WireFreeVisionSlots = decode_binary(&data, "free vision slots")?;
    Ok(wire.into())
}
