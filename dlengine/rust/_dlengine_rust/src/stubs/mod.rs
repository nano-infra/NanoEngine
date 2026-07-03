use pyo3::prelude::*;

mod metadata;
mod prepare;
mod serialization;
mod wire;

use metadata::{BatchAuxData, DecodeMeta, MigrateSequenceView, PrefillMeta};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchAuxData>()?;
    m.add_class::<PrefillMeta>()?;
    m.add_class::<DecodeMeta>()?;
    m.add_class::<MigrateSequenceView>()?;
    m.add_function(wrap_pyfunction!(serialization::serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::deserialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::serialize_run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::serialize_migrate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::parse_migrate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prepare::extract_aux_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(
        prepare::extract_vision_slots_from_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        serialization::serialize_dummy_run_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(prepare::prepare_prefill_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(prepare::prepare_decode_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::encode_run_result, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::decode_run_result, m)?)?;
    Ok(())
}
