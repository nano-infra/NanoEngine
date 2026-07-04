use pyo3::prelude::*;

mod metadata;
mod prepare;
mod serialization;
pub(crate) mod wire;

pub(crate) use metadata::{
    BatchAuxData, DecodeMeta, FreeSequences, FreeVisionSlots, MigrateSequenceView, MigrationIn,
    Packet, PrefillMeta, RequestIn, RequestMigrate, RunnerIn, RunnerOut, StepOut,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RequestIn>()?;
    m.add_class::<RequestMigrate>()?;
    m.add_class::<MigrationIn>()?;
    m.add_class::<RunnerIn>()?;
    m.add_class::<RunnerOut>()?;
    m.add_class::<BatchAuxData>()?;
    m.add_class::<PrefillMeta>()?;
    m.add_class::<DecodeMeta>()?;
    m.add_class::<MigrateSequenceView>()?;
    m.add_class::<Packet>()?;
    m.add_class::<StepOut>()?;
    m.add_class::<FreeSequences>()?;
    m.add_class::<FreeVisionSlots>()?;
    m.add_function(wrap_pyfunction!(serialization::encode_packet, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::decode_packet, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::encode_free_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(serialization::decode_free_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(
        serialization::encode_free_vision_slots,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        serialization::decode_free_vision_slots,
        m
    )?)?;
    Ok(())
}
