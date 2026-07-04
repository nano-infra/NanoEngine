mod manager;
mod runner;
mod sequence;
mod server;
mod stats;

use pyo3::prelude::*;

pub(crate) use manager::RuntimeMetrics;
pub(crate) use sequence::SequenceMetric;
pub(crate) use server::ServerMetric;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SequenceMetric>()?;
    m.add_class::<ServerMetric>()?;
    m.add_class::<RuntimeMetrics>()?;
    Ok(())
}
