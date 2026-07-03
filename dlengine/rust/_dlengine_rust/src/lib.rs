#![allow(non_upper_case_globals)]

use pyo3::prelude::*;

mod cache_plan;
mod common;
mod l3;
mod metrics;
mod scheduler;
mod sequence;
mod snapshots;
mod stubs;

#[pymodule]
fn _dlengine_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cache_plan::register(m)?;
    sequence::register(m)?;
    l3::register(m)?;
    metrics::register(m)?;
    snapshots::register(m)?;
    scheduler::register(m)?;
    stubs::register(m)?;
    Ok(())
}
