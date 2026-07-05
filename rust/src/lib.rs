#![allow(non_upper_case_globals)]

use pyo3::prelude::*;

mod common;
mod config;
mod l3;
mod logging;
mod metrics;
mod proto;
mod router;
mod sampling;
mod scheduler;
mod sequence;
mod snapshots;
mod table;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register(m)?;
    logging::register(m)?;
    sampling::register(m)?;
    l3::register(m)?;
    metrics::register(m)?;
    snapshots::register(m)?;
    scheduler::register(m)?;
    proto::register(m)?;
    router::register(m)?;
    Ok(())
}
