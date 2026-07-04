#![allow(non_upper_case_globals)]

use pyo3::prelude::*;

mod common;
mod config;
mod l3;
mod logging;
mod metrics;
mod proto;
mod router;
mod scheduler;
mod sequence;
mod snapshots;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register(m)?;
    logging::register(m)?;
    sequence::register(m)?;
    l3::register(m)?;
    metrics::register(m)?;
    snapshots::register(m)?;
    scheduler::register(m)?;
    proto::register(m)?;
    router::register(m)?;
    Ok(())
}
