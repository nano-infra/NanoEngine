use pyo3::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};

// 0=ERROR, 1=INFO/WARN, 2=DEBUG — matches legacy C++ nanocommon::set_log_level.
static LOG_LEVEL: AtomicI32 = AtomicI32::new(1);

pub fn get_log_level() -> i32 {
    LOG_LEVEL.load(Ordering::Relaxed)
}

#[pyfunction]
fn set_log_level(level: i32) {
    LOG_LEVEL.store(level.clamp(0, 2), Ordering::Relaxed);
}

#[pyfunction]
fn get_rust_log_level() -> i32 {
    get_log_level()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_rust_log_level, m)?)?;
    Ok(())
}
