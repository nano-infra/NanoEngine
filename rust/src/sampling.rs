use pyo3::prelude::*;

#[pyclass(module = "dlengine._engine")]
#[derive(Clone)]
pub struct SamplingParams {
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub max_tokens: i32,
    #[pyo3(get, set)]
    pub ignore_eos: bool,
    #[pyo3(get, set)]
    pub return_completion_logprobs: bool,
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature = 1.0, max_tokens = 256, ignore_eos = false, return_completion_logprobs = false))]
    pub(crate) fn new(
        temperature: f64,
        max_tokens: i32,
        ignore_eos: bool,
        return_completion_logprobs: bool,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            return_completion_logprobs,
        }
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<SamplingParams>();
        let args = (
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
            self.return_completion_logprobs,
        )
            .into_pyobject(py)?;
        Ok((cls, args).into_pyobject(py)?.unbind().into())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SamplingParams>()?;
    Ok(())
}
