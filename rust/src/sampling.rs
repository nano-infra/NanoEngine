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
    /// Canonical JSON Schema used for constrained decoding.  ``None`` keeps
    /// the existing unconstrained sampling path.
    #[pyo3(get, set)]
    pub json_schema: Option<String>,
    /// Serialized XGrammar StructuralTag for model-native tool calling.
    #[pyo3(get, set)]
    pub structural_tag: Option<String>,
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature = 1.0, max_tokens = 256, ignore_eos = false, return_completion_logprobs = false, json_schema = None, structural_tag = None))]
    pub(crate) fn new(
        temperature: f64,
        max_tokens: i32,
        ignore_eos: bool,
        return_completion_logprobs: bool,
        json_schema: Option<String>,
        structural_tag: Option<String>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            return_completion_logprobs,
            json_schema,
            structural_tag,
        }
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<SamplingParams>();
        let args = (
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
            self.return_completion_logprobs,
            self.json_schema.clone(),
            self.structural_tag.clone(),
        )
            .into_pyobject(py)?;
        Ok((cls, args).into_pyobject(py)?.unbind().into())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SamplingParams>()?;
    Ok(())
}
