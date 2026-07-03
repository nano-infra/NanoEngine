use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

macro_rules! spec {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        #[pyclass(module = "dlengine._dlengine_rust")]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $name {
            $(#[pyo3(get, set)] pub $field: i32,)+
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self { $($field: -1,)+ }
            }
        }
    };
}

spec!(GqaCacheSpec {
    num_pages,
    page_size,
    max_blocks_per_seq,
    num_layers,
    num_kv_heads,
    head_dim,
});

spec!(MlaCacheSpec {
    num_pages,
    page_size,
    max_blocks_per_seq,
    num_layers,
    kv_lora_rank,
    qk_rope_head_dim,
    head_dim,
});

spec!(GdnCacheSpec {
    num_pages,
    page_size,
    max_blocks_per_seq,
    state_slots,
    state_bytes,
});

spec!(HcaCacheSpec {
    bytes_per_token,
    block_size_multiple,
    compression_ratio,
    num_pages,
    page_size,
    max_blocks_per_seq,
});

spec!(CsaCacheSpec {
    compressor_head_dim,
    compression_ratio,
    num_pages,
    page_size,
    max_blocks_per_seq,
});

spec!(IndexerCacheSpec {
    num_pages,
    page_size,
    max_blocks_per_seq,
    index_head_dim,
    bytes_per_token,
});

spec!(HiSparseCacheSpec {
    max_num_seqs,
    device_buffer_size,
    host_to_device_ratio,
    swap_in_block_size,
    dummy_slot,
});
#[pyclass(module = "dlengine._dlengine_rust")]
pub struct CachePlanFlag;

#[pymethods]
impl CachePlanFlag {
    #[classattr]
    const Gqa: u32 = 1 << 0;
    #[classattr]
    const Mla: u32 = 1 << 1;
    #[classattr]
    const Gdn: u32 = 1 << 2;
    #[classattr]
    const Hca: u32 = 1 << 3;
    #[classattr]
    const Csa: u32 = 1 << 4;
    #[classattr]
    const Indexer: u32 = 1 << 5;
    #[classattr]
    const Hisparse: u32 = 1 << 6;
}

#[pyfunction]
fn cache_plan_flag(flag: u32) -> u32 {
    flag
}

fn flag_name(flag: u32) -> &'static str {
    match flag {
        CachePlanFlag::Gqa => "gqa",
        CachePlanFlag::Mla => "mla",
        CachePlanFlag::Gdn => "gdn",
        CachePlanFlag::Hca => "hca",
        CachePlanFlag::Csa => "csa",
        CachePlanFlag::Indexer => "indexer",
        CachePlanFlag::Hisparse => "hisparse",
        _ => "unknown",
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CachePlan {
    #[pyo3(get, set)]
    pub flags: u32,
    #[pyo3(get, set)]
    pub gqa: GqaCacheSpec,
    #[pyo3(get, set)]
    pub mla: MlaCacheSpec,
    #[pyo3(get, set)]
    pub gdn: GdnCacheSpec,
    #[pyo3(get, set)]
    pub hca: HcaCacheSpec,
    #[pyo3(get, set)]
    pub csa: CsaCacheSpec,
    #[pyo3(get, set)]
    pub indexer: IndexerCacheSpec,
    #[pyo3(get, set)]
    pub hisparse: HiSparseCacheSpec,
}

#[pymethods]
impl CachePlan {
    #[new]
    #[pyo3(signature = (flags = 0))]
    pub fn new(flags: u32) -> Self {
        Self {
            flags,
            gqa: GqaCacheSpec::new(),
            mla: MlaCacheSpec::new(),
            gdn: GdnCacheSpec::new(),
            hca: HcaCacheSpec::new(),
            csa: CsaCacheSpec::new(),
            indexer: IndexerCacheSpec::new(),
            hisparse: HiSparseCacheSpec::new(),
        }
    }

    fn has_flag(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    fn set_flag(&mut self, flag: u32) {
        self.flags |= flag;
    }

    fn has_gqa(&self) -> bool {
        self.has_flag(CachePlanFlag::Gqa)
    }
    fn has_mla(&self) -> bool {
        self.has_flag(CachePlanFlag::Mla)
    }
    fn has_gdn(&self) -> bool {
        self.has_flag(CachePlanFlag::Gdn)
    }
    fn has_hca(&self) -> bool {
        self.has_flag(CachePlanFlag::Hca)
    }
    fn has_csa(&self) -> bool {
        self.has_flag(CachePlanFlag::Csa)
    }
    fn has_indexer(&self) -> bool {
        self.has_flag(CachePlanFlag::Indexer)
    }
    fn has_hisparse(&self) -> bool {
        self.has_flag(CachePlanFlag::Hisparse)
    }
    fn has_linear_attention(&self) -> bool {
        self.has_gdn()
    }

    fn cache_mode(&self) -> &'static str {
        if self.has_hca() || self.has_csa() {
            "dsv4"
        } else if self.has_mla() || self.has_indexer() || self.has_hisparse() {
            "mla"
        } else {
            "gqa"
        }
    }

    #[pyo3(signature = (indent = None))]
    fn to_json_string(&self, indent: Option<usize>) -> PyResult<String> {
        let mut root = serde_json::Map::new();
        root.insert("mode".to_string(), serde_json::json!(self.cache_mode()));
        root.insert("flags".to_string(), serde_json::json!(self.flags));

        let mut enabled = Vec::new();
        macro_rules! push_spec {
            ($flag:expr, $field:ident) => {
                if self.has_flag($flag) {
                    enabled.push(flag_name($flag));
                    root.insert(
                        stringify!($field).to_string(),
                        serde_json::to_value(&self.$field)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
                    );
                }
            };
        }
        push_spec!(CachePlanFlag::Gqa, gqa);
        push_spec!(CachePlanFlag::Mla, mla);
        push_spec!(CachePlanFlag::Gdn, gdn);
        push_spec!(CachePlanFlag::Hca, hca);
        push_spec!(CachePlanFlag::Csa, csa);
        push_spec!(CachePlanFlag::Indexer, indexer);
        push_spec!(CachePlanFlag::Hisparse, hisparse);
        root.insert("enabled".to_string(), serde_json::json!(enabled));

        let value = serde_json::Value::Object(root);
        match indent {
            Some(_) => serde_json::to_string_pretty(&value),
            None => serde_json::to_string(&value),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (indent = None))]
    fn to_json(&self, indent: Option<usize>) -> PyResult<String> {
        self.to_json_string(indent)
    }

    fn __getstate__(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __getnewargs__(&self) -> (u32,) {
        (self.flags,)
    }

    fn __setstate__(&mut self, state: &str) -> PyResult<()> {
        *self = serde_json::from_str(state)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CachePlanFlag>()?;
    m.add_function(wrap_pyfunction!(cache_plan_flag, m)?)?;
    m.add_class::<GqaCacheSpec>()?;
    m.add_class::<MlaCacheSpec>()?;
    m.add_class::<GdnCacheSpec>()?;
    m.add_class::<HcaCacheSpec>()?;
    m.add_class::<CsaCacheSpec>()?;
    m.add_class::<IndexerCacheSpec>()?;
    m.add_class::<HiSparseCacheSpec>()?;
    m.add_class::<CachePlan>()?;
    Ok(())
}
