use std::collections::HashMap;

pub(crate) struct ParkedSession {
    pub(crate) affinity_key: u64,
    pub(crate) state_slot: i32,
    pub(crate) hisparse_slot: i32,
    pub(crate) dp_idx: usize,
    pub(crate) group_id: usize,
    pub(crate) length: i32,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) compressed_tables: HashMap<i32, Vec<i32>>,
}
