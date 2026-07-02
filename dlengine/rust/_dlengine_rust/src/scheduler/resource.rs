use std::collections::HashMap;

#[derive(Default)]
pub(super) struct GroupResource {
    pub(super) free_blocks: Vec<i32>,
    pub(super) seq_blocks: HashMap<u64, Vec<i32>>,
}

pub(super) struct CompressedPool {
    pub(super) ratio: i32,
    pub(super) page_size: i32,
    pub(super) max_blocks_per_seq: i32,
    pub(super) free_pages: Vec<i32>,
    pub(super) seq_pages: HashMap<u64, Vec<i32>>,
}
