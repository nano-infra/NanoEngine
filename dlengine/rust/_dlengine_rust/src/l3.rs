use crate::sequence::Sequence;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

const ACTIVE_SLOT: i32 = 0;
const MIGRATE_SLOT: i32 = 1;

#[pyclass(module = "dlengine._dlengine_rust")]
pub struct BlockContextSlot;

#[pymethods]
impl BlockContextSlot {
    #[classattr]
    const ACTIVE: i32 = ACTIVE_SLOT;
    #[classattr]
    const MIGRATE: i32 = MIGRATE_SLOT;
}

#[derive(Clone, Debug)]
struct Block {
    ref_count: i32,
    hash: i64,
    token_ids: Vec<i32>,
}

impl Block {
    fn new() -> Self {
        Self {
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.ref_count = 0;
        self.hash = -1;
        self.token_ids.clear();
    }

    fn update(&mut self, hash: i64, tokens: &[i32]) {
        self.hash = hash;
        self.token_ids.clear();
        self.token_ids.extend_from_slice(tokens);
        self.ref_count = self.ref_count.max(1);
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
pub struct BlockManager {
    #[pyo3(get)]
    engine_id: String,
    #[pyo3(get)]
    group_id: i32,
    block_size: i32,
    blocks: Vec<Block>,
    hash_to_block_id: HashMap<i64, i32>,
    free_block_ids: VecDeque<i32>,
    used_block_ids: HashSet<i32>,
    l3_enabled: bool,
    prefix_caching_enabled: bool,
    l3_resident_hashes: HashSet<i64>,
    pending_loads: Vec<(i64, i32)>,
    pending_offloads: Vec<(i64, i32)>,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (engine_id, group_id, num_blocks, block_size))]
    fn new(engine_id: String, group_id: i32, num_blocks: i32, block_size: i32) -> Self {
        Self {
            engine_id,
            group_id,
            block_size: block_size.max(1),
            blocks: (0..num_blocks.max(0)).map(|_| Block::new()).collect(),
            hash_to_block_id: HashMap::new(),
            free_block_ids: (0..num_blocks.max(0)).collect(),
            used_block_ids: HashSet::new(),
            l3_enabled: false,
            prefix_caching_enabled: true,
            l3_resident_hashes: HashSet::new(),
            pending_loads: Vec::new(),
            pending_offloads: Vec::new(),
        }
    }

    #[getter]
    fn l3_enabled(&self) -> bool {
        self.l3_enabled
    }

    fn set_l3_enabled(&mut self, enabled: bool) {
        self.l3_enabled = enabled;
    }

    #[getter]
    fn prefix_caching_enabled(&self) -> bool {
        self.prefix_caching_enabled
    }

    fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.prefix_caching_enabled = enabled;
    }

    fn set_l3_resident_hashes(&mut self, hashes: Vec<i64>) {
        self.l3_resident_hashes = hashes.into_iter().collect();
    }

    fn mark_l3_resident(&mut self, hashes: Vec<i64>) {
        self.l3_resident_hashes.extend(hashes);
    }

    fn is_l3_resident(&self, hash: i64) -> bool {
        self.l3_resident_hashes.contains(&hash)
    }

    fn compute_block_hashes(&self, seq: PyRef<'_, Sequence>) -> Vec<i64> {
        self.compute_block_hashes_inner(&seq)
    }

    #[pyo3(signature = (seq, _prefix_hint = -1))]
    fn allocate(&mut self, mut seq: PyRefMut<'_, Sequence>, _prefix_hint: i32) -> PyResult<()> {
        let group_id = self.group_id.max(0);
        if seq
            .active_block_tables
            .get(&group_id)
            .map(|table| !table.is_empty())
            .unwrap_or(false)
            || !seq.active_block_table.is_empty()
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Block table is not empty",
            ));
        }

        let mut h = -1i64;
        let mut cache_miss = false;
        let mut num_prefix_cached = 0;
        let num_blocks = self.num_blocks_for_seq(&seq);
        let mut new_table = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let view = self.block_view(&seq, block_idx);
            if self.prefix_caching_enabled && view.len() == self.block_size as usize {
                h = compute_hash(view, h);
            } else {
                h = -1;
            }

            let mut block_id = self.hash_to_block_id.get(&h).copied().unwrap_or(-1);
            let mut l3_hit = false;
            let gpu_hit = block_id >= 0
                && self
                    .blocks
                    .get(block_id as usize)
                    .map(|block| block.token_ids == view)
                    .unwrap_or(false);

            if !cache_miss && !gpu_hit {
                if self.l3_enabled && h != -1 && self.l3_resident_hashes.contains(&h) {
                    l3_hit = true;
                } else {
                    cache_miss = true;
                }
            }

            if cache_miss || l3_hit {
                block_id = self.allocate_fresh_block()?;
                if l3_hit {
                    self.pending_loads.push((h, block_id));
                    num_prefix_cached += 1;
                }
            } else if gpu_hit {
                if !self.used_block_ids.contains(&block_id) {
                    self.mark_block_used(block_id)?;
                }
                let block = &mut self.blocks[block_id as usize];
                block.ref_count += 1;
                if view.len() == self.block_size as usize {
                    num_prefix_cached += 1;
                }
            } else {
                block_id = self.allocate_fresh_block()?;
                cache_miss = true;
            }

            if h != -1 {
                self.blocks[block_id as usize].update(h, view);
                self.hash_to_block_id.insert(h, block_id);
            }
            new_table.push(block_id);
        }
        seq.active_block_table = new_table.clone();
        seq.active_block_tables.insert(group_id, new_table);
        seq.num_cached_tokens =
            (num_prefix_cached * self.block_size).min((seq.num_tokens - 1).max(0));
        Ok(())
    }

    #[pyo3(signature = (seq, _slot = ACTIVE_SLOT))]
    fn deallocate(&mut self, mut seq: PyRefMut<'_, Sequence>, _slot: i32) {
        let group_id = self.group_id.max(0);
        let table = seq
            .active_block_tables
            .remove(&group_id)
            .unwrap_or_else(|| seq.active_block_table.clone());
        for block_id in table.into_iter().rev() {
            self.release_one_block(block_id);
        }
        seq.active_block_table.clear();
        seq.num_cached_tokens = 0;
    }

    fn can_allocate(&self, seq: PyRef<'_, Sequence>) -> i32 {
        let hits = self.count_active_prefix_hits_inner(&seq);
        let needed = self.num_blocks_for_seq(&seq).saturating_sub(hits as usize);
        if self.free_block_ids.len() >= needed {
            hits
        } else {
            -1
        }
    }

    fn count_active_prefix_hits(&self, seq: PyRef<'_, Sequence>) -> i32 {
        self.count_active_prefix_hits_inner(&seq)
    }

    #[pyo3(signature = (seq, scan_cap = 512))]
    fn matched_prefix_blocks(&self, seq: PyRef<'_, Sequence>, scan_cap: i32) -> i32 {
        if !self.prefix_caching_enabled {
            return 0;
        }
        let mut h = -1i64;
        let mut matched = 0;
        let num_blocks = self.num_blocks_for_seq(&seq);
        let limit = if scan_cap > 0 {
            num_blocks.min(scan_cap as usize)
        } else {
            num_blocks
        };
        for block_idx in 0..limit {
            let view = self.block_view(&seq, block_idx);
            if view.len() != self.block_size as usize {
                break;
            }
            h = compute_hash(view, h);
            let Some(block_id) = self.hash_to_block_id.get(&h).copied() else {
                break;
            };
            let Some(block) = self.blocks.get(block_id as usize) else {
                break;
            };
            if block.token_ids != view {
                break;
            }
            matched += 1;
        }
        matched
    }

    fn drain_pending_loads(&mut self) -> Vec<(i64, i32)> {
        let mut out = Vec::new();
        for (hash, block_id) in self.pending_loads.drain(..) {
            if self
                .blocks
                .get(block_id as usize)
                .map(|block| block.hash == hash)
                .unwrap_or(false)
            {
                out.push((hash, block_id));
            }
        }
        out
    }

    fn drain_pending_offloads(&mut self) -> Vec<(i64, i32)> {
        let mut out = Vec::new();
        for (hash, block_id) in self.pending_offloads.drain(..) {
            if self
                .blocks
                .get(block_id as usize)
                .map(|block| block.hash == hash)
                .unwrap_or(false)
            {
                out.push((hash, block_id));
            }
        }
        out
    }

    fn free_block_ids(&self) -> Vec<i32> {
        self.free_block_ids.iter().copied().collect()
    }

    fn num_free_blocks(&self) -> i32 {
        self.free_block_ids.len() as i32
    }
}

impl BlockManager {
    fn allocate_fresh_block(&mut self) -> PyResult<i32> {
        let Some(block_id) = self.free_block_ids.pop_front() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No free blocks available",
            ));
        };
        self.mark_block_used(block_id)?;
        self.blocks[block_id as usize].reset();
        self.blocks[block_id as usize].ref_count = 1;
        Ok(block_id)
    }

    fn mark_block_used(&mut self, block_id: i32) -> PyResult<()> {
        if block_id < 0 || block_id as usize >= self.blocks.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "invalid block_id={block_id}"
            )));
        }
        self.used_block_ids.insert(block_id);
        self.free_block_ids.retain(|id| *id != block_id);
        Ok(())
    }

    fn release_one_block(&mut self, block_id: i32) {
        if block_id < 0 || block_id as usize >= self.blocks.len() {
            return;
        }
        let block = &mut self.blocks[block_id as usize];
        if block.ref_count > 0 {
            block.ref_count -= 1;
        }
        if block.ref_count == 0 && self.used_block_ids.remove(&block_id) {
            if self.l3_enabled && block.hash != -1 {
                self.pending_offloads.push((block.hash, block_id));
            }
            self.free_block_ids.push_back(block_id);
        }
    }

    fn compute_block_hashes_inner(&self, seq: &Sequence) -> Vec<i64> {
        let mut out = Vec::new();
        let mut h = -1i64;
        for block_idx in 0..self.num_blocks_for_seq(seq) {
            let view = self.block_view(seq, block_idx);
            if view.len() != self.block_size as usize {
                break;
            }
            h = compute_hash(view, h);
            out.push(h);
        }
        out
    }

    fn count_active_prefix_hits_inner(&self, seq: &Sequence) -> i32 {
        if !self.prefix_caching_enabled {
            return 0;
        }
        let mut h = -1i64;
        let mut hits = 0;
        for block_idx in 0..self.num_blocks_for_seq(seq) {
            let view = self.block_view(seq, block_idx);
            if view.len() != self.block_size as usize {
                break;
            }
            h = compute_hash(view, h);
            let Some(block_id) = self.hash_to_block_id.get(&h).copied() else {
                break;
            };
            if !self.used_block_ids.contains(&block_id) {
                break;
            }
            let Some(block) = self.blocks.get(block_id as usize) else {
                break;
            };
            if block.token_ids != view {
                break;
            }
            hits += 1;
        }
        hits
    }

    fn num_blocks_for_seq(&self, seq: &Sequence) -> usize {
        ((seq.num_tokens.max(1) + self.block_size - 1) / self.block_size) as usize
    }

    fn block_view<'a>(&self, seq: &'a Sequence, block_idx: usize) -> &'a [i32] {
        let start = block_idx * self.block_size as usize;
        let end = (start + self.block_size as usize).min(seq.token_ids.len());
        if start >= end {
            &[]
        } else {
            &seq.token_ids[start..end]
        }
    }
}

fn compute_hash(tokens: &[i32], prefix: i64) -> i64 {
    let mut hash = 0xcbf29ce484222325u64;
    if prefix != -1 {
        for b in prefix.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    for token in tokens {
        for b in token.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash as i64
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockContextSlot>()?;
    m.add_class::<BlockManager>()?;
    Ok(())
}
