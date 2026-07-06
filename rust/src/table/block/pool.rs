use pyo3::prelude::*;
use std::collections::HashMap;

use super::prefix::compute_block_hash;

#[derive(Clone, Debug)]
pub(crate) struct EvictedBlock {
    pub(crate) block_id: i32,
    pub(crate) hash: i64,
    pub(crate) token_ids: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BlockState {
    Empty,
    Pending,
    Ready,
}

#[derive(Clone, Debug)]
struct BlockMeta {
    ref_count: i32,
    hash: i64,
    token_ids: Vec<i32>,
    state: BlockState,
}

impl BlockMeta {
    fn new() -> Self {
        Self {
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
            state: BlockState::Empty,
        }
    }

    fn reset_pending(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids.clear();
        self.state = BlockState::Pending;
    }

    fn ready_matches(&self, tokens: &[i32], hash: i64) -> bool {
        self.state == BlockState::Ready && self.hash == hash && self.token_ids == tokens
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BlockPool {
    block_size: i32,
    blocks: Vec<BlockMeta>,
    free_blocks: Vec<i32>,
    seq_blocks: HashMap<u64, Vec<i32>>,
    hash_to_block_id: HashMap<i64, i32>,
    ready_lru: Vec<i64>,
    prefix_caching_enabled: bool,
}

impl BlockPool {
    pub(crate) fn new(num_blocks: i32, block_size: i32) -> Self {
        Self {
            block_size: block_size.max(1),
            blocks: (0..num_blocks.max(0)).map(|_| BlockMeta::new()).collect(),
            free_blocks: (0..num_blocks.max(0)).rev().collect(),
            seq_blocks: HashMap::new(),
            hash_to_block_id: HashMap::new(),
            ready_lru: Vec::new(),
            prefix_caching_enabled: true,
        }
    }

    pub(crate) fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.prefix_caching_enabled = enabled;
    }

    pub(crate) fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub(crate) fn num_used_blocks(&self) -> i32 {
        self.blocks.len() as i32 - self.free_blocks.len() as i32
    }

    pub(crate) fn insert_existing(&mut self, seq_id: u64, blocks: Vec<i32>) {
        for block_id in &blocks {
            self.free_blocks.retain(|id| id != block_id);
        }
        self.seq_blocks.insert(seq_id, blocks);
    }

    pub(crate) fn has_seq(&self, seq_id: u64) -> bool {
        self.seq_blocks.contains_key(&seq_id)
    }

    pub(crate) fn insert_seq_blocks(&mut self, seq_id: u64, blocks: Vec<i32>) {
        for block_id in &blocks {
            self.free_blocks.retain(|id| id != block_id);
        }
        self.seq_blocks.insert(seq_id, blocks);
    }

    pub(crate) fn remove_seq(&mut self, seq_id: u64) -> Option<Vec<i32>> {
        let blocks = self.seq_blocks.remove(&seq_id)?;
        for block_id in &blocks {
            self.release_one(*block_id);
        }
        Some(blocks)
    }

    pub(crate) fn take_seq_without_release(&mut self, seq_id: u64) -> Vec<i32> {
        self.seq_blocks.remove(&seq_id).unwrap_or_default()
    }

    pub(crate) fn release_blocks_without_owner(&mut self, blocks: &[i32]) {
        for block_id in blocks {
            self.release_one(*block_id);
        }
    }

    pub(crate) fn matched_prefix_blocks(&self, token_ids: &[i32], tokens: i32) -> i32 {
        if !self.prefix_caching_enabled {
            return 0;
        }
        let mut h = -1i64;
        let mut matched = 0;
        for block_idx in 0..self.num_blocks_for_tokens(tokens) {
            let view = self.block_view(token_ids, block_idx);
            if view.len() != self.block_size as usize {
                break;
            }
            h = compute_block_hash(view, h);
            let Some(block_id) = self.hash_to_block_id.get(&h).copied() else {
                break;
            };
            let Some(block) = self.blocks.get(block_id as usize) else {
                break;
            };
            if !block.ready_matches(view, h) {
                break;
            }
            matched += 1;
        }
        matched
    }

    pub(crate) fn cached_tokens_for(&self, token_ids: &[i32], tokens: i32) -> i32 {
        let matched = self.matched_prefix_blocks(token_ids, tokens);
        (matched * self.block_size).min((tokens - 1).max(0))
    }

    pub(crate) fn ensure_blocks(
        &mut self,
        seq_id: u64,
        token_ids: &[i32],
        tokens: i32,
        use_prefix_cache: bool,
    ) -> PyResult<Vec<i32>> {
        let needed = self.num_blocks_for_tokens(tokens);
        if self.seq_blocks.contains_key(&seq_id) {
            return self.grow_existing(seq_id, needed);
        }

        let plan = self.plan_new_blocks(token_ids, needed, use_prefix_cache)?;
        let fresh_needed = plan.iter().filter(|block_id| block_id.is_none()).count();
        if self.free_blocks.len() < fresh_needed {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "out of KV cache blocks",
            ));
        }

        let mut out = Vec::with_capacity(plan.len());
        for planned in plan {
            if let Some(block_id) = planned {
                self.retain_block(block_id)?;
                out.push(block_id);
            } else {
                out.push(self.allocate_fresh()?);
            }
        }
        self.seq_blocks.insert(seq_id, out.clone());
        Ok(out)
    }

    fn plan_new_blocks(
        &self,
        token_ids: &[i32],
        needed: usize,
        use_prefix_cache: bool,
    ) -> PyResult<Vec<Option<i32>>> {
        let mut plan = Vec::with_capacity(needed);
        let mut h = -1i64;
        let mut cache_miss = !use_prefix_cache || !self.prefix_caching_enabled;
        for block_idx in 0..needed {
            let view = self.block_view(token_ids, block_idx);
            let full = view.len() == self.block_size as usize;
            if full && !cache_miss {
                h = compute_block_hash(view, h);
                if let Some(block_id) = self.lookup_ready_block(h, view) {
                    plan.push(Some(block_id));
                    continue;
                }
            }
            cache_miss = true;
            plan.push(None);
        }
        Ok(plan)
    }

    pub(crate) fn commit_ready(&mut self, seq_id: u64, token_ids: &[i32], committed_tokens: i32) {
        let Some(table) = self.seq_blocks.get(&seq_id).cloned() else {
            return;
        };
        let mut h = -1i64;
        let full_blocks = (committed_tokens.max(0) / self.block_size) as usize;
        for block_idx in 0..full_blocks.min(table.len()) {
            let view = self.block_view(token_ids, block_idx);
            if view.len() != self.block_size as usize {
                break;
            }
            h = compute_block_hash(view, h);
            let block_id = table[block_idx];
            let Some(block) = self.blocks.get_mut(block_id as usize) else {
                continue;
            };
            if block.state == BlockState::Ready && block.hash == h && block.token_ids == view {
                self.hash_to_block_id.insert(h, block_id);
                continue;
            }
            block.hash = h;
            block.token_ids.clear();
            block.token_ids.extend_from_slice(view);
            block.state = BlockState::Ready;
            self.hash_to_block_id.insert(h, block_id);
            self.touch_ready_hash(h);
        }
    }

    pub(crate) fn lookup_ready_prefix_block(&mut self, hash: i64, tokens: &[i32]) -> Option<i32> {
        let block_id = self.lookup_ready_block(hash, tokens)?;
        self.touch_ready_hash(hash);
        Some(block_id)
    }

    pub(crate) fn retain_ready_prefix_block(&mut self, block_id: i32) -> PyResult<()> {
        self.retain_block(block_id)
    }

    pub(crate) fn allocate_pending_fresh(&mut self) -> PyResult<(i32, Option<EvictedBlock>)> {
        self.allocate_fresh_with_eviction()
    }

    pub(crate) fn allocate_promoted_ready(
        &mut self,
        hash: i64,
        tokens: &[i32],
    ) -> PyResult<(i32, Option<EvictedBlock>)> {
        let (block_id, evicted) = self.allocate_fresh_with_eviction()?;
        let block = &mut self.blocks[block_id as usize];
        block.ref_count = 1;
        block.hash = hash;
        block.token_ids.clear();
        block.token_ids.extend_from_slice(tokens);
        block.state = BlockState::Ready;
        self.hash_to_block_id.insert(hash, block_id);
        self.touch_ready_hash(hash);
        Ok((block_id, evicted))
    }

    pub(crate) fn store_ready_cache_block(&mut self, hash: i64, tokens: &[i32]) -> PyResult<i32> {
        if let Some(block_id) = self.lookup_ready_block(hash, tokens) {
            self.touch_ready_hash(hash);
            return Ok(block_id);
        }
        let block_id = match self.free_blocks.pop() {
            Some(block_id) => block_id,
            None => self.evict_ready_cache_block().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("out of host KV cache blocks")
            })?,
        };
        if let Some(block) = self.blocks.get(block_id as usize) {
            if block.hash != -1 && self.hash_to_block_id.get(&block.hash) == Some(&block_id) {
                self.hash_to_block_id.remove(&block.hash);
                self.ready_lru.retain(|h| *h != block.hash);
            }
        }
        let block = &mut self.blocks[block_id as usize];
        block.ref_count = 0;
        block.hash = hash;
        block.token_ids.clear();
        block.token_ids.extend_from_slice(tokens);
        block.state = BlockState::Ready;
        self.hash_to_block_id.insert(hash, block_id);
        self.touch_ready_hash(hash);
        Ok(block_id)
    }

    fn grow_existing(&mut self, seq_id: u64, needed: usize) -> PyResult<Vec<i32>> {
        loop {
            let len = self.seq_blocks.get(&seq_id).map(|b| b.len()).unwrap_or(0);
            if len >= needed {
                break;
            }
            let (block_id, _evicted) = self.allocate_fresh_with_eviction()?;
            self.seq_blocks.entry(seq_id).or_default().push(block_id);
        }
        Ok(self.seq_blocks.get(&seq_id).cloned().unwrap_or_default())
    }

    fn lookup_ready_block(&self, hash: i64, tokens: &[i32]) -> Option<i32> {
        let block_id = self.hash_to_block_id.get(&hash).copied()?;
        let block = self.blocks.get(block_id as usize)?;
        if block.ready_matches(tokens, hash) {
            Some(block_id)
        } else {
            None
        }
    }

    fn retain_block(&mut self, block_id: i32) -> PyResult<()> {
        let Some(block) = self.blocks.get_mut(block_id as usize) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "invalid block_id={block_id}"
            )));
        };
        block.ref_count += 1;
        self.free_blocks.retain(|id| *id != block_id);
        Ok(())
    }

    fn allocate_fresh(&mut self) -> PyResult<i32> {
        Ok(self.allocate_fresh_with_eviction()?.0)
    }

    fn allocate_fresh_with_eviction(&mut self) -> PyResult<(i32, Option<EvictedBlock>)> {
        let Some(block_id) = self.free_blocks.pop() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "out of KV cache blocks",
            ));
        };
        let evicted = self.evicted_ready_meta(block_id);
        if let Some(evicted) = &evicted {
            self.hash_to_block_id.remove(&evicted.hash);
            self.ready_lru.retain(|hash| *hash != evicted.hash);
        }
        let block = &mut self.blocks[block_id as usize];
        block.reset_pending();
        Ok((block_id, evicted))
    }

    fn release_one(&mut self, block_id: i32) {
        let Some(block) = self.blocks.get_mut(block_id as usize) else {
            return;
        };
        if block.ref_count > 0 {
            block.ref_count -= 1;
        }
        if block.ref_count == 0 && !self.free_blocks.contains(&block_id) {
            self.free_blocks.push(block_id);
        }
    }

    fn evicted_ready_meta(&self, block_id: i32) -> Option<EvictedBlock> {
        let block = self.blocks.get(block_id as usize)?;
        if block.state != BlockState::Ready || block.hash == -1 || block.token_ids.is_empty() {
            return None;
        }
        Some(EvictedBlock {
            block_id,
            hash: block.hash,
            token_ids: block.token_ids.clone(),
        })
    }

    fn evict_ready_cache_block(&mut self) -> Option<i32> {
        while let Some(hash) = self.ready_lru.first().copied() {
            self.ready_lru.remove(0);
            let Some(block_id) = self.hash_to_block_id.remove(&hash) else {
                continue;
            };
            let Some(block) = self.blocks.get_mut(block_id as usize) else {
                continue;
            };
            if block.ref_count == 0 && block.state == BlockState::Ready {
                block.hash = -1;
                block.token_ids.clear();
                block.state = BlockState::Empty;
                return Some(block_id);
            }
            self.hash_to_block_id.insert(hash, block_id);
        }
        None
    }

    fn touch_ready_hash(&mut self, hash: i64) {
        if hash == -1 {
            return;
        }
        self.ready_lru.retain(|h| *h != hash);
        self.ready_lru.push(hash);
    }

    fn num_blocks_for_tokens(&self, tokens: i32) -> usize {
        ((tokens.max(1) + self.block_size - 1) / self.block_size) as usize
    }

    fn block_view<'a>(&self, token_ids: &'a [i32], block_idx: usize) -> &'a [i32] {
        let start = block_idx * self.block_size as usize;
        let end = (start + self.block_size as usize).min(token_ids.len());
        if start >= end {
            &[]
        } else {
            &token_ids[start..end]
        }
    }
}

pub(crate) struct CompressedPool {
    pub(crate) ratio: i32,
    pub(crate) page_size: i32,
    pub(crate) max_blocks_per_seq: i32,
    pub(crate) free_pages: Vec<i32>,
    pub(crate) seq_pages: HashMap<u64, Vec<i32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: i32 = 4;

    #[test]
    fn release_then_reuse_ready_prefix() {
        let mut pool = BlockPool::new(8, BS);
        let tokens: Vec<i32> = (0..8).collect();
        let first = pool.ensure_blocks(1, &tokens, 8, true).unwrap();
        pool.commit_ready(1, &tokens, 8);
        pool.remove_seq(1);

        assert_eq!(pool.cached_tokens_for(&tokens, 8), 7);
        let second = pool.ensure_blocks(2, &tokens, 8, true).unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn pending_blocks_are_not_prefix_hits_until_commit() {
        let mut pool = BlockPool::new(8, BS);
        let tokens: Vec<i32> = (0..8).collect();
        pool.ensure_blocks(1, &tokens, 8, true).unwrap();
        assert_eq!(pool.cached_tokens_for(&tokens, 8), 0);

        pool.commit_ready(1, &tokens, 8);
        assert_eq!(pool.cached_tokens_for(&tokens, 8), 7);
    }

    #[test]
    fn partial_block_is_not_indexed() {
        let mut pool = BlockPool::new(8, BS);
        let tokens: Vec<i32> = (0..6).collect();
        pool.ensure_blocks(1, &tokens, 6, true).unwrap();
        pool.commit_ready(1, &tokens, 6);
        assert_eq!(pool.matched_prefix_blocks(&tokens, 6), 1);
        assert_eq!(pool.cached_tokens_for(&tokens, 6), 4);
    }

    #[test]
    fn token_mismatch_stops_prefix_match() {
        let mut pool = BlockPool::new(8, BS);
        let tokens: Vec<i32> = (0..8).collect();
        let table = pool.ensure_blocks(1, &tokens, 8, true).unwrap();
        pool.commit_ready(1, &tokens, 8);
        let first_hash = compute_block_hash(&tokens[..4], -1);
        let second_hash = compute_block_hash(&tokens[4..8], first_hash);
        pool.hash_to_block_id.insert(second_hash, table[1]);
        pool.blocks[table[1] as usize].token_ids = vec![9, 9, 9, 9];
        assert_eq!(pool.matched_prefix_blocks(&tokens, 8), 1);
    }

    #[test]
    fn disabled_prefix_cache_reports_no_hits() {
        let mut pool = BlockPool::new(8, BS);
        let tokens: Vec<i32> = (0..8).collect();
        pool.ensure_blocks(1, &tokens, 8, true).unwrap();
        pool.commit_ready(1, &tokens, 8);
        pool.set_prefix_caching_enabled(false);
        assert_eq!(pool.cached_tokens_for(&tokens, 8), 0);
    }
}
