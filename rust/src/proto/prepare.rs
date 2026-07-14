use super::metadata::{BatchAuxData, DecodeMeta, PrefillMeta};
use super::wire::decode_wire;
use pyo3::prelude::*;

pub(crate) fn runner_in_aux(data: &[u8], _sp_rank: usize) -> PyResult<BatchAuxData> {
    let batch = decode_wire(data)?;
    Ok(BatchAuxData {
        num_group_seqs: batch.num_group_seqs(),
        temperatures: batch.temperatures,
        state_slots: batch.state_slots,
        compressed_block_tables: batch.compressed_block_tables,
        hisparse_slots: batch.hisparse_slots,
        seq_ids: batch.seq_ids,
        any_return_completion_logprobs: false,
    })
}

pub(crate) fn runner_in_prefill(
    data: &[u8],
    _sp_rank: usize,
    _sp_size: usize,
    _block_size: usize,
    _max_num_seqs: usize,
    _num_kvcache_blocks: usize,
) -> PyResult<PrefillMeta> {
    let batch = decode_wire(data)?;
    let mut cu_q = Vec::with_capacity(batch.seq_lens.len() + 1);
    let mut cu_k = Vec::with_capacity(batch.seq_lens.len() + 1);
    cu_q.push(0);
    cu_k.push(0);
    let mut total_q = 0i32;
    let mut total_k = 0i32;
    let mut max_len_q = 0usize;
    let mut max_len_k = 0usize;
    let max_blocks = batch
        .block_tables
        .iter()
        .map(|blocks| blocks.len())
        .max()
        .unwrap_or(0);
    let mut block_tables_flat = Vec::new();
    if !batch.is_dummy && max_blocks > 0 {
        block_tables_flat.resize(_sp_size * _max_num_seqs.max(1) * max_blocks, 0);
        for sp in 0.._sp_size {
            for (seq_idx, blocks) in batch.block_tables.iter().enumerate() {
                if seq_idx >= _max_num_seqs {
                    break;
                }
                let base = (sp * _max_num_seqs.max(1) + seq_idx) * max_blocks;
                for (block_idx, block) in blocks.iter().copied().enumerate() {
                    block_tables_flat[base + block_idx] = block;
                }
            }
        }
    }
    let mut sampling_token_indices = Vec::new();
    let mut sampling_seq_indices = Vec::new();
    for (idx, len) in batch.seq_lens.iter().copied().enumerate() {
        let q_len = len.max(0);
        let q_len_usize = q_len as usize;
        total_q += q_len;
        let pos_end = if q_len > 0 {
            batch
                .positions
                .get(total_q as usize - 1)
                .copied()
                .unwrap_or(i64::from(total_q - 1))
                .saturating_add(1)
                .max(0) as i32
        } else {
            0
        };
        total_k += pos_end;
        cu_q.push(total_q);
        cu_k.push(total_k);
        max_len_q = max_len_q.max(q_len_usize);
        max_len_k = max_len_k.max(pos_end.max(0) as usize);
        if len > 0 && batch.sample_mask.get(idx).copied().unwrap_or(true) {
            sampling_token_indices.push((total_q - 1) as i64);
            sampling_seq_indices.push(idx as i64);
        }
    }
    let mut token_seq_indices = Vec::with_capacity(total_q.max(0) as usize);
    for (seq_idx, len) in batch.seq_lens.iter().copied().enumerate() {
        for _ in 0..len.max(0) {
            token_seq_indices.push(seq_idx);
        }
    }
    Ok(PrefillMeta {
        input_ids: batch.input_ids,
        positions: batch.positions.clone(),
        cu_seqlens_q: cu_q,
        cu_seqlens_k: cu_k,
        slot_mapping: if batch.is_dummy {
            vec![-1; total_q.max(0) as usize]
        } else {
            batch
                .positions
                .iter()
                .enumerate()
                .map(|(idx, pos)| {
                    let seq_idx = token_seq_indices.get(idx).copied().unwrap_or(0);
                    let block_table = batch.block_tables.get(seq_idx).cloned().unwrap_or_default();
                    let block_size = _block_size.max(1) as i64;
                    let logical_block = (*pos / block_size).max(0) as usize;
                    let offset = (*pos % block_size) as i32;
                    block_table
                        .get(logical_block)
                        .copied()
                        .map(|block| block * _block_size.max(1) as i32 + offset)
                        .unwrap_or(idx as i32)
                })
                .collect()
        },
        use_block_tables: !block_tables_flat.is_empty(),
        block_tables_flat,
        max_num_blocks: max_blocks,
        max_seqlen_q: max_len_q,
        max_seqlen_k: max_len_k,
        sampling_token_indices,
        sampling_seq_indices,
    })
}

pub(crate) fn runner_in_decode(
    data: &[u8],
    _sp_rank: usize,
    sp_size: usize,
    _block_size: usize,
    max_num_seqs: usize,
    _num_kvcache_blocks: usize,
) -> PyResult<DecodeMeta> {
    let batch = decode_wire(data)?;
    let num = batch.num_group_seqs();
    let max_num_blocks = batch
        .block_tables
        .iter()
        .map(|blocks| blocks.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let mut block_tables_flat = vec![0; sp_size * num.max(1) * max_num_blocks];
    if !batch.is_dummy {
        for sp in 0..sp_size {
            for seq_idx in 0..num.max(1) {
                let base = (sp * num.max(1) + seq_idx) * max_num_blocks;
                if let Some(blocks) = batch.block_tables.get(seq_idx) {
                    for (block_idx, block) in blocks.iter().copied().enumerate() {
                        block_tables_flat[base + block_idx] = block;
                    }
                }
            }
        }
    }
    let mut context_lens_flat = vec![0; sp_size * max_num_seqs];
    for i in 0..num.min(max_num_seqs) {
        context_lens_flat[i] = batch.positions.get(i).copied().unwrap_or(0) as i32 + 1;
    }
    Ok(DecodeMeta {
        input_ids: batch.input_ids,
        positions: batch.positions.clone(),
        slot_mapping: if batch.is_dummy {
            vec![-1; num]
        } else {
            batch
                .positions
                .iter()
                .enumerate()
                .map(|(idx, pos)| {
                    let block_table = batch.block_tables.get(idx).cloned().unwrap_or_default();
                    let block_size = _block_size.max(1) as i64;
                    let logical_block = (*pos / block_size).max(0) as usize;
                    let offset = (*pos % block_size) as i32;
                    block_table
                        .get(logical_block)
                        .copied()
                        .map(|block| block * _block_size.max(1) as i32 + offset)
                        .unwrap_or(idx as i32)
                })
                .collect()
        },
        context_lens_flat,
        block_tables_flat: if batch.is_dummy {
            Vec::new()
        } else {
            block_tables_flat
        },
        max_num_blocks: if batch.is_dummy { 0 } else { max_num_blocks },
    })
}

pub(crate) fn runner_in_vision_slots(_data: &[u8]) -> PyResult<Vec<PyObject>> {
    Ok(Vec::new())
}
