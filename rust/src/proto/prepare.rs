use super::metadata::{BatchAuxData, DecodeControl, DecodeMeta, PrefillMeta};
use super::wire::{
    decode_wire, DECODE_FLAG_ALL_GREEDY, DECODE_FLAG_COMPLETION_LOGPROBS, DECODE_FLAG_DUMMY,
    DECODE_FLAT_HEADER_BYTES, DECODE_FLAT_MAGIC, DECODE_FLAT_VERSION,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn checked_array(
    payload_len: usize,
    name: &str,
    offset: usize,
    count: usize,
    element_bytes: usize,
) -> PyResult<(usize, usize)> {
    if offset < DECODE_FLAT_HEADER_BYTES || offset % 16 != 0 {
        return Err(PyValueError::new_err(format!(
            "flat decode {name} offset {offset} is not 16-byte aligned"
        )));
    }
    let bytes = count
        .checked_mul(element_bytes)
        .ok_or_else(|| PyValueError::new_err(format!("flat decode {name} size overflows")))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| PyValueError::new_err(format!("flat decode {name} range overflows")))?;
    if end > payload_len {
        return Err(PyValueError::new_err(format!(
            "flat decode {name} range [{offset}, {end}) exceeds payload {payload_len}"
        )));
    }
    Ok((offset, end))
}

#[pyfunction]
pub(crate) fn decode_flat_control(data: &Bound<'_, PyAny>) -> PyResult<DecodeControl> {
    let bytes = data
        .downcast::<PyBytes>()
        .map_err(|_| PyValueError::new_err("flat decode payload must be bytes"))?;
    let data = bytes.as_bytes();
    if data.len() < DECODE_FLAT_HEADER_BYTES {
        return Err(PyValueError::new_err(format!(
            "flat decode payload is truncated: {} < {DECODE_FLAT_HEADER_BYTES}",
            data.len()
        )));
    }
    let magic = read_u32(data, 0);
    let version = read_u16(data, 4);
    let flags = read_u16(data, 6);
    let payload_bytes = read_u32(data, 8) as usize;
    let num_seqs = read_u32(data, 12) as usize;
    let max_num_seqs = read_u32(data, 16) as usize;
    let max_num_blocks = read_u32(data, 20) as usize;
    let block_size = read_u32(data, 24) as usize;
    let offsets = [
        read_u32(data, 28) as usize,
        read_u32(data, 32) as usize,
        read_u32(data, 36) as usize,
        read_u32(data, 40) as usize,
        read_u32(data, 44) as usize,
        read_u32(data, 48) as usize,
        read_u32(data, 52) as usize,
        read_u32(data, 56) as usize,
    ];
    let block_count = read_u32(data, 60) as usize;

    if magic != DECODE_FLAT_MAGIC {
        return Err(PyValueError::new_err(format!(
            "flat decode magic mismatch: 0x{magic:08x}"
        )));
    }
    if version != DECODE_FLAT_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported flat decode version {version}, expected {DECODE_FLAT_VERSION}"
        )));
    }
    if payload_bytes != data.len() {
        return Err(PyValueError::new_err(format!(
            "flat decode payload length mismatch: header={payload_bytes}, actual={}",
            data.len()
        )));
    }
    if num_seqs == 0 || max_num_seqs == 0 || num_seqs > max_num_seqs {
        return Err(PyValueError::new_err(format!(
            "flat decode invalid batch dimensions: num_seqs={num_seqs}, max_num_seqs={max_num_seqs}"
        )));
    }
    if max_num_blocks == 0 || block_size == 0 {
        return Err(PyValueError::new_err(
            "flat decode block dimensions must be positive",
        ));
    }
    if block_count > num_seqs.saturating_mul(max_num_blocks) {
        return Err(PyValueError::new_err(format!(
            "flat decode block_count {block_count} exceeds batch capacity {}",
            num_seqs * max_num_blocks
        )));
    }

    let specs = [
        ("input_ids", offsets[0], num_seqs, 8),
        ("positions", offsets[1], num_seqs, 8),
        ("temperatures", offsets[2], num_seqs, 4),
        ("state_slots", offsets[3], num_seqs, 8),
        ("hisparse_slots", offsets[4], num_seqs, 8),
        ("block_row_offsets", offsets[5], num_seqs + 1, 4),
        ("block_ids", offsets[6], block_count, 4),
        ("seq_ids", offsets[7], num_seqs, 8),
    ];
    let mut non_empty_ranges = Vec::new();
    for (name, offset, count, element_bytes) in specs {
        let range = checked_array(payload_bytes, name, offset, count, element_bytes)?;
        if range.0 != range.1 {
            non_empty_ranges.push((range.0, range.1, name));
        }
    }
    non_empty_ranges.sort_by_key(|range| range.0);
    for pair in non_empty_ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(PyValueError::new_err(format!(
                "flat decode arrays {} and {} overlap",
                pair[0].2, pair[1].2
            )));
        }
    }

    let row_offset = offsets[5];
    let mut previous = 0usize;
    for row in 0..=num_seqs {
        let current = read_u32(data, row_offset + row * 4) as usize;
        if current < previous || current > block_count {
            return Err(PyValueError::new_err(format!(
                "flat decode block row offsets are invalid at row {row}: {current} after {previous}"
            )));
        }
        if row > 0 && current - previous > max_num_blocks {
            return Err(PyValueError::new_err(format!(
                "flat decode row {} has {} blocks, exceeds {max_num_blocks}",
                row - 1,
                current - previous
            )));
        }
        previous = current;
    }
    if previous != block_count {
        return Err(PyValueError::new_err(format!(
            "flat decode final row offset {previous} does not match block_count {block_count}"
        )));
    }
    let block_ids_offset = offsets[6];
    for index in 0..block_count {
        let offset = block_ids_offset + index * 4;
        let block_id = i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if block_id < 0 {
            return Err(PyValueError::new_err(format!(
                "flat decode block id at index {index} is negative"
            )));
        }
    }

    let positions_offset = offsets[1];
    let mut page_plan_key = Vec::with_capacity(num_seqs);
    for index in 0..num_seqs {
        let offset = positions_offset + index * 8;
        let position = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        if position < 0 {
            return Err(PyValueError::new_err(format!(
                "flat decode position at index {index} is negative"
            )));
        }
        page_plan_key.push((position as usize + 1).div_ceil(block_size));
    }

    Ok(DecodeControl {
        num_group_seqs: num_seqs,
        payload_bytes,
        max_num_seqs,
        max_num_blocks,
        block_size,
        block_count,
        page_plan_key,
        is_dummy: flags & DECODE_FLAG_DUMMY != 0,
        all_greedy: flags & DECODE_FLAG_ALL_GREEDY != 0,
        any_return_completion_logprobs: flags & DECODE_FLAG_COMPLETION_LOGPROBS != 0,
    })
}

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
