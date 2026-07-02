use super::Scheduler;
use pyo3::prelude::*;
use std::cmp::Reverse;

impl Scheduler {
    pub(super) fn dispatch_for_master(&self, master: usize, tokens: i32) -> Vec<i32> {
        let mut out = vec![0; self.group()];
        if master < out.len() {
            out[master] = tokens.max(0);
        }
        out
    }

    pub(super) fn compute_dispatch(
        &self,
        py: Python<'_>,
        dp_idx: usize,
        master: usize,
        tokens: i32,
    ) -> Vec<i32> {
        let group = self.group();
        if group <= 1 || tokens <= 256 {
            return self.dispatch_for_master(master, tokens);
        }
        let mut out = vec![0; group];
        let mut loads = self.group_loads(py, dp_idx);
        let mut remaining = tokens;
        while remaining > 0 {
            let gid = (0..group)
                .min_by_key(|gid| {
                    (
                        loads[*gid],
                        Reverse(
                            self.group_resources[self.flat_idx(dp_idx, *gid)]
                                .free_blocks
                                .len(),
                        ),
                    )
                })
                .unwrap_or(master);
            let take = remaining.min(self.config.kvcache_block_size.max(1));
            out[gid] += take;
            loads[gid] += take;
            remaining -= take;
        }
        if out[master] == 0 {
            if let Some(gid) = (0..group).max_by_key(|gid| out[*gid]) {
                out[master] = out[gid];
                out[gid] = 0;
            }
        }
        out
    }

    pub(super) fn make_dummy_seq(&self, py: Python<'_>, group_id: i32) -> PyResult<Py<PyAny>> {
        let module = py.import("dlengine._dlengine_rust")?;
        let sp = module.getattr("SamplingParams")?.call0()?;
        let seq = module.getattr("Sequence")?.call1((vec![0i32], sp))?;
        seq.setattr("status", 1)?;
        seq.setattr("active_group_id", group_id)?;
        seq.call_method1(
            "set_active_dispatched_tokens",
            (self.dispatch_for_master(group_id as usize, 1),),
        )?;
        Ok(seq.unbind())
    }

    pub(super) fn blocks_needed_for_tokens(&self, tokens: i32) -> usize {
        let block_size = self.config.kvcache_block_size.max(1);
        ((tokens.max(1) + block_size - 1) / block_size) as usize
    }

    pub(super) fn group_loads(&self, py: Python<'_>, dp_idx: usize) -> Vec<i32> {
        let mut loads = vec![0; self.group()];
        for seq in &self.running[dp_idx] {
            let obj = seq.bind(py);
            let gid = obj
                .getattr("active_group_id")
                .and_then(|v| v.extract::<usize>())
                .unwrap_or(0)
                .min(self.group() - 1);
            loads[gid] += obj
                .getattr("num_tokens")
                .and_then(|v| v.extract::<i32>())
                .unwrap_or(0);
        }
        loads
    }

    pub(super) fn choose_master_group(
        &self,
        py: Python<'_>,
        dp_idx: usize,
        batch_seqs: &[i32],
        batch_tokens: &[i32],
    ) -> Option<usize> {
        let mut candidates: Vec<usize> = (0..self.group()).collect();
        let loads = self.group_loads(py, dp_idx);
        candidates.sort_by_key(|gid| {
            (
                batch_seqs.get(*gid).copied().unwrap_or(0),
                loads.get(*gid).copied().unwrap_or(0)
                    + batch_tokens.get(*gid).copied().unwrap_or(0),
                Reverse(
                    self.group_resources[self.flat_idx(dp_idx, *gid)]
                        .free_blocks
                        .len(),
                ),
            )
        });
        candidates.into_iter().find(|gid| {
            batch_seqs.get(*gid).copied().unwrap_or(0) < self.config.max_num_seqs.max(1)
        })
    }
}
