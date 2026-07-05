use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct SlotPool {
    free_slots: Vec<i32>,
    seq_slots: HashMap<u64, i32>,
}

impl SlotPool {
    pub(crate) fn new(num_slots: i32) -> Self {
        Self {
            free_slots: (0..num_slots.max(0)).rev().collect(),
            seq_slots: HashMap::new(),
        }
    }

    pub(crate) fn ensure(&mut self, seq_id: u64) -> Option<i32> {
        if let Some(slot) = self.seq_slots.get(&seq_id).copied() {
            return Some(slot);
        }
        let slot = self.free_slots.pop()?;
        self.seq_slots.insert(seq_id, slot);
        Some(slot)
    }

    pub(crate) fn get(&self, seq_id: u64) -> Option<i32> {
        self.seq_slots.get(&seq_id).copied()
    }

    pub(crate) fn insert_existing(&mut self, seq_id: u64, slot: i32) {
        self.free_slots.retain(|free| *free != slot);
        self.seq_slots.insert(seq_id, slot);
    }

    pub(crate) fn remove(&mut self, seq_id: u64) -> Option<i32> {
        let slot = self.seq_slots.remove(&seq_id)?;
        self.release_slot(slot);
        Some(slot)
    }

    pub(crate) fn take_without_release(&mut self, seq_id: u64) -> Option<i32> {
        self.seq_slots.remove(&seq_id)
    }

    pub(crate) fn release_slot(&mut self, slot: i32) {
        if slot >= 0 && !self.free_slots.contains(&slot) {
            self.free_slots.push(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_release_and_insert_existing() {
        let mut pool = SlotPool::new(2);
        let a = pool.ensure(10).unwrap();
        assert_eq!(pool.ensure(10), Some(a));
        assert_eq!(pool.remove(10), Some(a));
        let b = pool.ensure(11).unwrap();
        assert_eq!(b, a);
        pool.insert_existing(12, 1);
        assert_eq!(pool.get(12), Some(1));
        assert_eq!(pool.take_without_release(12), Some(1));
        pool.release_slot(1);
    }
}
