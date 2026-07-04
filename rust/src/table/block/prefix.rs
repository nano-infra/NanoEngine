pub(crate) fn compute_block_hash(tokens: &[i32], prefix: i64) -> i64 {
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
