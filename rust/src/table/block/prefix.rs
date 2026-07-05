use xxhash_rust::xxh3::Xxh3;

pub(crate) fn compute_block_hash(tokens: &[i32], prefix: i64) -> i64 {
    let seed = if prefix == -1 { 0 } else { prefix as u64 };
    let mut hasher = Xxh3::with_seed(seed);
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
    hasher.digest() as i64
}
