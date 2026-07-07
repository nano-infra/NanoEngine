pub(crate) enum CacheHit {
    Session { new_tokens: i32 },
    Prefix { cached_tokens: i32 },
    None,
}
