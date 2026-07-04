#[path = "mod.rs"]
mod router;

fn main() -> anyhow::Result<()> {
    router::run_from_iter(std::env::args())
}
