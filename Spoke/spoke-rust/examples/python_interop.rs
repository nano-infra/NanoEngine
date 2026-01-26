use spoke::client::SpokeClient;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Wait for python to start (manual coordination usually, but here we assume it's running)
    let addr = "127.0.0.1:5555";
    println!("Connecting to Python Engine at {}", addr);

    let mut client = SpokeClient::new("rust_interop_client".to_string());

    // Retry loop
    for _ in 0..5 {
        if let Ok(_) = client.connect(addr).await {
            break;
        }
        sleep(Duration::from_millis(500)).await;
    }

    if client.conn.is_none() {
        anyhow::bail!("Failed to connect to Python Engine");
    }

    println!("Connected! Sending payload...");

    let payload = b"Hello from Rust!";
    client.send_message(1, 100, payload).await?;

    println!("Message sent. Waiting for echo is not impl in GenericClient yet (it returns Ok on send).");
    // Note: client.send_message is fire-and-forget or requires read loop?
    // SpokeClient::send_message calls conn.send_req which flushes.
    // It does NOT wait for response currently in the generic impl I wrote (checks impl).

    Ok(())
}
