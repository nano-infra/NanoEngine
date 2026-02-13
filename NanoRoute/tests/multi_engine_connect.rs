use nanodeploy_server::config::{EngineConfig, EngineNode};
use nanodeploy_server::engine_manager::EngineManager;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::sleep;

async fn start_mock_engine(port: u16) {
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)
        .await
        .expect("Failed to bind mock engine");
    tokio::spawn(async move {
        loop {
            // Accept and drop, just to keep port open and allow connect
            let _ = listener.accept().await;
        }
    });
}

#[tokio::test]
async fn test_multi_engine_connect() {
    // 1. Start Mock Engines
    start_mock_engine(6000).await;
    start_mock_engine(7000).await;

    // Allow startup
    sleep(Duration::from_millis(100)).await;

    // 2. Create Config
    let config = EngineConfig::Disaggregated {
        prefill: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 6000,
        }],
        decode: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 7000,
        }],
    };

    // 3. Connect
    let mut manager = EngineManager::new();
    let result = manager.connect_all(&config).await;

    assert!(
        result.is_ok(),
        "Failed to connect to engines: {:?}",
        result.err()
    );

    // 4. Verify internal state (requires public access or specific method, for now assume OK result implies success)
    // We can add a method to manager to get counts if needed, but connect_all returns error if any fail.
}
