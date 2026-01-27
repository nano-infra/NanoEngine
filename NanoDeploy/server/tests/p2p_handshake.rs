use nanodeploy_server::config::{EngineConfig, EngineNode};
use nanodeploy_server::engine_manager::EngineManager;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::time::sleep;

async fn start_mock_engine_p2p(port: u16) -> tokio::task::JoinHandle<()> {
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)
        .await
        .expect("Failed to bind mock engine");

    tokio::spawn(async move {
        if let Ok((mut socket, _)) = listener.accept().await {
            // Read Loop
            loop {
                // Read Header (12 bytes)
                let mut head = [0u8; 12];
                if socket.read_exact(&mut head).await.is_err() {
                    break;
                }

                // Parse Header
                // Layout: Magic(4), MetaSize(4), DataSize(4)
                let magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
                let meta_size = u32::from_le_bytes(head[4..8].try_into().unwrap());
                let data_size = u32::from_le_bytes(head[8..12].try_into().unwrap());

                if magic != 0x504F4B45 {
                    break;
                }

                // Read Meta
                let mut meta = vec![0u8; meta_size as usize];
                if socket.read_exact(&mut meta).await.is_err() {
                    break;
                }

                // Read Action from Meta (first 4 bytes)
                let action = u32::from_le_bytes(meta[0..4].try_into().unwrap());

                // Read Body
                let mut body = vec![0u8; data_size as usize];
                if socket.read_exact(&mut body).await.is_err() {
                    break;
                }

                println!("MockEngine {} Received Action: {}", port, action);

                // Validation Logic
                if action == 3 {
                    // P2P Init
                    println!("Verified P2PInit on {}", port);
                } else if action == 4 {
                    // P2P Connect
                    println!("Verified P2PConnect on {}", port);
                }
            }
        }
    })
}

#[tokio::test]
async fn test_p2p_handshake() {
    // 1. Start Mock Engines
    start_mock_engine_p2p(6001).await; // Prefill
    start_mock_engine_p2p(7001).await; // Decode

    sleep(Duration::from_millis(100)).await;

    // 2. Config
    let config = EngineConfig::Disaggregated {
        prefill: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 6001,
        }],
        decode: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 7001,
        }],
    };

    // 3. Manager
    let mut manager = EngineManager::new();
    manager.connect_all(&config).await.expect("Connect");

    // 4. Handshake
    let res = manager.initialize_p2p_mesh(&config).await;
    assert!(res.is_ok(), "Handshake failed");

    // Give time for mocks to log
    sleep(Duration::from_millis(200)).await;
}
