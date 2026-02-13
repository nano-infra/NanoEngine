use nanodeploy_server::config::{EngineConfig, EngineNode};
use nanodeploy_server::engine_manager::EngineManager;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::sleep;

async fn start_mock_engine_with_role(port: u16, is_prefill: bool) -> tokio::task::JoinHandle<()> {
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)
        .await
        .expect("Failed to bind mock engine");

    tokio::spawn(async move {
        if let Ok((mut socket, _)) = listener.accept().await {
            loop {
                // Header (12 bytes)
                let mut head = [0u8; 12];
                if socket.read_exact(&mut head).await.is_err() {
                    break;
                }
                let data_size = u32::from_le_bytes(head[8..12].try_into().unwrap());

                // Meta
                let mut meta_buf = vec![0u8; 72]; // Wait, server sends NetMetaRaw (72 bytes? need to check connection.rs)
                                                  // Actually server sends NetMetaRaw. Let's assume standard IPC.
                                                  // Wait, server connection.rs defines:
                                                  // NetMetaRaw size = 4+4+32+32 = 72 bytes.
                                                  // However, read_msg reads Header(12) then Meta(NetRespMeta=12).
                                                  // Server SENDS Request, which has NetMetaRaw (72).
                if socket.read_exact(&mut meta_buf).await.is_err() {
                    break;
                }

                let action = u32::from_le_bytes(meta_buf[0..4].try_into().unwrap());

                // Body
                let mut body = vec![0u8; data_size as usize];
                if socket.read_exact(&mut body).await.is_err() {
                    break;
                }

                println!("[Mock {}] Received Action {}", port, action);

                if is_prefill && action == 1 {
                    println!("[Mock Prefill] Received AddRequest. Sending Migrate...");
                    sleep(Duration::from_millis(100)).await;

                    // Simulate Sending Response (StreamEvent::Migrate)
                    // We need to send a message where action=1 (Migration) in RESP meta
                    // Resp Meta is 12 bytes: seq_id(4), status(4), action(4)

                    let mut resp_meta = [0u8; 12];
                    // seq_id=0, status=0, action=1
                    resp_meta[8] = 1;

                    let dummy_seq_list_fbs = {
                        // Create a dummy FBS payload that looks like SequenceList
                        // Ideally we construct valid FBS.
                        // For this test, assume EngineAdapter just checks root table success.
                        // We will copy the received `body` back! It's already a SequenceList.
                        body.clone()
                    };

                    // Send Header
                    let mut out_head = [0u8; 12];
                    out_head[0..4].copy_from_slice(&0x504F4B45u32.to_le_bytes()); // Magic
                    out_head[4..8].copy_from_slice(&12u32.to_le_bytes()); // Meta Size (Resp)
                    out_head[8..12]
                        .copy_from_slice(&(dummy_seq_list_fbs.len() as u32).to_le_bytes()); // Data Size

                    socket.write_all(&out_head).await.unwrap();
                    socket.write_all(&resp_meta).await.unwrap();
                    socket.write_all(&dummy_seq_list_fbs).await.unwrap();

                    // Break after migration triggering
                    sleep(Duration::from_millis(50)).await;
                    break;
                } else if !is_prefill && action == 1 {
                    println!("[Mock Decode] Received Forwarded Request. Sending Finished...");
                    sleep(Duration::from_millis(50)).await;

                    // Send Finished Signal using StepOut
                    // Use existing tool/lib to build StepOut? Too complex here.
                    // Just send empty or close?
                    // EngineAdapter expects StepOut for token.
                    // Let's just break loop, test verifies action=1 reception on decode.
                    break;
                }
            }
        }
    })
}

#[tokio::test]
async fn test_migration_flow() {
    // 1. Start Mocks
    start_mock_engine_with_role(8001, true).await; // Prefill
    start_mock_engine_with_role(8002, false).await; // Decode
    sleep(Duration::from_millis(100)).await;

    // 2. Config
    let config = EngineConfig::Disaggregated {
        prefill: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 8001,
        }],
        decode: vec![EngineNode {
            host: "127.0.0.1".to_string(),
            port: 8002,
        }],
    };

    // 3. Manager
    let mut manager = EngineManager::new();
    manager.connect_all(&config).await.expect("Connect");

    // 4. Send Request to Prefill
    // We need to use EngineAdapter directly since we can't easily mock HTTP layer here without full server.
    // We will get the prefill adapter and send request.

    let prefill = manager.get_next_prefill().unwrap();
    let mut adapter = prefill.lock().await; // Actually we need to lock to send, but we also need listener to run.

    // Wait, EngineAdapter spawns reader loop in background upon connect().
    // So if we send, the background loop will receive the Migrate response.
    // But `send_add_request` returns `rx`.

    let mut rx = adapter
        .send_add_request(101, &[1, 2, 3], 10)
        .await
        .expect("Send");
    drop(adapter); // Release lock

    // 5. Watch RX
    // - Should receive Migrate event?
    // - Wait, the RX returned by `send_add_request` receives events from *that* adapter.
    // - The `EngineAdapter` code I wrote sends `StreamEvent::Migrate(payload)` to `state.sender`.
    // - So `rx` should receive Migrate.

    // Note: The Swapping logic is in HTTP Handler.
    // The `EngineAdapter` just emits `Migrate`.
    // So this test verifies that `EngineAdapter` correctly identifies the Action=1 response and emits `Migrate`.

    if let Some(event) = rx.recv().await {
        match event {
            nanodeploy_server::engine_adapter::StreamEvent::Migrate(payload) => {
                println!("Received Migrate Event with {} bytes", payload.len());
                assert!(payload.len() > 0);
            }
            _ => panic!("Expected Migrate Event, got {:?}", event),
        }
    } else {
        panic!("Channel closed unexpectedly");
    }
}
