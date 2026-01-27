use nanodeploy_server::engine_adapter::EngineAdapter;
use spoke::client::SpokeClient;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_mock_engine_ipc() {
    // 1. Start a simple TCP Echo Server (Mock Engine)
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind");
    let addr = listener.local_addr().expect("Failed to get addr");
    println!("Mock Engine listening on {}", addr);

    let server_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("Failed to accept");

        // Read Magic (4) + MetaSize (4) + DataSize (4)
        let mut head = [0u8; 12];
        socket
            .read_exact(&mut head)
            .await
            .expect("Failed to read header");
        let magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
        assert_eq!(magic, 0x504F4B45);

        let meta_size = u32::from_le_bytes(head[4..8].try_into().unwrap());

        // Read Meta
        let mut meta = vec![0u8; meta_size as usize];
        socket
            .read_exact(&mut meta)
            .await
            .expect("Failed to read meta");

        let data_size = u32::from_le_bytes(head[8..12].try_into().unwrap());
        if data_size > 0 {
            let mut data = vec![0u8; data_size as usize];
            socket
                .read_exact(&mut data)
                .await
                .expect("Failed to read data");
            assert!(data.len() > 10);
        }
    });

    // 2. Client uses EngineAdapter (which wraps Spoke)
    let mut client = EngineAdapter::new("integration_test".to_string());
    client
        .connect(&addr.to_string())
        .await
        .expect("Connect failed");

    // Send an application-specific request (Add Sequence)
    let tokens = vec![1, 2, 3];
    client
        .send_add_request(999, &tokens)
        .await
        .expect("Send failed");

    server_task.await.expect("Server task failed");
}
