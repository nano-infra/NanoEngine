use spoke::client::SpokeClient;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_generic_echo() {
    // 1. Mock Engine
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind");
    let addr = listener.local_addr().expect("Failed to get addr");

    let server_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("Accept failed");

        let mut head = [0u8; 12];
        socket
            .read_exact(&mut head)
            .await
            .expect("Header read failed");

        // Skip parsing for this test, just read correct amount
        let meta_size = u32::from_le_bytes(head[4..8].try_into().unwrap());
        let mut meta = vec![0u8; meta_size as usize];
        socket
            .read_exact(&mut meta)
            .await
            .expect("Meta read failed");

        let data_size = u32::from_le_bytes(head[8..12].try_into().unwrap());
        let mut data = vec![0u8; data_size as usize];
        socket
            .read_exact(&mut data)
            .await
            .expect("Data read failed");

        // Validate Data content (byte-for-byte)
        // We expect the client to send [0xAA; 1024]
        assert!(data.iter().all(|&b| b == 0xAA));
    });

    // 2. Generic Client
    let mut client = SpokeClient::new("test_generic".to_string());
    client
        .connect(&addr.to_string())
        .await
        .expect("Connect failed");

    let payload = vec![0xAAu8; 1024];
    // Action 1, Seq 100
    client
        .send_message(1, 100, &payload)
        .await
        .expect("Send failed");

    server_task.await.expect("Server task failed");
}
