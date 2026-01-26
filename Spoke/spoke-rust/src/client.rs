use crate::connection::{SpokeConnection, SpokeWriter, SpokeReader};

pub struct SpokeClient {
    pub conn: Option<SpokeWriter>,
    pub id: String,
}

impl SpokeClient {
    pub fn new(id: String) -> Self {
        Self { conn: None, id }
    }

    // Connects and returns the Reader half for the caller to handle loop
    pub async fn connect(&mut self, addr: &str) -> anyhow::Result<SpokeReader> {
        let conn = SpokeConnection::connect(addr).await.map_err(|e| anyhow::anyhow!(e))?;
        let (reader, writer) = conn.into_split();
        self.conn = Some(writer);
        Ok(reader)
    }

    pub async fn send_message(&mut self, action: u32, seq_id: u64, payload: &[u8]) -> anyhow::Result<()> {
        if let Some(conn) = &mut self.conn {
            conn.send_req(action, seq_id as u32, &self.id, "Client", payload)
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
        } else {
            return Err(anyhow::anyhow!("Not connected"));
        }
        Ok(())
    }
}
