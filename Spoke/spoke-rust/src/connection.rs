use bytes::{Buf, BufMut, BytesMut};
use std::error::Error;
use std::io::Cursor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[allow(dead_code)]
struct NetHeader {
    magic: u32,
    meta_size: u32,
    data_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NetMetaRaw {
    pub action: u32,
    pub seq_id: u32,
    pub actor_id: [u8; 32],
    pub actor_type: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NetRespMeta {
    pub seq_id: u32,
    pub status: i32,
    pub action: u32,
}

use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

pub struct SpokeConnection {
    stream: TcpStream,
}

pub struct SpokeReader {
    stream: OwnedReadHalf,
}

pub struct SpokeWriter {
    stream: OwnedWriteHalf,
}

impl SpokeConnection {
    pub async fn connect(addr: &str) -> Result<Self, BoxError> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| Box::new(e) as BoxError)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub fn into_split(self) -> (SpokeReader, SpokeWriter) {
        let (read, write) = self.stream.into_split();
        (SpokeReader { stream: read }, SpokeWriter { stream: write })
    }
}

impl SpokeWriter {
    pub fn string_to_bytes<const N: usize>(s: &str) -> [u8; N] {
        let mut buf = [0u8; N];
        let bytes = s.as_bytes();
        let len = std::cmp::min(bytes.len(), N - 1);
        buf[..len].copy_from_slice(&bytes[..len]);
        buf
    }

    pub async fn send_req(
        &mut self,
        action: u32,
        seq: u32,
        actor_id: &str,
        actor_type: &str,
        body: &[u8],
    ) -> Result<(), BoxError> {
        let meta = NetMetaRaw {
            action,
            seq_id: seq,
            actor_id: Self::string_to_bytes(actor_id),
            actor_type: Self::string_to_bytes(actor_type),
        };

        // Serialize Header
        let mut head_buf = BytesMut::with_capacity(12);
        head_buf.put_u32_le(0x504F4B45); // Magic
        head_buf.put_u32_le(std::mem::size_of::<NetMetaRaw>() as u32);
        head_buf.put_u32_le(body.len() as u32);

        // Serialize Meta
        let meta_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const _ as *const u8,
                std::mem::size_of::<NetMetaRaw>(),
            )
        };

        self.stream.write_all(&head_buf).await?;
        self.stream.write_all(meta_bytes).await?;
        if !body.is_empty() {
            self.stream.write_all(body).await?;
        }
        self.stream.flush().await?;
        Ok(())
    }
}

impl SpokeReader {
    pub async fn read_msg(&mut self) -> Result<(NetRespMeta, Vec<u8>), BoxError> {
        // Read Header
        let mut rh_buf = [0u8; 12];
        self.stream.read_exact(&mut rh_buf).await?;
        let mut rh_cur = Cursor::new(&rh_buf);
        let magic = rh_cur.get_u32_le();
        let _meta_size = rh_cur.get_u32_le();
        let data_size = rh_cur.get_u32_le();

        if magic != 0x504F4B45 {
            return Err(format!("Invalid Magic: expected 0x504F4B45, got 0x{:x}", magic).into());
        }

        // Read Meta (12 bytes for NetRespMeta)
        let mut rm_buf = [0u8; 12];
        self.stream.read_exact(&mut rm_buf).await?;
        let rm: NetRespMeta = unsafe { std::ptr::read(rm_buf.as_ptr() as *const _) };

        // Read Body
        let mut body = vec![0u8; data_size as usize];
        if data_size > 0 {
            self.stream.read_exact(&mut body).await?;
        }
        Ok((rm, body))
    }
}
