use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct PeerAgent {
    pub name: String,
    pub device: String,
    pub ib_port: u32,
    pub link_type: String,
    pub address: String, // e.g., "ip:port"
}

#[derive(Debug, Deserialize)]
pub struct QueryBody {
    // Add query parameters if needed
}

#[derive(Debug, Deserialize)]
pub struct StartPeerAgentBody {
    pub alias: String,
    pub device: String,
    pub ib_port: u32,
    pub link_type: String,
    pub address: String, // IP address
}

#[derive(Debug, Deserialize)]
pub struct InitBody {
    pub src: String,
    pub dst: String,
    pub qp_num: u32,
}

#[derive(Debug, Deserialize)]
pub struct ConnectBody {
    pub src: String,
    pub dst: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MrInfo {
    pub addr: u64,
    pub length: usize,
    pub rkey: u32,
    pub lkey: u32,
}

#[derive(Debug, Deserialize)]
pub struct RegisterMrBody {
    pub agent_name: String,
    pub mr_name: String,
    pub addr: u64,
    pub length: usize,
    pub rkey: u32,
    #[serde(default)]
    pub lkey: u32, // Optional, local key (not needed for remote access)
}

#[derive(Debug, Deserialize)]
pub struct GetMrInfoBody {
    #[allow(dead_code)] // API field, reserved for future use
    pub src: String, // Who is asking
    pub dst: String, // Whose MR to get
    pub mr_name: String,
}

#[derive(Debug, Deserialize)]
pub struct GetEndpointInfoBody {
    pub src: String,
    pub dst: String,
}

#[derive(Debug, Serialize)]
pub struct GetEndpointInfoResponse {
    pub endpoint_info: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct InitResponse {
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ConnectResponse {
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterMrResponse {
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct GetMrInfoResponse {
    pub mr_info: Option<MrInfo>,
}

#[derive(Debug, Deserialize)]
pub struct AckInitBody {
    pub src: String,
    pub dst: String,
    pub endpoint_info: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AckConnectBody {
    pub src: String,
    pub dst: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateEndpointInfoBody {
    pub agent_name: String,
    pub endpoint_info: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct AckResponse {
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct CleanupBody {
    pub agent_name: String,
}

#[derive(Debug, Serialize)]
pub struct CleanupResponse {
    pub status: String,
    pub message: String,
}
