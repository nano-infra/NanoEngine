use axum::{
    extract::{State, Json},
    response::{sse::{Event, Sse}, IntoResponse},
    routing::post,
    Router,
};
use bytes::{Buf, BufMut, BytesMut};
use serde::Deserialize;
use std::{collections::HashMap, error::Error, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::{mpsc, oneshot, Mutex},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

// ================== Spoke Protocol & Structs ==================

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct NetHeader {
    magic: u32,
    meta_size: u32,
    data_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
enum Action {
    Init            = 0x01,
    NetLaunch       = 0x33,
    NetAllocate     = 0x32,
    StreamPush      = 0x40,
    UserActionStart = 0x10,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NetMetaRaw {
    action: u32,
    seq_id: u32,
    actor_id: [u8; 32],
    actor_type: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NetRespMeta {
    seq_id: u32,
    status: i32,
    action: u32,
}

// Helpers
fn string_to_bytes<const N: usize>(s: &str) -> [u8; N] {
    let mut buf = [0u8; N];
    let bytes = s.as_bytes();
    let len = std::cmp::min(bytes.len(), N - 1);
    buf[..len].copy_from_slice(&bytes[..len]);
    buf
}

// ================== Request Structs ==================

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct ResourceSpec {
    num_gpus: i32,
    num_cpus: i32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct AllocateReq {
    num_nodes: u32,
    actors_per_node: u32,
    res_per_actor: ResourceSpec,
    strict_pack: bool,
    master_node_ip: [u8; 64],
}

#[repr(C)]
struct AllocateResp {
    ticket_id: [u8; 64],
    num_members: u32,
}

#[repr(C)]
struct LaunchReq {
    ticket_id: [u8; 64],
    global_rank: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct EngineInitReq {
    config_path: [u8; 256],
    tp: i32,
    pp: i32,
    dp: i32,
    hub_ip: [u8; 64],
    hub_port: i32,
    attention_tp: i32,
    attention_dp: i32,
    attention_sp: i32,
    ffn_tp: i32,
    ffn_dp: i32,
    ffn_ep: i32,
    enable_rdma: bool,
    enable_cuda_graph: bool,
    _padding: [u8; 2],
}

// ================== Internal Logic ==================

type BoxError = Box<dyn Error + Send + Sync>;

struct SpokeConnection {
    stream: TcpStream,
}

impl SpokeConnection {
    async fn connect(addr: &str) -> Result<Self, BoxError> {
        let stream = TcpStream::connect(addr).await.map_err(|e| Box::new(e) as BoxError)?;
        Ok(Self { stream })
    }

    async fn send_req(&mut self, action: u32, seq: u32, actor_id: &str, actor_type: &str, body: &[u8]) -> Result<(), BoxError> {
        let meta = NetMetaRaw {
            action,
            seq_id: seq,
            actor_id: string_to_bytes(actor_id),
            actor_type: string_to_bytes(actor_type),
        };

        let header = NetHeader {
            magic: 0x504F4B45,
            meta_size: std::mem::size_of::<NetMetaRaw>() as u32,
            data_size: body.len() as u32,
        };

        let hdr_bytes = unsafe { std::slice::from_raw_parts(&header as *const _ as *const u8, std::mem::size_of::<NetHeader>()) };
        self.stream.write_all(hdr_bytes).await.map_err(|e| Box::new(e) as BoxError)?;

        let meta_bytes = unsafe { std::slice::from_raw_parts(&meta as *const _ as *const u8, std::mem::size_of::<NetMetaRaw>()) };
        self.stream.write_all(meta_bytes).await.map_err(|e| Box::new(e) as BoxError)?;

        if !body.is_empty() {
            self.stream.write_all(body).await.map_err(|e| Box::new(e) as BoxError)?;
        }
        self.stream.flush().await.map_err(|e| Box::new(e) as BoxError)?;
        Ok(())
    }

    async fn read_msg(&mut self) -> Result<(NetRespMeta, Vec<u8>), BoxError> {
        let mut rh_buf = [0u8; 12];
        self.stream.read_exact(&mut rh_buf).await.map_err(|e| Box::new(e) as BoxError)?;
        let rh: NetHeader = unsafe { std::ptr::read(rh_buf.as_ptr() as *const _) };

        let mut rm_buf = [0u8; 12]; // Note: NetRespMeta is 12 bytes now (seq, status, action)
        self.stream.read_exact(&mut rm_buf).await.map_err(|e| Box::new(e) as BoxError)?;
        let rm: NetRespMeta = unsafe { std::ptr::read(rm_buf.as_ptr() as *const _) };

        let mut body = vec![0u8; rh.data_size as usize];
        if rh.data_size > 0 {
            self.stream.read_exact(&mut body).await.map_err(|e| Box::new(e) as BoxError)?;
        }
        Ok((rm, body))
    }
}

// ================== App State & Actor ==================

enum ClientCmd {
    Request {
        action: u32,
        actor_id: String,
        body: Vec<u8>,
        resp_tx: oneshot::Sender<Result<Vec<u8>, String>>,
    },
    RegisterStream {
        slot_id: u32,
        tx: mpsc::Sender<Result<Vec<i32>, String>>,
    },
}

struct AppState {
    tx: mpsc::Sender<ClientCmd>,
    next_slot_id: Arc<Mutex<u32>>,
}

async fn client_loop(mut conn: SpokeConnection, mut rx: mpsc::Receiver<ClientCmd>) {
    let mut pending_reqs: HashMap<u32, oneshot::Sender<Result<Vec<u8>, String>>> = HashMap::new();
    let mut streams: HashMap<u32, mpsc::Sender<Result<Vec<i32>, String>>> = HashMap::new();
    let mut seq_counter = 0u32;

    loop {
        tokio::select! {
            cmd = rx.recv() => {
                match cmd {
                    Some(ClientCmd::Request { action, actor_id, body, resp_tx }) => {
                        seq_counter += 1;
                        if let Err(e) = conn.send_req(action, seq_counter, &actor_id, "", &body).await {
                            let _ = resp_tx.send(Err(e.to_string()));
                        } else {
                            pending_reqs.insert(seq_counter, resp_tx);
                        }
                    }
                    Some(ClientCmd::RegisterStream { slot_id, tx }) => {
                        streams.insert(slot_id, tx);
                    }
                    None => break, // Channel closed
                }
            }
            res = conn.read_msg() => {
                match res {
                    Ok((meta, body)) => {
                        // Check if it's a push message (Stream)
                        if meta.action == Action::StreamPush as u32 {
                            // Parse StreamToken
                            // Body: size_t seq_id, vector<int> tokens, bool finished

                            // Let's manually unpack carefully
                            let mut ptr = body.as_slice();
                            if ptr.len() >= 17 { // Min size
                                let _req_seq_id = ptr.get_u64_le(); // The request sequence ID this stream belongs to
                                let len = ptr.get_u64_le() as usize;
                                let mut tokens = Vec::new();
                                for _ in 0..len {
                                    if ptr.len() < 4 { break; }
                                    tokens.push(ptr.get_i32_le());
                                }
                                let finished = if ptr.has_remaining() { ptr.get_u8() != 0 } else { false };

                                let slot_id = meta.seq_id;

                                if let Some(tx) = streams.get(&slot_id) {
                                    if !tokens.is_empty() {
                                        let _ = tx.send(Ok(tokens)).await;
                                    }
                                    if finished {
                                        // End of stream
                                        streams.remove(&slot_id);
                                    }
                                }
                            }
                        } else {
                            // Normal Response
                            if let Some(tx) = pending_reqs.remove(&meta.seq_id) {
                                if meta.status > 0 {
                                    let _ = tx.send(Ok(body));
                                } else {
                                    let _ = tx.send(Err("Remote Error".into()));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("[Rust] Connection Error: {}", e);
                        break;
                    }
                }
            }
        }
    }
}

// ================== Web Handlers ==================

#[derive(Deserialize)]
struct ChatReq {
    prompt_ids: Vec<i32>,
    max_tokens: Option<i32>,
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatReq>,
) -> impl IntoResponse {
    let slot_id;
    {
        let mut g = state.next_slot_id.lock().await;
        slot_id = *g;
        *g += 1;
    }

    let (stream_tx, stream_rx) = mpsc::channel(100);

    // Register Stream
    let _ = state.tx.send(ClientCmd::RegisterStream { slot_id, tx: stream_tx }).await;

    // Send Generate Request
    // Serialize EngineAddReq
    let mut add_buf = BytesMut::new();
    add_buf.put_u64_le(req.prompt_ids.len() as u64);
    for p in req.prompt_ids { add_buf.put_i32_le(p); }
    add_buf.put_i32_le(req.max_tokens.unwrap_or(20));
    add_buf.put_u32_le(slot_id);

    let (resp_tx, resp_rx) = oneshot::channel();
    let _ = state.tx.send(ClientCmd::Request {
        action: 18, // kUserActionStart + 2 = AddReq
        actor_id: "EngineActor_0".to_string(),
        body: add_buf.to_vec(),
        resp_tx
    }).await;

    // Wait for Add Ack
        match resp_rx.await {
        Ok(Ok(_)) => {
            // Success, start streaming
            let stream = ReceiverStream::new(stream_rx).map(|res| {
                match res {
                    Ok(tokens) => {
                        // Convert [1, 2] -> "data: [1, 2]\n\n"
                        let json = serde_json::to_string(&tokens).unwrap();
                        Ok::<Event, axum::Error>(Event::default().data(json))
                    }
                    Err(e) => Ok(Event::default().event("error").data(e)),
                }
            });
            Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()).into_response()
        }
        _ => {
             // Create an empty stream with explicit type to satisfy type inference
             let (_, rx) = mpsc::channel::<Result<Event, axum::Error>>(1);
             Sse::new(ReceiverStream::new(rx)).into_response() // Empty/Error
        }
    }
}

// ================== Main ==================

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    tracing_subscriber::fmt::init();
    println!("[Rust] NanoDeploy Server Starting...");

    // 1. Initialization Phase (Sync/One-off)
    // Hub uses short connections (closes after response), so we connect for each request.
    let hub_addr = "127.0.0.1:8888";

    println!("[Rust] Allocating...");
    let mut conn = SpokeConnection::connect(hub_addr).await?;
    let req = AllocateReq {
        num_nodes: 1, actors_per_node: 1, res_per_actor: ResourceSpec::default(),
        strict_pack: true, master_node_ip: string_to_bytes("127.0.0.1"),
    };
    let req_bytes = unsafe { std::slice::from_raw_parts(&req as *const _ as *const u8, std::mem::size_of::<AllocateReq>()) };
    conn.send_req(Action::NetAllocate as u32, 1, "", "", req_bytes).await?;
    let (_, body) = conn.read_msg().await?;
    let resp: AllocateResp = unsafe { std::ptr::read(body.as_ptr() as *const _) };
    let ticket = String::from_utf8_lossy(&resp.ticket_id).trim_matches(char::from(0)).to_string();
    println!("[Rust] Ticket: {}", ticket);
    // Hub closes connection here.

    println!("[Rust] Launching EngineActor...");
    let mut conn = SpokeConnection::connect(hub_addr).await?; // Reconnect for Launch
    let launch_req = LaunchReq { ticket_id: string_to_bytes(&ticket), global_rank: 0 };
    let launch_bytes = unsafe { std::slice::from_raw_parts(&launch_req as *const _ as *const u8, std::mem::size_of::<LaunchReq>()) };
    conn.send_req(Action::NetLaunch as u32, 2, "EngineActor_0", "EngineActor", launch_bytes).await?;
    let _ = conn.read_msg().await?; // Ack

    // Wait for actor startup
    tokio::time::sleep(Duration::from_secs(2)).await;

    // 2. Long-lived Connection to Agent (Hub acts as Agent in this setup usually, or we connect to Hub port)
    // In Spoke, if we want to talk to EngineActor, we connect to the Hub/Agent that hosts it.
    // Hub is 8888 (Short-lived), Agent is 4469 (Long-lived).
    let agent_addr = "127.0.0.1:4469";

    println!("[Rust] Connecting to Agent for Runtime at {}...", agent_addr);
    let mut runtime_conn = SpokeConnection::connect(agent_addr).await?;

    // Parse Arguments
    let args: Vec<String> = std::env::args().collect();
    let enable_cuda_graph = args.iter().any(|a| a == "--enable-cuda-graph");
    println!("[Rust] CUDA Graph Enabled: {}", enable_cuda_graph);

    // Init Engine (One-off via runtime connection)
    println!("[Rust] Initializing Engine...");
    let init_req = EngineInitReq {
        config_path: string_to_bytes("/models/Qwen3-235B-A22B-Instruct-2507/config.json"), // Hardcoded for convenience
        tp: 1, pp: 1, dp: 1, hub_ip: string_to_bytes("127.0.0.1"), hub_port: 8888,
        attention_tp: 1, attention_dp: 8, attention_sp: 1, ffn_tp: 1, ffn_dp: 1, ffn_ep: 8,
        enable_rdma: true, enable_cuda_graph: enable_cuda_graph, _padding: [0; 2],
    };
    let init_bytes = unsafe { std::slice::from_raw_parts(&init_req as *const _ as *const u8, std::mem::size_of::<EngineInitReq>()) };
    runtime_conn.send_req(17, 3, "EngineActor_0", "", init_bytes).await?;
    let (_, _) = runtime_conn.read_msg().await?; // Wait for Init Ack
    println!("[Rust] Engine Initialized.");

    // 3. Start Client Loop
    let (tx, rx) = mpsc::channel(100);
    tokio::spawn(client_loop(runtime_conn, rx));

    // 4. Start Web Server
    let app_state = Arc::new(AppState {
        tx,
        next_slot_id: Arc::new(Mutex::new(1000)),
    });

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("[Rust] Server listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;

    Ok(())
}
