use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub model_config_path: String,
    pub ffn_ep: i32,
    pub ffn_tp: i32,
    pub ffn_dp: i32,
    pub pp: i32,
    pub attention_dp: i32,
    pub attention_tp: i32,
    pub attention_sp: i32,
    pub enable_cuda_graph: bool,
    pub enable_rdma: bool,
}

impl Config {
    pub fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut config = Config {
            model_config_path: String::new(),
            ffn_ep: 1,
            ffn_tp: 1,
            ffn_dp: 1,
            pp: 1,
            attention_dp: 1,
            attention_tp: 1,
            attention_sp: 1,

            enable_cuda_graph: false,
            enable_rdma: false,
        };

        let mut i = 1;
        while i < args.len() {
            let arg = &args[i];
            match arg.as_str() {
                "--model-config-path" => {
                    if i + 1 < args.len() {
                        config.model_config_path = args[i + 1].clone();
                        i += 1;
                    }
                }
                "--ffn-ep" => {
                    if i + 1 < args.len() {
                        config.ffn_ep = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--ffn-tp" => {
                    if i + 1 < args.len() {
                        config.ffn_tp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--ffn-dp" => {
                    if i + 1 < args.len() {
                        config.ffn_dp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--pp" => {
                    if i + 1 < args.len() {
                        config.pp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--attention-dp" => {
                    if i + 1 < args.len() {
                        config.attention_dp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--attention-tp" => {
                    if i + 1 < args.len() {
                        config.attention_tp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--attention-sp" => {
                    if i + 1 < args.len() {
                        config.attention_sp = args[i + 1].parse().unwrap_or(1);
                        i += 1;
                    }
                }
                "--enable-cuda-graph" => {
                    config.enable_cuda_graph = true;
                }
                "--enable-rdma" => {
                    config.enable_rdma = true;
                }
                _ => {
                    // Ignore unknown args or handle error? For now ignore.
                }
            }
            i += 1;
        }

        if config.model_config_path.is_empty() {
            eprintln!("Error: --model-config-path is required");
            std::process::exit(1);
        }

        config
    }
}
