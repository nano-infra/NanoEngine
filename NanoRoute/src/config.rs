use config::{Config, ConfigError, File};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub port: u16,
    #[allow(dead_code)]
    pub model_name: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct TokenizerConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct EngineNode {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
#[serde(tag = "mode")]
pub enum EngineConfig {
    Unified {
        host: String,
        port: u16,
        #[serde(default)]
        nanoctrl_address: Option<String>, // e.g., "http://127.0.0.1:3000"
        #[serde(default)]
        redis_url: Option<String>, // e.g., "redis://127.0.0.1:6379"
        #[serde(default)]
        nanoctrl_scope: Option<String>,
    },
    Disaggregated {
        #[serde(default)]
        prefill: Vec<EngineNode>,
        #[serde(default)]
        decode: Vec<EngineNode>,
        #[serde(default)]
        nanoctrl_address: Option<String>, // e.g., "http://127.0.0.1:3000"
        #[serde(default)]
        redis_url: Option<String>, // e.g., "redis://127.0.0.1:6379"
        #[serde(default)]
        nanoctrl_scope: Option<String>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct SchedulerConfig {
    pub queue_size: usize,
    pub timeout_ms: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct AppConfig {
    pub server: ServerConfig,
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>, // optional; loaded lazily from engine if absent
    pub engine: EngineConfig,
    pub scheduler: SchedulerConfig,
}

impl AppConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let s = Config::builder()
            .add_source(File::from(path.as_ref()))
            .build()?;

        s.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_config() {
        let toml_content = r#"
            [server]
            host = "127.0.0.1"
            port = 8080
            model_name = "TestModel"

            [tokenizer]
            path = "/tmp/tokenizer.json"

            [engine]
            mode = "Unified"
            host = "127.0.0.1"
            port = 5000

            [scheduler]
            queue_size = 100
            timeout_ms = 5000
        "#;

        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("Failed to create temp file");
        write!(file, "{}", toml_content).expect("Failed to write to temp file");

        let config = AppConfig::load_from_file(file.path()).expect("Failed to load config");

        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.model_name, "TestModel");
        match config.engine {
            EngineConfig::Unified { host, port, .. } => {
                assert_eq!(host, "127.0.0.1");
                assert_eq!(port, 5000);
            }
            _ => panic!("Expected Unified config"),
        }
    }

    #[test]
    fn test_load_disaggregated_config() {
        let toml_content = r#"
            [server]
            host = "0.0.0.0"
            port = 3000
            model_name = "Test"

            [tokenizer]
            path = "tok.json"

            [engine]
            mode = "Disaggregated"

            [[engine.prefill]]
            host = "1.1.1.1"
            port = 6000

            [[engine.decode]]
            host = "2.2.2.2"
            port = 7000

            [scheduler]
            queue_size = 100
            timeout_ms = 1000
        "#;

        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("TempFile");
        write!(file, "{}", toml_content).expect("Write");
        let config = AppConfig::load_from_file(file.path()).expect("Load");

        if let EngineConfig::Disaggregated {
            prefill, decode, ..
        } = config.engine
        {
            assert_eq!(prefill.len(), 1);
            assert_eq!(prefill[0].host, "1.1.1.1");
            assert_eq!(decode.len(), 1);
            assert_eq!(decode[0].port, 7000);
        } else {
            panic!("Expected Disaggregated");
        }
    }
}
