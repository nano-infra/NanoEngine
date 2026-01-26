use serde::Deserialize;
use config::{Config, ConfigError, File};
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    pub engine_host: Option<String>,
    pub engine_port: Option<u16>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct TokenizerConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct EngineGroupConfig {
    pub count: usize,
    pub config_path: String,
    pub tensor_parallel: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
#[serde(tag = "mode")]
pub enum EngineConfig {
    Unified {
        #[serde(flatten)]
        config: EngineGroupConfig,
    },
    Disaggregated {
        prefill: EngineGroupConfig,
        decode: EngineGroupConfig,
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
    pub tokenizer: TokenizerConfig,
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
    use tempfile::NamedTempFile;

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
            count = 1
            config_path = "/tmp/engine_config.py"
            tensor_parallel = 1

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
            EngineConfig::Unified { config } => {
                assert_eq!(config.count, 1);
            }
            _ => panic!("Expected Unified config"),
        }
    }
}
