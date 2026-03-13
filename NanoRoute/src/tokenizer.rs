use minijinja::Environment;
use serde::Serialize;

use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

pub struct TokenizerService {
    /// Model directory used for `req.model` comparison (no trailing slash).
    /// e.g. `/models/Qwen3-235B-A22B-Instruct-2507`
    model_dir: String,
    /// Actual tokenizer file passed to `Tokenizer::from_file`.
    /// e.g. `/models/Qwen3-235B-A22B-Instruct-2507/tokenizer.json`
    tokenizer_file: String,
    tokenizer: Arc<Option<Tokenizer>>,
    template_env: Arc<Option<Environment<'static>>>,
}

impl TokenizerService {
    /// Probe a model directory for a loadable tokenizer file.
    ///
    /// Tries common names in order; falls back to `tokenizer.json` if none found
    /// (the subsequent `load()` will then surface a clear "file not found" error).
    fn find_tokenizer_file(model_dir: &str) -> String {
        const CANDIDATES: &[&str] = &["tokenizer.json", "tokenizer.model"];
        for name in CANDIDATES {
            let candidate = format!("{}/{}", model_dir, name);
            if Path::new(&candidate).exists() {
                return candidate;
            }
        }
        format!("{}/tokenizer.json", model_dir)
    }

    /// Accept either a model directory or an explicit tokenizer file path.
    ///
    /// Uses `fs::metadata` to distinguish the two cases reliably — avoids
    /// false-positives from dots in directory names like `Qwen3.5-35B-A3B`.
    ///
    /// - Directory: probes for `tokenizer.json` / `tokenizer.model` inside it.
    /// - File: uses as-is; `model_dir` is set to the parent directory.
    pub fn new(path: &str) -> Self {
        let trimmed = path.trim_end_matches('/');
        let (model_dir, tokenizer_file) = match std::fs::metadata(trimmed) {
            Ok(meta) if meta.is_dir() => {
                let dir = trimmed.to_string();
                let file = Self::find_tokenizer_file(&dir);
                (dir, file)
            }
            _ => {
                // Explicit file path (or path not yet on disk — treat as file).
                let dir = Path::new(trimmed)
                    .parent()
                    .map(|d| d.to_string_lossy().trim_end_matches('/').to_string())
                    .unwrap_or_else(|| trimmed.to_string());
                (dir, trimmed.to_string())
            }
        };
        Self {
            model_dir,
            tokenizer_file,
            tokenizer: Arc::new(None),
            template_env: Arc::new(None),
        }
    }

    /// Spawn a background task to load a tokenizer from `path` into `slot`.
    ///
    /// No-op if the slot is already populated (fast-path read lock check,
    /// then double-checked under the write lock before writing).
    pub fn spawn_load(slot: Arc<RwLock<Option<Arc<TokenizerService>>>>, path: String) {
        tokio::spawn(async move {
            // Fast path: already loaded — avoid spawning unnecessary work.
            if slot.read().await.is_some() {
                return;
            }
            let mut svc = TokenizerService::new(&path);
            match svc.load().await {
                Ok(()) => {
                    let mut w = slot.write().await;
                    if w.is_none() {
                        // Double-check under write lock to handle concurrent loaders.
                        *w = Some(Arc::new(svc));
                        info!("Tokenizer loaded (model_dir: {})", path);
                    }
                }
                Err(e) => error!("Failed to load tokenizer from {}: {}", path, e),
            }
        });
    }

    pub async fn load(&mut self) -> anyhow::Result<()> {
        let file = self.tokenizer_file.clone();

        debug!(
            "Loading tokenizer from {} (model_dir: {})",
            file, self.model_dir
        );

        let path_clone = file.clone();
        let tokenizer = tokio::task::spawn_blocking(move || {
            Tokenizer::from_file(&path_clone).map_err(|e| anyhow::anyhow!(e))
        })
        .await??;

        self.tokenizer = Arc::new(Some(tokenizer));

        // Initialize Jinja environment with hardcoded ChatML template for stability
        // The original Qwen template uses complex logic (namespace, tools) incompatible with minijinja 1.0 default settings.
        let mut env = Environment::new();
        let simple_template = r#"
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;
        env.add_template("chat", simple_template)?;
        self.template_env = Arc::new(Some(env));

        /*
        // Original loading logic disabled due to minijinja compatibility issues
        let config_path = path.parent().unwrap().join("tokenizer_config.json");
        // ... (removed)
        */

        debug!("Tokenizer service ready with simplified ChatML template.");
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn encode(&self, text: String) -> anyhow::Result<Vec<u32>> {
        let t = self.tokenizer.clone();
        if let Some(tokenizer) = t.as_ref() {
            let tokenizer_ref = tokenizer.clone();
            let encoding = tokio::task::spawn_blocking(move || {
                tokenizer_ref
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .await??;

            Ok(encoding.get_ids().to_vec())
        } else {
            Err(anyhow::anyhow!("Tokenizer not loaded"))
        }
    }

    pub async fn encode_messages<T: Serialize + Send + Sync + 'static>(
        &self,
        messages: T,
    ) -> anyhow::Result<Vec<u32>> {
        // Render template first
        let formatted_text = if let Some(env) = self.template_env.as_ref() {
            let tmpl = env.get_template("chat").map_err(|e| anyhow::anyhow!(e))?;
            // We need to pass arguments: messages, add_generation_prompt
            let ctx = serde_json::json!({
                "messages": messages,
                "add_generation_prompt": true,
                // Add common special tokens just in case template needs them
                "bos_token": "<|im_start|>",
                "eos_token": "<|im_end|>",
                "tools": [], // Default empty tools
            });
            tmpl.render(ctx)
                .map_err(|e| anyhow::anyhow!("Template render error: {}", e))?
        } else {
            // Fallback: Just join simple content (not right for Qwen but fail-safe)
            // Or error?
            return Err(anyhow::anyhow!("Chat template not loaded"));
        };

        // info!("Formatted chat: {}", formatted_text);

        let t = self.tokenizer.clone();
        if let Some(tokenizer) = t.as_ref() {
            let tokenizer_ref = tokenizer.clone();
            let encoding = tokio::task::spawn_blocking(move || {
                tokenizer_ref
                    .encode(formatted_text, true)
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .await??;

            Ok(encoding.get_ids().to_vec())
        } else {
            Err(anyhow::anyhow!("Tokenizer not loaded"))
        }
    }

    pub async fn decode(&self, ids: Vec<u32>) -> anyhow::Result<String> {
        let t = self.tokenizer.clone();
        if let Some(tokenizer) = t.as_ref() {
            let tokenizer_ref = tokenizer.clone();
            let decoded = tokio::task::spawn_blocking(move || {
                tokenizer_ref
                    .decode(&ids, true)
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .await??;
            Ok(decoded)
        } else {
            Err(anyhow::anyhow!("Tokenizer not loaded"))
        }
    }
}
