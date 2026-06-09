use anyhow::Context;
use minijinja::{Environment, ErrorKind};
use minijinja_contrib::add_to_environment;
use serde::Serialize;

use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// Rewrite Python method-call syntax used by HuggingFace chat templates (notably Qwen3.5)
/// into equivalent minijinja syntax.
///
/// minijinja-contrib adds Python string methods as filters/tests but cannot dispatch them
/// via Python's `obj.method(args)` syntax. This function fixes the handful of patterns
/// that appear in the Qwen3.5 chat template.
fn normalize_for_minijinja(template: &str) -> String {
    template
        // Boolean predicates → global function calls (registered on the Environment below)
        .replace(
            "content.startswith('<tool_response>')",
            "startswith(content, '<tool_response>')",
        )
        .replace(
            "content.endswith('</tool_response>')",
            "endswith(content, '</tool_response>')",
        )
        // Chained split/strip method calls → minijinja filter pipelines
        // e.g. content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
        .replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "(content | split('</think>') | first | rstrip | split('<think>') | last | lstrip)",
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "(content | split('</think>') | last | lstrip)",
        )
}

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

        // Load chat template: prefer chat_template.jinja (Qwen3.5+ new-style),
        // fall back to the "chat_template" key in tokenizer_config.json,
        // and finally use a minimal ChatML template as last resort.
        let jinja_path = format!("{}/chat_template.jinja", self.model_dir);
        let template_str = if Path::new(&jinja_path).exists() {
            debug!("Loading chat template from {}", jinja_path);
            std::fs::read_to_string(&jinja_path)?
        } else {
            let cfg_path = format!("{}/tokenizer_config.json", self.model_dir);
            if Path::new(&cfg_path).exists() {
                debug!("Loading chat template from tokenizer_config.json");
                let cfg: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
                cfg["chat_template"].as_str().unwrap_or("").to_string()
            } else {
                String::new()
            }
        };

        // Fall back to a minimal ChatML template when nothing was found on disk.
        let template_str = if template_str.is_empty() {
            debug!("No chat template found — using minimal ChatML fallback");
            r#"{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"#
                .to_string()
        } else {
            template_str
        };

        // Rewrite Python-specific method calls that minijinja does not support.
        //
        // minijinja-contrib (pycompat) adds Python string methods as filters/tests but
        // does NOT enable Python method-call dispatch syntax (`obj.method(args)`).
        // The Qwen template uses that syntax in a few places; rewrite them here.
        let template_str = normalize_for_minijinja(&template_str);

        let mut env = Environment::new();

        // Add Python-compatible string methods (startswith, endswith, split,
        // upper, lower, rstrip, lstrip, etc.) needed by HuggingFace chat templates.
        add_to_environment(&mut env);

        // startswith / endswith — also registered as global functions so that
        // normalize_for_minijinja's rewrites (`startswith(content, '...')`) resolve.
        env.add_function("startswith", |s: String, prefix: String| -> bool {
            s.starts_with(prefix.as_str())
        });
        env.add_function("endswith", |s: String, suffix: String| -> bool {
            s.ends_with(suffix.as_str())
        });

        // raise_exception(msg) — called by the Qwen template for input validation.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<(), minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
            },
        );

        env.add_template_owned("chat".to_string(), template_str)?;
        self.template_env = Arc::new(Some(env));

        debug!("Tokenizer service ready (model_dir: {}).", self.model_dir);
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
        tools: Option<serde_json::Value>,
    ) -> anyhow::Result<Vec<u32>> {
        // Render template first
        let formatted_text = if let Some(env) = self.template_env.as_ref() {
            let tmpl = env.get_template("chat")?;
            let ctx = serde_json::json!({
                "messages": messages,
                "add_generation_prompt": true,
                "tools": tools,
                "add_vision_id": false,
                "enable_thinking": serde_json::Value::Null,
            });
            tmpl.render(ctx).context("Template render error")?
        } else {
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
