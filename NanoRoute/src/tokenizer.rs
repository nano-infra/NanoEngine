use minijinja::Environment;
use serde::Serialize;

use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

pub struct TokenizerService {
    path: String,
    tokenizer: Arc<Option<Tokenizer>>,
    // Store compiled environment in Arc for thread safety
    // We use 'static because we will use add_template_owned (if available) or source
    // Actually minijinja::Environment holds templates.
    template_env: Arc<Option<Environment<'static>>>,
}

impl TokenizerService {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            tokenizer: Arc::new(None),
            template_env: Arc::new(None),
        }
    }

    pub async fn load(&mut self) -> anyhow::Result<()> {
        let path_str = self.path.clone();
        let _path = Path::new(&path_str);

        info!("Loading tokenizer from {}", path_str);

        let path_clone = path_str.clone();
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

        info!("Tokenizer service ready with simplified ChatML template.");
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
