# Supported Models

DLEngine is designed primarily for large-model inference on multi-GPU and multi-node clusters. The canonical support boundary is the lazy architecture registry in [`dlengine/runtime/models/registry.py`](https://github.com/JimyMa/NanoDeploy/blob/Pure_dp/dlengine/runtime/models/registry.py); model names are less reliable than the `architectures` value in the checkpoint configuration.

## Text model families

| Model family                   | Checkpoint architecture                                 | Core design                      | Notable DLEngine paths                                    |
| ------------------------------ | ------------------------------------------------------- | -------------------------------- | --------------------------------------------------------- |
| Qwen3 dense                    | `Qwen3ForCausalLM`                                      | GQA + dense FFN                  | TP/DP execution, paged KV cache, chunked prefill          |
| Qwen3 MoE                      | `Qwen3MoeForCausalLM`                                   | GQA + MoE                        | Wide EP, packed FP8 weights, distributed serving          |
| Qwen3.5 dense                  | `Qwen3_5ForConditionalGeneration`                       | Full attention + GDN + dense FFN | Mixed-attention KV/GDN state management                   |
| Qwen3.5 MoE                    | `Qwen3_5MoeForConditionalGeneration`                    | Full attention + GDN + MoE       | Mixed-attention state, wide EP, optional model-native MTP |
| DeepSeek-V3                    | `DeepseekV3ForCausalLM`                                 | MLA + MoE                        | Wide EP, FP8 KV/cache paths, optional MTP                 |
| DeepSeek-V3.2                  | `DeepseekV32ForCausalLM`                                | MLA + MoE + NSA                  | Indexer cache, sparse attention, HiSparse decode          |
| DeepSeek-V4                    | `DeepseekV4ForCausalLM`                                 | MLA/DSA/SWA + MoE                | Compressed cache, Hyper-Connection, mega-MoE paths        |
| GLM-5 family                   | `GlmMoeDsaForCausalLM`                                  | MLA + MoE + DSA/NSA              | GLM Indexer, long context, recurrent N=5/K=6 MTP          |
| Gemma4 text                    | `Gemma4ForCausalLM` or `Gemma4ForConditionalGeneration` | Sliding-window + global GQA      | HiSparse ring buffer and graph-safe decode                |
| Kimi-K2 compatible checkpoints | compatible `DeepseekV3ForCausalLM` config               | MLA + MoE                        | Uses the DeepSeek-V3-compatible model path                |

A checkpoint is supported only when its architecture and tensor layout match the corresponding loader. A marketing model name alone does not guarantee compatibility.

## Vision-language path

`dlengine.vl` provides the vision-encoder path used by Qwen3.5-MoE vision-language checkpoints. The language backbone still resolves through `Qwen3_5MoeForConditionalGeneration`, while the vision component produces embeddings for the prefill engine.

Vision-language deployments may require a separate encoder service and have a different topology from the text-only quick start.

## Large-model inference capabilities

The exact combination depends on the model architecture, but DLEngine is built around:

- Ray-managed multi-node GPU resource allocation and worker placement.
- Attention data parallelism and wide expert parallelism.
- Experimental pipeline parallelism for bounded-microbatch prefill; decode remains single-stage.
- Prefill/decode disaggregation with GPUDirect RDMA KV migration.
- FP8 weights and paged FP8 KV/Indexer caches where supported.
- Chunked prefill, prefix caching, and long-context execution.
- MLA, NSA/DSA, GDN, sliding-window attention, and model-native MTP paths.
- OpenAI- and Anthropic-compatible serving through `dlengine-router`.

## Architecture-specific constraints

- DeepSeek/GLM MLA paths require `attention_tp=1`; their KV cache block size is normalized to `64`.
- HiSparse NSA/MLA targets decode mode for `DeepseekV32ForCausalLM` and `GlmMoeDsaForCausalLM` and requires a control plane for real PD migration. GLM can compose it with linear MTP; other HiSparse architectures still reject MTP.
- GLM multi-step MTP targets Hopper with `attention_tp=attention_sp=1`. Colocated hybrid serving uses `pp=1`; PD serving supports a PP prefill engine paired with a `pp=1` decode engine. Set `num_speculative_tokens=5` on both roles for five recurrent calls to the single checkpoint predictor and a derived six-token target verification span. The predictor runs only on the final prefill PP stage; its KV and first five-token draft line migrate through the data plane. Keep each prefill forward bounded with `--max_num_batched_tokens 8192`; `--pp_prefill_scheduler_depth 0` automatically admits enough microbatches to fill the pipeline and, for long prompts, up to 64 consecutive microbatches. On the decode role, `--enable_hisparse true --hisparse_device_buffer_size 12288` provides the minimum hot capacity for six `index_topk=2048` rows. Tree speculation and PP decode are not enabled.
- Gemma4 mixed sliding/global attention requires the HiSparse hot-buffer path when the two attention types use different head dimensions.
- Gemma4 MoE blocks and some TP combinations are not wired yet; read runtime validation errors before assuming every Gemma4 checkpoint variant is supported.
- Qwen3.5 checkpoints can contain text, vision, and MTP tensors. The selected server/component determines which parts are loaded.
- Kernel availability can depend on the CUDA, FlashMLA, FlashInfer, DeepGEMM, and flash-attn versions in the development image.

## Checking a checkpoint

Inspect `config.json` before deployment:

```bash
python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("architectures"), c.get("model_type"))' \
  /path/to/model/config.json
```

Then compare the architecture with [`dlengine/runtime/models/registry.py`](https://github.com/JimyMa/NanoDeploy/blob/Pure_dp/dlengine/runtime/models/registry.py). DLEngine also validates unsupported parallel, cache, HiSparse, and MTP combinations during `Config` construction and should fail before worker execution.

## Documentation maintenance

When adding a model family:

1. Add or reuse a model implementation and weight loader.
2. Register the Hugging Face architecture string in `dlengine/runtime/models/registry.py`.
3. Add focused loader and inference tests.
4. Record supported parallel/cache features and known constraints here.
5. Add a serving or offline validation command using a representative checkpoint.
