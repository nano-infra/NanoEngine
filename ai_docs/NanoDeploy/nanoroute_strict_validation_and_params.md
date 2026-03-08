# NanoRoute API Safety and Parameter Injection Update

This document summarizes the recent updates made to NanoRoute to improve API safety and feature completeness by supporting strict parameter validation and new LLM sampling parameters.

## 1. Strict Request Validation

Previously, NanoRoute would silently ignore unmapped fields in incoming JSON payloads. This could lead to confusing behavior if clients sent misspelled fields (e.g., `max_length` instead of `max_tokens`) or attempted injection.

**Changes:**

- Enabled `#[serde(deny_unknown_fields)]` on the `ChatCompletionRequest` and `Message` structures in `NanoRoute/src/http_server.rs`.
- NanoRoute will now strictly validate the HTTP request body. Any request containing unrecognized properties (e.g., `"random_inject": "true"`) will be immediately rejected.
- On validation failure, Axum automatically returns an HTTP `422 Unprocessable Entity` response, safeguarding the backend engines from malformed requests or unexpected properties.

## 2. Parameter Injection: `temperature` and `ignore_eos`

We expanded the `ChatCompletionRequest` struct to natively parse and route two new LLM sampling parameters: `temperature` and `ignore_eos`.

**Changes:**

- Added `temperature: Option<f32>` and `ignore_eos: Option<bool>` to the request payload mapping.
- Due to the `#[serde(default)]` attribute, these fields remain optional for clients. If omitted, they default to `0.1` and `false` respectively when being passed down to the engine.
- Refactored `EngineAdapter::send_add_request` in `NanoRoute/src/engine_adapter.rs`. Instead of hardcoding the `temperature` and `ignore_eos` values in the FlatBuffer `SamplingParamsArgs` payload, the function now dynamically sets them based on the incoming HTTP request.
- The `temperature` (f32) is cleanly cast to `f64` (`temperature as f64`) to match the FlatBuffers schema definitions before being serialized and dispatched over ZMQ to the NanoDeploy engine instances.

These updates allow developers to confidently tune engine outputs at inference time while retaining robust perimeter security at the NanoRoute load balancing layer.
