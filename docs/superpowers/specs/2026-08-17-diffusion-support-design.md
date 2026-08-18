# vLLM-Omni Diffusion Model Support in llm-d

**Author:** Kapil Jain  
**Date:** 2026-08-17  
**Status:** Design (Ready for Review)  
**Scope:** MVP - Single diffusion model server support with request-type filtering

---

## Table of Contents

1. [Overview](#overview)
2. [Goals](#goals)
3. [Architecture](#architecture)
4. [Components](#components)
5. [Request Lifecycle](#request-lifecycle)
6. [Data Flow](#data-flow)
7. [Configuration](#configuration)
8. [Testing Strategy](#testing-strategy)
9. [Future Extensions](#future-extensions)
10. [Open Questions](#open-questions)

---

## Overview

This design adds **vLLM-Omni diffusion model support** to llm-d's routing infrastructure. The MVP enables:

- Routing diffusion requests (`/v1/images/generations`) to dedicated diffusion pods
- Keeping LLM and diffusion workloads in the same `InferencePool` via request-type filtering
- Passthrough execution via sidecar (no disaggregation)
- Automatic load balancing across diffusion pods

**Key Constraint:** This MVP targets **diffusion-only deployments** (dedicated pods for diffusion, separate pods for LLMs). Future work will support hybrid pods and model-specific optimizations.

---

## Goals

1. **Enable diffusion serving** alongside LLM inference in llm-d
2. **Maintain separation** of LLM and diffusion workloads (request-type filters prevent misrouting)
3. **Leverage existing infrastructure** (reuse scheduling profiles, data producers, plugins)
4. **Keep sidecar changes minimal** (passthrough for diffusion, NIXL for LLM)
5. **Establish foundation** for future optimizations (caching, disaggregation, cache-aware routing)

---

## Architecture

### High-Level Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ User Request                                                    │
│ • /v1/images/generations (diffusion)                           │
│ • /v1/chat/completions (LLM)                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Envoy Proxy                                                     │
│ (forwards to EPP)                                               │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ EPP (Endpoint Picker Plugin)                                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Request Parsing                                              │
│    • diffusion-images-parser: recognizes /v1/images/generations│
│    • extracts modality: ModalityDiffusion or ModalityText      │
│                                                                 │
│ 2. Request Control & Screeners                                 │
│    • Standard admission, flow control                           │
│                                                                 │
│ 3. Scheduling Profile Selection                                │
│    • Selects appropriate profile (e.g., "diffusion", "default")│
│                                                                 │
│ 4. Filtering Stage                                              │
│    • model-type-filter: removes incompatible pods              │
│      - diffusion requests: skip pods with model-type=llm       │
│      - LLM requests: skip pods with model-type=diffusion       │
│                                                                 │
│ 5. Scoring Stage                                                │
│    • Basic load scorer (inflight request count per pod)        │
│    • Optional: Cache-DiT capability scorer (future)            │
│                                                                 │
│ 6. Pod Selection                                                │
│    • max-score-picker selects highest-scored pod               │
│                                                                 │
│ 7. Response Headers                                             │
│    • Injects x-model-server-port → target pod address          │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Sidecar (on target pod)                                         │
├─────────────────────────────────────────────────────────────────┤
│ detectRequestType(path):                                        │
│   if path == /v1/images/generations:                           │
│     → routeDirectly(vLLMOmniEndpoint, request)                │
│   else:                                                         │
│     → handleLLMRequest(NIXL coordination)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ vLLM-Omni                                                       │
│ • LLM Engine (if LLM pod)                                       │
│ • Diffusion Engine (if diffusion pod)                           │
│ • API Endpoint (/v1/chat/completions or /v1/images/generations)│
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Response                                                        │
│ • Completion tokens + usage (LLM)                              │
│ • Base64-encoded PNG (diffusion)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Model

**InferencePool Structure:**
- **Single `InferencePool`** containing both LLM and diffusion pods
- **Pod Labels** distinguish model type:
  - `llm-d.ai/model-type=llm` (LLM pods)
  - `llm-d.ai/model-type=diffusion` (diffusion pods)
- **Request-type filtering** ensures no cross-contamination

**Pod Isolation:**
- Each pod maintains independent GPU/VRAM allocation
- No VRAM sharing between pods (vLLM-Omni design constraint)
- Diffusion pods run vLLM-Omni with only diffusion models loaded
- LLM pods run vLLM-Omni with only LLM models loaded

---

## Components

### 1. New Parser: `diffusion-images-parser`

**File:** `pkg/epp/handlers/diffusion_parser.go` (new)

**Responsibility:** Recognize and parse `/v1/images/generations` requests

**Implementation Details:**
- Implements `fwkrh.Parser` interface
- Registers path: `/v1/images/generations`
- Extracts fields from request body:
  - `prompt` (string, required)
  - `num_inference_steps` (int, optional, default ~20)
  - `guidance_scale` (float, optional, default ~7.5)
  - `seed` (int, optional)
  - `size` (string, e.g., "1024x1024", optional)
  - `n` (int, optional, number of images)
  - `response_format` (string, optional, default "b64_json")
- Creates `InferenceRequestBody` with:
  - `Modality = ModalityDiffusion` (new constant)
  - `APIType = APITypeDiffusion` (new constant)
  - Parsed fields stored in opaque JSON or structured fields
- Delegates fallback to `openai-parser` for unrecognized paths

**Constraints:**
- Does not validate image size constraints (vLLM-Omni responsibility)
- Does not apply guidance_scale bounds checking (model-specific)

**Validation:**
- `prompt` must be non-empty
- `num_inference_steps` must be > 0 if provided
- `seed` is accepted as-is (no bounds checking)

---

### 2. New Filter: `model-type-filter`

**File:** `pkg/epp/framework/plugins/scheduling/filter/modeltype/filter.go` (new)

**Responsibility:** Remove pods whose model type doesn't match request type

**Implementation Details:**
- Implements `scheduling.Filter` interface
- Reads from endpoint metadata:
  - Pod labels via endpoint descriptor (Kubernetes labels on InferencePool pods)
  - Label key: `llm-d.ai/model-type`
  - Values: `llm` | `diffusion`
- Filtering logic:
  ```
  for each pod:
    if request.Modality == ModalityDiffusion:
      keep if pod label == "diffusion"
    else (ModalityText or other):
      keep if pod label == "llm" OR pod label absent
  ```
- Pods without the label default to `llm` (backward compatible)
- Returns filtered pod list

**Configuration:**
```yaml
plugins:
- name: model-type-filter
  type: model-type-filter
  parameters: {}  # no parameters needed
```

**Behavior:**
- **Hard filter** (not soft scorer) to prevent accidental misrouting
- Logs skipped pods at DEBUG level
- Returns empty list if all pods filtered (error path, should not happen in normal operation)

---

### 3. Basic Load Scorer (Existing + Minor Enhancement)

**File:** `pkg/epp/framework/plugins/scheduling/scorer/inflight/scorer.go` (existing)

**Enhancement:**
- Works for both LLM and diffusion requests
- Tracks inflight requests per pod (model-agnostic)
- Scores pods with lower inflight count higher (for both LLM and diffusion)
- No model-specific tweaking needed for MVP

**No new scoring required for MVP** since both LLM and diffusion benefit from low load.

---

### 4. Sidecar Route Handler

**File:** `pkg/sidecar/proxy/proxy.go` (modifications)

**Changes:**
1. **Add `APITypeDiffusion` constant** (alongside `APITypeChatCompletions`, `APITypeGenerate`)
   ```go
   const (
       APITypeChatCompletions APIType = iota
       APITypeResponses
       APITypeGenerate
       APITypeDiffusion  // new
   )
   ```

2. **Update `detectAPIType()` or equivalent**
   ```go
   func detectRequestType(path string) APIType {
       switch {
       case strings.Contains(path, "/v1/chat/completions"):
           return APITypeChatCompletions
       case strings.Contains(path, "/v1/completions"):
           return APITypeChatCompletions
       case strings.Contains(path, "/inference/v1/generate"):
           return APITypeGenerate
       case strings.Contains(path, "/v1/images/generations"):
           return APITypeDiffusion  // new
       default:
           return APITypeUnknown
       }
   }
   ```

3. **Add diffusion routing in `ServeHTTP()` or main handler**
   ```go
   func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
       apiType := detectRequestType(r.URL.Path)
       
       switch apiType {
       case APITypeDiffusion:
           s.handleDiffusion(w, r)
       case APITypeChatCompletions, APITypeGenerate:
           s.handleLLM(w, r)  // existing logic
       default:
           // error
       }
   }
   ```

4. **Implement `handleDiffusion()`**
   ```go
   func (s *Server) handleDiffusion(w http.ResponseWriter, r *http.Request) {
       // 1. Read request body
       original, diffRequest, ok := s.readJSONBody(r, w)
       if !ok {
           return
       }
       
       // 2. Forward directly to vLLM-Omni /v1/images/generations endpoint
       //    (no disaggregation, no KV coordination)
       diffEndpoint := fmt.Sprintf("http://localhost:%d", s.config.ModelServerPort)
       proxyReq := r.Clone(r.Context())
       proxyReq.URL.Scheme = "http"
       proxyReq.URL.Host = diffEndpoint
       proxyReq.URL.Path = r.URL.Path  // preserve /v1/images/generations
       proxyReq.Body = io.NopCloser(bytes.NewReader(original))
       proxyReq.ContentLength = int64(len(original))
       
       // 3. Execute proxy request
       resp, err := s.httpClient.Do(proxyReq)
       if err != nil {
           errorInternalServerError(err, w)
           return
       }
       defer resp.Body.Close()
       
       // 4. Stream response back to client
       copyResponse(w, resp)
   }
   ```

**Key Points:**
- **Passthrough behavior:** diffusion requests go directly to vLLM-Omni, no wrapping
- **No state coordination:** no KV transfer params, no prefill/decode splitting
- **Reuse HTTP utilities:** leverage existing `readJSONBody()`, `copyResponse()` helpers
- **Simple error handling:** standard HTTP error responses

---

### 5. Request Body Extension: Modality Type

**File:** `pkg/epp/framework/interface/requesthandling/types.go` (modifications)

**Add to `InferenceRequestBody`:**
```go
type InferenceRequestBody struct {
    // existing fields...
    
    // Modality indicates the model type: text (LLM) or diffusion
    Modality Modality
}

type Modality string

const (
    ModalityText      Modality = "text"      // LLM, default
    ModalityDiffusion Modality = "diffusion" // diffusion models
    ModalityImage     Modality = "image"     // existing, for multimodal embeddings
)
```

**Usage:**
- Set by diffusion-images-parser to `ModalityDiffusion`
- Set by openai-parser to `ModalityText`
- Read by model-type-filter to determine pod compatibility

---

## Request Lifecycle

### Diffusion Request (MVP)

```
1. Client sends POST /v1/images/generations
   {
     "prompt": "a dragon on mountains",
     "num_inference_steps": 30,
     "seed": 42,
     "size": "1024x1024"
   }

2. Envoy receives, forwards to EPP on x-model-server-port header

3. EPP processing:
   a. Request parsing
      - diffusion-images-parser recognizes /v1/images/generations
      - Creates InferenceRequest with Modality=ModalityDiffusion
      - Stores parsed fields in InferenceRequestBody
   
   b. Request control (standard flow)
      - Flow control admission
      - Screeners (standard)
   
   c. Scheduling profile selection
      - Selects profile (e.g., "default" or custom "diffusion" profile)
   
   d. Filtering stage
      - model-type-filter removes all pods with model-type=llm label
      - Keeps only pods with model-type=diffusion
   
   e. Data producer stage
      - inflight-load-producer counts active requests per pod
   
   f. Scoring stage
      - inflight-load scorer: lower load → higher score
      - Scores diffusion pods only
   
   g. Pod selection
      - max-score-picker selects highest-scoring pod
      - Injects x-model-server-port header with target pod address

4. Sidecar receives request
   - detectRequestType() identifies /v1/images/generations → APITypeDiffusion
   - handleDiffusion() called
   - Proxies request to localhost:/v1/images/generations on vLLM-Omni engine
   - Request execution on diffusion engine (Cache-DiT acceleration if enabled)

5. Response
   - vLLM-Omni returns:
     {
       "created": 1726618400,
       "data": [{"b64_json": "iVBORw0KGgoAAAANS..."}]
     }
   - Sidecar forwards to client via Envoy

6. Result
   - Client receives base64-encoded image
```

### LLM Request (Unchanged)

```
1. Client sends POST /v1/chat/completions
   {
     "model": "Qwen3-7B",
     "messages": [{"role": "user", "content": "Hello"}]
   }

2. EPP processing:
   a. openai-parser recognizes path, sets Modality=ModalityText
   b. model-type-filter removes pods with model-type=diffusion label
   c. Keeps pods with model-type=llm or unlabeled
   d. Scoring proceeds with KV-cache, load, etc.
   e. Pod selected

3. Sidecar receives request
   - detectRequestType() → APITypeChatCompletions
   - handleLLM() called (existing logic)
   - NIXL P/D coordination (prefill/decode disaggregation)

4. Response returned
```

---

## Data Flow

### Per-Request Attribute Flow

```
User Request
  ↓
diffusion-images-parser (creates InferenceRequest)
  ├─ Modality = ModalityDiffusion
  ├─ APIType = APITypeDiffusion
  └─ Fields: prompt, num_inference_steps, seed, size, ...
  ↓
model-type-filter (reads Modality, filters endpoints)
  ├─ Checks endpoint metadata for llm-d.ai/model-type label
  ├─ Keeps diffusion pods only
  └─ Returns filtered pod list
  ↓
inflight-load-producer (reads endpoint state)
  ├─ Counts active requests per pod
  └─ Stores as attribute for scoring
  ↓
inflight-load-scorer (scores filtered pods)
  └─ Higher score = lower load
  ↓
max-score-picker (selects best pod)
  └─ Returns selected endpoint
  ↓
Sidecar (routes request)
```

### Pod Metadata (Static)

Endpoint metadata (from Kubernetes labels):
```yaml
metadata:
  name: diffusion-pod-0
  labels:
    llm-d.ai/model-type: diffusion
    llm-d.ai/model: FLUX-1.0
---
metadata:
  name: llm-pod-0
  labels:
    llm-d.ai/model-type: llm
    llm-d.ai/model: Qwen3-7B
```

---

## Configuration

### Example EPP Config

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig

plugins:
  # Parsers (auto-wired in order)
  - name: openai-parser
    type: openai-parser
  - name: diffusion-images-parser
    type: diffusion-images-parser
  - name: vllmhttp-parser
    type: vllmhttp-parser

  # Filters
  - name: model-type-filter
    type: model-type-filter

  # Data producers
  - name: inflight-load-producer
    type: inflight-load-producer

  # Scorers
  - name: inflight-load-scorer
    type: inflight-load-scorer
    parameters:
      dataProducerName: inflight-load-producer

  # Picker
  - name: max-score-picker
    type: max-score-picker

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: model-type-filter
      - pluginRef: inflight-load-scorer
        weight: 1.0
      - pluginRef: max-score-picker
```

### Kubernetes Pod Labels

**Diffusion Pod:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: diffusion-pod-0
  namespace: inference
  labels:
    llm-d.ai/model-type: diffusion
    llm-d.ai/model: FLUX-1.0
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai-cu121:v0.26.0-omni
    command: ["vllm", "serve", "FLUX-1.0", "--port", "8000"]
```

**LLM Pod:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-pod-0
  namespace: inference
  labels:
    llm-d.ai/model-type: llm
    llm-d.ai/model: Qwen3-7B
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai-cu121:v0.26.0
    command: ["vllm", "serve", "Qwen3-7B", "--port", "8000"]
```

---

## Testing Strategy

### Unit Tests

1. **`diffusion_parser_test.go`**
   - Parses valid `/v1/images/generations` request
   - Extracts prompt, num_inference_steps, seed, size
   - Sets Modality=ModalityDiffusion, APIType=APITypeDiffusion
   - Rejects malformed requests (missing prompt)
   - Falls back to openai-parser for non-diffusion paths

2. **`model_type_filter_test.go`**
   - Filters pods: keeps diffusion pods for diffusion requests
   - Filters pods: keeps LLM pods for LLM requests
   - Defaults unlabeled pods to LLM
   - Returns error if all pods filtered

3. **`proxy_diffusion_test.go`**
   - `handleDiffusion()` reads request body correctly
   - Routes to localhost:model-server-port/v1/images/generations
   - Proxies response back (base64-encoded image)
   - Handles errors (connection refused, timeout)

### Integration Tests

1. **`e2e_diffusion_test.go`**
   - Full flow: diffusion request → EPP → sidecar → vLLM-Omni → image response
   - Verify image response format (b64_json)
   - Verify request headers passed through
   - Verify error handling (invalid prompt, OOM, etc.)

2. **`mixed_workload_test.go`**
   - Simultaneous LLM and diffusion requests
   - Verify LLM requests skip diffusion pods
   - Verify diffusion requests skip LLM pods
   - Verify load balancing across pods of same type

3. **`pod_filtering_test.go`**
   - Create mixed InferencePool with both LLM and diffusion pods
   - Route diffusion requests → only see diffusion pods
   - Route LLM requests → only see LLM pods

---

## Future Extensions

### Phase 2: Cache-Aware Routing

- **Add data producer:** Track Cache-DiT status per diffusion pod
- **Add scorer:** Prefer pods with Cache-DiT enabled (faster requests)
- **Metadata:** Discover cache backend from pod metrics

### Phase 3: Diffusion Disaggregation

- **Encode stage:** Separate text encoder on dedicated pod
- **Diffusion stage:** Separate denoising on another pod
- **Similar to E/P/D architecture** for multimodal LLMs

### Phase 4: Hybrid Pods

- **One pod runs both LLM and diffusion models**
- **Extend model-type-filter to soft scorer** (allow but penalize mismatch)
- **Implement request queuing** to prevent starvation

### Phase 5: Cross-Request Caching

- **Prompt deduplication:** Cache final images for identical prompts
- **Requires separate image cache layer** (not in vLLM-Omni)
- **TTL-based eviction** (semantic similarity matching optional)

---

## Open Questions

1. **Pod Label Strategy:**
   - Should we use a single `llm-d.ai/model-type` label or separate labels for each?
   - Should pods be auto-labeled based on model discovery?
   - Answer: Start with `llm-d.ai/model-type` label; auto-labeling in Phase 2.

2. **Error Handling in `handleDiffusion()`:**
   - Should we differentiate between vLLM-Omni errors and network errors?
   - Should we log request metadata (prompt length, seed) for debugging?
   - Answer: Standard HTTP error responses; add debug logging of metadata.

3. **Request Size Limits:**
   - Should EPP enforce limits on prompt length or output size?
   - Or defer to vLLM-Omni?
   - Answer: Defer to vLLM-Omni for MVP; add validation in Phase 2 if needed.

4. **Metrics & Observability:**
   - What metrics should be exposed for diffusion requests?
   - Should we track per-pod VRAM utilization during inference?
   - Answer: MVP: standard request count, latency, error rate. Phase 2: VRAM tracking.

5. **Load Profile Differences:**
   - Diffusion inference is bursty (fixed iteration count). LLM is variable.
   - Should we use different batching strategies per pod type?
   - Answer: MVP passthrough; Phase 2: pod-specific batching configuration.

---

## Success Criteria

### MVP Completion

- [ ] Parser recognizes `/v1/images/generations` requests
- [ ] Model-type-filter prevents LLM/diffusion cross-routing
- [ ] Sidecar proxies diffusion requests directly to vLLM-Omni
- [ ] Base64-encoded image responses return to client
- [ ] Load balancing works across multiple diffusion pods
- [ ] E2E tests pass (mixed workload, error scenarios)
- [ ] No regression in existing LLM scheduling

### Deployment

- [ ] Updated Helm charts with pod label examples
- [ ] Documentation on configuring mixed InferencePool
- [ ] Troubleshooting guide (common mislabelings, logs)

---

## References

- [vLLM-Omni Architecture Overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/)
- [Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/)
- [vLLM-Omni Diffusion Cache Acceleration](https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration)
- [llm-d Architecture](./docs/architecture.md)
- [Disaggregation Design](./docs/disaggregation.md)