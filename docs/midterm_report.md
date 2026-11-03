# Cache-Aware Routing of Agent Workflows on Heterogeneous Accelerators

**11-868 LLM Systems · Project Mid-term Report** · Yanze, Keyi, Hannah · November 2026
Code: https://github.com/clemrho/hetero-agent-routing

## 1. Introduction and Motivation

LLM agents do not issue independent requests. An agent workflow is a chain of LLM calls with a
tool call (search, code execution, retrieval) between consecutive calls. Each call's prompt is
the previous prompt plus the model's output and the tool's observation, so the context grows
step by step and every prompt is a prefix of the next one. Serving engines such as vLLM keep the
KV cache of recent prefixes (prefix caching), so a step that lands on the device that served the
previous step only needs to prefill the new tokens. A step that lands elsewhere recomputes the
whole context, and that cost grows with context length.

Serving fleets are increasingly heterogeneous: different GPU generations, and non-GPU
accelerators such as Google TPUs. With more than one device the router must decide, at every
step, between **keeping a workflow where its cache is** and **moving it where there is spare
capacity**. On homogeneous devices this is already a trade-off; on heterogeneous devices it is
harder, because the cost of a cache miss, the cost of a hit, and the queueing delay all differ per
device and per request length.

Agent workflows also have a property that request-level routers ignore: **the tool gap**. Between
two calls the workflow is idle for the duration of the tool call (hundreds of milliseconds to
seconds). During that time the router knows the next prompt's prefix and can act on it, for
example by building the cache on another device before the next call arrives.

We study this setting on a TPU v5e and an NVIDIA L40S, both running vLLM with prefix caching.
**Expected contributions:**

1. A testbed for routing agent workflows across a TPU and a GPU: a client-side router with
   per-call timing, a trace generator and replay client, and the official vLLM Production Stack
   router as an external baseline. All of it is open source and scripted end to end.
2. A characterization of the cache-vs-load trade-off for agent workflows on heterogeneous
   devices, with device latency profiles (cold vs. warm prefill across context lengths) and a
   comparison of standard routing policies.
3. **GapShift**, a routing policy that uses tool gaps to re-place a workflow's context on the
   device where its next call is cheapest, using measured device profiles instead of fixed
   "fast/slow" labels.

## 2. Related Work and Background

**Prefix caching within one engine.** vLLM's PagedAttention [Kwon et al., SOSP'23] stores the KV
cache in blocks and supports automatic prefix caching; SGLang's RadixAttention [Zheng et al.,
NeurIPS'24] keeps prefixes in a radix tree and schedules requests to maximize reuse. Both
optimize reuse inside a single engine; we take them as given and study placement across engines.

**Prefix-aware load balancing across replicas.** The closest work is Preble [Srivatsa et al.,
ICLR'25], which distributes prompts across GPUs by trading prefix reuse against load. In
industry, the vLLM Production Stack router offers session-affinity, prefix-aware and
KV-cache-aware routing. Both target **homogeneous** GPU replicas and route **individual
requests**. We differ in three ways: devices with different latency profiles (TPU + GPU),
workflow-level structure (a known growing prefix), and the tool gap as a window for acting
before the next request arrives. We use the Production Stack router (session mode) directly as
a baseline.

**Serving agentic and tool-augmented workloads.** InferCept [Abhyankar et al., ICML'24] decides
what to do with a request's KV cache while it waits for a tool call (keep, discard and recompute,
or swap) within one engine. Autellix [Luo et al., 2025] schedules agentic programs as a whole and
adds locality-aware load balancing across engines. Like us they exploit workflow structure, but
on homogeneous GPUs, and they do not use the gap to move a context to another device type.

**Heterogeneous LLM serving.** Mélange [Griggs et al., 2024] picks a cost-efficient mix of GPU
types by request size; Helix [Mei et al., ASPLOS'25] places models and schedules requests on
heterogeneous GPU clusters. Both treat requests as independent and focus on GPUs; our setting
adds prefix-cache state that depends on past routing decisions, and a TPU.

## 3. Methodology and Implementation

### 3.1 Problem setup

A workflow is a sequence of steps *k = 1..K*. Step *k* sends prompt *p_k* (with *p_k* a prefix
of *p_{k+1}*), generates *o_k* tokens, then waits *g_k* seconds for a tool. The router picks a
device *d_k* for each step. If *d_k = d_{k-1}* and the cache was not evicted, only the new
tokens are prefilled; otherwise all of *p_k* is. Per call, TTFT ≈ queueing + prefill of
uncached tokens on *d_k*; both terms depend on the device. Metrics: per-workflow completion
time, makespan of the whole trace, TTFT p50/p99, and prefix-cache hit rate (cached tokens over
prompt tokens), reported against its upper bound (every step reuses the whole previous prompt).

### 3.2 Research questions

- **RQ1.** How large is the cache-vs-load trade-off for agent workflows on a TPU + GPU pair, as a
  function of context length and concurrency?
- **RQ2.** How do the two devices' latency profiles differ (cold vs. warm prefill, context length,
  decode), and does a fixed "fast device / slow device" label hold?
- **RQ3.** Can a policy that acts during tool gaps (GapShift) beat both affinity-only and
  load-only policies?

RQ1 is answered by sweeping baselines over traces with growing contexts and concurrency; RQ2 by
unloaded profiling of both devices; RQ3 by implementing GapShift and comparing it on the same
sweeps, with ablations over gap length and context length.

### 3.3 System

- **Router** (`router/router.py`): a Python object in front of the vLLM servers' OpenAI-compatible
  API. It streams each call and records arrival, send, first-token and done timestamps, prompt
  tokens and `cached_tokens` (from `prompt_tokens_details`). Timing is done on the client for
  every policy, including the external one, so all policies are measured the same way.
- **Workloads** (`traces/`, `replay/`): a trace stores, per step, prompt length, output length and
  tool time. The replay client rebuilds prompts as token-id sequences from a per-workflow,
  per-run random seed, so every step is an exact prefix of the next, both devices see identical
  input, and runs cannot reuse each other's cache. Replay is closed-loop at a fixed concurrency.
  Synthetic traces are generated now; HotpotQA ReAct traces are in progress.
- **Baselines:** LB (fewest in-flight requests), FF (prefer one device until it has 4 in flight,
  then spill), CA (first step by LB, then sticky), RR (round robin), and PS, the official vLLM
  Production Stack router in session mode (hashes the workflow id to a backend), run standalone
  with static backends.
- **Deployment:** vLLM-TPU on a Colab TPU v5e (1 chip, 16 GB HBM); vLLM on a PSC Bridges-2 L40S
  (48 GB). Both serve Qwen3-4B in bfloat16 with prefix caching and `max_model_len` 16384. The
  Colab runtime only allows outgoing connections, so the router runs next to the TPU and reaches
  the GPU through an SSH tunnel via the PSC login node. All of it is scripted: one setup script
  and one batch job on PSC, one notebook on Colab.
- **Sanity checks:** a backend check sends the same long prefix twice and requires
  `cached_tokens > 0` on the second request; the analysis flags any run whose hit rate exceeds
  its upper bound. This caught an early bug where back-to-back runs reused each other's cache.

Getting a working stack took several fixes worth noting for others: current vLLM wheels are
built with CUDA 13, which has no V100 kernels; with an older vLLM, V100 starts but its
prefix-prefill Triton kernel fails to compile, so we moved to L40S. On Colab, vllm-tpu has no
wheel for the default Python 3.13 and runs from a Python 3.12 environment.

### 3.4 GapShift (planned design)

At the start of the tool gap after step *k*, the workflow sits on device *h* and the router knows
*p_k*. For each device *d* it estimates the next call's cost as expected queueing on *d* plus the
prefill time on *d* for the tokens *d* does not have cached, using the measured profile from RQ2.
If another device is cheaper by a margin and the gap is long enough, GapShift sends a pre-warm
request (*p_k*, one output token) to that device during the gap, so its cache already holds the
context when the next step arrives and only the new tokens need prefilling. Otherwise the
workflow stays. A second candidate signal is the TPU's length bucketing (padding to fixed shapes),
which makes TPU latency step-shaped in context length; we will check this in the profiles.

## 4. Experiments

### 4.1 Pipeline validation

On a single PSC L40S running two vLLM instances, all policies ran end to end with real vLLM and
the official router, and session-mode PS and CA kept every workflow on one backend. This run
also exposed the cross-run cache reuse bug described in Section 3.3 (hit rates above the upper
bound), which we fixed before the cross-device runs below.

### 4.2 First cross-device baseline (TPU v5e + L40S)

Trace: 12 synthetic workflows, 3 or 5 steps, starting context 2k or 8k tokens and growing to
10.5k, 20–60 output tokens per step, 0.2–0.8 s tool waits (48 LLM calls). Qwen3-4B, bfloat16.
Closed-loop replay at concurrency 4 and 8, one repetition. Full setup and raw output:
`results/2026-11-03_tpu-v5e_l40s_quick12/`.

**Device check** (3k-token prompt, one output token, unloaded):

| | TPU v5e | L40S (tunnel) | L40S minus RTT |
|---|---|---|---|
| Cold prefill | 216 ms | 469 ms | ≈ 270 ms |
| Warm (prefix hit) | 74 ms | 227 ms | ≈ 28 ms |
| Network RTT | 0.6 ms | 199 ms | |

**Baselines at concurrency 8** (cache-hit upper bound 71.9%):

| Policy | Makespan | TTFT p50 | TTFT p99 | Cache hit | Workflows that switched device |
|---|---|---|---|---|---|
| CA | **12.4 s** | **0.18 s** | 1.86 s | **71.8%** | 0 / 12 |
| LB | 12.7 s | 0.27 s | 1.86 s | 53.3% | 9 / 12 |
| FF | 13.4 s | 0.27 s | 1.42 s | 54.7% | 9 / 12 |
| RR | 14.4 s | 0.34 s | 2.00 s | 48.7% | 10 / 12 |

At concurrency 4 the ranking is the same (CA makespan 18.7 s, TTFT p50 0.17 s; RR 19.4 s, 0.23 s),
and FF keeps all 48 calls on the GPU.

**Observations.**
1. *Affinity matters.* CA reaches the cache-hit bound and roughly halves median TTFT relative to
   RR. Load-only policies lose about 20 points of hit rate because workflows switch devices and
   re-prefill their context (RQ1).
2. *Spilling costs cache.* FF keeps the cache while one device absorbs the load (concurrency 4),
   but once it passes its threshold (concurrency 8) it splits 9 of 12 workflows across devices
   and its hit rate drops to the LB/RR level.
3. *No fixed fast/slow order* (RQ2). Cold prefill is about equal on the two devices (v5e slightly
   faster), while on a cache hit the L40S is about 2.6× faster. The better device depends on the
   request, which is the case GapShift targets.
4. *Makespan gaps are small* (12.4–14.4 s) because tool waits dominate at this load; TTFT and hit
   rate already separate the policies.

**Limitations of these results.** One repetition per setting (p99 is close to the single slowest
call). The GPU path includes a 199 ms network round trip from Colab to PSC, which penalizes the
GPU for placement reasons, not hardware; it also explains why FF at concurrency 4 (all GPU) has a
higher median TTFT than CA. The load is light, and PS was not yet run in the cross-device setup.

## 5. Remaining Work and Timeline

| Dates | Work |
|---|---|
| 11/5–11/9 | Remove the network confound: run the TPU in a GCP zone near PSC (or the course cluster) and report RTT-corrected TTFT. Add PS to the cross-device runs. Device profiles on both devices: cold/warm TTFT for 1k–16k tokens and decode speed (Keyi). |
| 11/10–11/16 | HotpotQA ReAct traces (Hannah, 11/10). Larger sweep: contexts up to 16k, concurrency {4, 8, 16}, 2–3 repetitions; figures for workflow completion time and latency breakdown. Check TPU memory limits at long contexts (16 GB HBM). |
| 11/17–11/26 | Implement GapShift (cost model from the profiles, pre-warm during gaps); ablations over gap length, context length and concurrency. |
| 11/27–12/4 | Final experiments, report and presentation (12/4). |

Roles: Yanze, system and experiments (vLLM on both devices, connection, router and baselines,
sweeps, GapShift); Keyi, device profiling and figures; Hannah, workloads and the replay client.

## References

- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024.
- V. Srivatsa et al. *Preble: Efficient Distributed Prompt Scheduling for LLM Serving.* ICLR 2025.
- R. Abhyankar et al. *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference.* ICML 2024.
- M. Luo et al. *Autellix: An Efficient Serving Engine for LLM Agents as General Programs.* arXiv 2025.
- T. Griggs et al. *Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity.* arXiv 2024.
- Y. Mei et al. *Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow.* ASPLOS 2025.
- vLLM Production Stack. https://github.com/vllm-project/production-stack
