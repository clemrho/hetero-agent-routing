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
harder, because the cost of a cache miss, the cost of a hit, the queueing delay and the KV-cache
capacity all differ per device.

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

**Prefix caching within one engine.** vLLM (PagedAttention) stores the KV cache in blocks and
supports automatic prefix caching; SGLang (RadixAttention) keeps prefixes in a radix tree and
schedules requests to maximize reuse. Both optimize reuse inside a single engine; we take them as
given and study placement across engines.

**Prefix-aware load balancing across replicas.** The closest work is Preble, which distributes
prompts across GPUs by trading prefix reuse against load. In industry, the vLLM Production Stack
router offers session-affinity, prefix-aware and KV-cache-aware routing. Both target
**homogeneous** GPU replicas and route **individual requests**. We differ in three ways: devices
with different latency profiles and memory sizes (TPU + GPU), workflow-level structure (a known,
growing prefix), and the tool gap as a window for acting before the next request arrives. We use
the Production Stack router (session mode) directly as a baseline.

**Serving agentic and tool-augmented workloads.** InferCept decides what to do with a request's
KV cache while it waits for a tool call (keep, discard and recompute, or swap) within one engine.
Autellix schedules agentic programs as a whole and adds locality-aware load balancing across
engines. Like us they exploit workflow structure, but on homogeneous GPUs, and they do not use the
gap to move a context to another device type.

**Heterogeneous LLM serving.** Mélange picks a cost-efficient mix of GPU types by request size;
Helix places models and schedules requests on heterogeneous GPU clusters. Both treat requests as
independent and focus on GPUs; our setting adds prefix-cache state that depends on past routing
decisions, and a TPU.

## 3. Methodology and Implementation

### 3.1 Problem setup

A workflow is a sequence of steps *k = 1..K*. Step *k* sends prompt *p_k* (with *p_k* a prefix
of *p_{k+1}*), generates *o_k* tokens, then waits *g_k* seconds for a tool. The router picks a
device *d_k* for each step. If *d_k = d_{k-1}* and the cache was not evicted, only the new
tokens are prefilled; otherwise all of *p_k* is. Per call, TTFT ≈ queueing + prefill of
uncached tokens on *d_k*; both terms depend on the device. Metrics: per-workflow completion
time, makespan of the whole trace, TTFT p50/p99, and prefix-cache hit rate (cached tokens over
prompt tokens), reported against its upper bound: every step after the first reuses the whole
previous prompt (the first step of a workflow is always a miss).

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

- **Router** (`router/router.py`): a Python object in front of the vLLM servers'
  OpenAI-compatible API. It streams each call and records arrival, send, first-token and done
  timestamps, prompt tokens and `cached_tokens` (from `prompt_tokens_details`). Timing is done on
  the client for every policy, including the external one, so all policies are measured the same way.
- **Workloads** (`traces/`, `replay/`): a trace stores, per step, prompt length, output length and
  tool time. The replay client rebuilds prompts as token-id sequences from a per-workflow,
  per-run random seed, so every step is an exact prefix of the next, both devices see identical
  input, and runs cannot reuse each other's cache. Replay is **closed-loop**: *C* workflows run at
  once and a new one starts when one finishes. Synthetic traces are generated now; HotpotQA
  ReAct traces are in progress.
- **Deployment:** vLLM-TPU on a Colab TPU v5e (1 chip, 16 GB HBM); vLLM on a PSC Bridges-2 L40S
  (48 GB). Both serve Qwen3-4B in bfloat16 with prefix caching and `max_model_len` 16384. The
  Colab runtime only allows outgoing connections, so the router runs next to the TPU and reaches
  the GPU through an SSH tunnel via the PSC login node. All of it is scripted: one setup script
  and one server job on PSC, one notebook on Colab.
- **Sanity checks:** a backend check sends the same long prefix twice and requires
  `cached_tokens > 0` on the second request; the analysis flags any run whose hit rate exceeds
  its upper bound. This caught an early bug where back-to-back runs reused each other's cache.

Getting a working stack took several fixes worth noting for others: current vLLM wheels are
built with CUDA 13, which has no V100 kernels; with an older vLLM, V100 starts but its
prefix-prefill Triton kernel fails to compile, so we moved to L40S. On Colab, vllm-tpu has no
wheel for the default Python 3.13 and runs from a Python 3.12 environment.

### 3.4 Baseline policies

All policies decide **once per LLM call, when the call arrives at the router**, and use only
what the router itself observes:

- The router knows its own **in-flight count** per device: requests it has sent and whose
  response stream has not finished (this includes time queued inside vLLM, prefill and decode).
- It does **not** see vLLM's internal queue, which prefixes a device currently holds, or when a
  prefix is evicted. Cache effects are only observed after the fact, through `cached_tokens`.
- The two backends are named `gpu` and `tpu`; where a policy needs a preferred device it is `gpu`.

| Policy | Rule | Parameters / tie-breaking | What it ignores |
|---|---|---|---|
| **LB** (least loaded) | Send each call to the device with the fewest in-flight requests. | Ties go to `gpu`. Decided per call, independently of the workflow's previous steps. | Cache location: consecutive steps of a workflow can land on different devices. |
| **FF** (fast first) | Send to `gpu` while it has fewer than *T* in-flight requests; otherwise send to `tpu`. | *T* = 4 (fixed). `gpu` is treated as the fast device by assumption, not by measurement. | Cache location and the actual device speeds; once `gpu` is above *T*, calls spill regardless of where the workflow's cache is. |
| **CA** (cache affinity) | The first step of a workflow goes to the least-loaded device (as LB); every later step of that workflow goes to the same device. | The home device is chosen once and never revisited. | Load after the first step: a busy home device keeps receiving its workflows. |
| **RR** (round robin) | Alternate `gpu`, `tpu`, `gpu`, … | One counter shared by all workflows (not per workflow). | Both load and cache location; used as the naive reference. |
| **PS** (Production Stack, session mode) | Official vLLM router; hashes the session id (= workflow id) to a backend. | Run standalone with static backends; header `x-session-id`. | Load and device speed. Ran on PSC; not yet in the cross-device runs. |

### 3.5 GapShift (planned design)

At the start of the tool gap after step *k*, the workflow sits on device *h* and the router knows
*p_k*. For each device *d* it estimates the next call's cost as expected queueing on *d* plus the
prefill time on *d* for the tokens *d* does not have cached, using the measured profile from RQ2.
If another device is cheaper by a margin and the gap is long enough, GapShift sends a pre-warm
request (*p_k*, one output token) to that device during the gap, so its cache already holds the
context when the next step arrives and only the new tokens need prefilling. Otherwise the
workflow stays. This keeps CA's cache reuse while letting workflows leave an overloaded or
memory-starved device, which Section 4.3 shows CA cannot do.

## 4. Experiments

All runs: Qwen3-4B, bfloat16, `max_model_len` 16384, synthetic traces with 3 or 5 steps per
workflow, starting context 2k or 8k tokens growing to ~10.5k, 20–60 output tokens and 0.2–0.8 s
tool waits per step, one repetition. Full setups and raw outputs are in `results/`.

### 4.1 Device check

Same 3k-token prompt sent twice, one output token, unloaded:

| | TPU v5e | L40S (through tunnel) | L40S minus RTT |
|---|---|---|---|
| Cold prefill | 216 ms | 469 ms | ≈ 270 ms |
| Warm (prefix hit) | 74 ms | 227 ms | ≈ 28 ms |
| Network RTT from the router | 0.6 ms | 199 ms | |

Cold prefill is a large dense matrix multiplication, which the TPU's systolic arrays handle well:
the two devices are about equal. A cache hit leaves only a few dozen new tokens; there the TPU's
fixed per-step overhead (padding to fixed shapes, paged KV reads) dominates and the L40S is about
2.6× faster. **There is no fixed fast/slow order** (RQ2): the better device depends on the request.

### 4.2 Light load: 12 workflows, concurrency 4 and 8

At concurrency 8 (hit-rate upper bound 71.9%), CA had the shortest makespan (12.4 s vs. 14.4 s
for RR), the lowest median TTFT (0.18 s vs. 0.34 s for RR) and reached the hit-rate bound
(71.8%); LB, FF and RR lost about 20 points of hit rate. At this load tool waits dominate the
makespan, so the differences are small; Section 4.3 raises the load.

### 4.3 High load: 64 workflows, concurrency 8, 16, 32

256 LLM calls per run; hit-rate upper bound 71.9%.

| Conc. | Policy | Makespan | Mean wf time | TTFT p50 | TTFT p99 | Cache hit | Calls tpu / gpu | Workflows switched |
|---|---|---|---|---|---|---|---|---|
| 8 | **CA** | **48.6 s** | **5.7 s** | **0.19 s** | 1.44 s | **71.8%** | 128 / 128 | 0 / 64 |
| 8 | LB | 54.3 s | 6.5 s | 0.27 s | 1.44 s | 52.1% | 120 / 136 | 49 / 64 |
| 8 | FF | 57.0 s | 6.7 s | 0.30 s | 1.53 s | 49.0% | 110 / 146 | 52 / 64 |
| 8 | RR | 57.4 s | 6.9 s | 0.27 s | 1.47 s | 45.9% | 128 / 128 | 62 / 64 |
| 16 | **CA** | **37.5 s** | **8.7 s** | **0.25 s** | 3.12 s | **61.7%** | 113 / 143 | 0 / 64 |
| 16 | LB | 43.9 s | 10.2 s | 0.40 s | 3.10 s | 43.8% | 113 / 143 | 49 / 64 |
| 16 | FF | 57.7 s | 13.3 s | 0.80 s | 4.88 s | 30.1% | 107 / 149 | 52 / 64 |
| 16 | RR | 57.4 s | 13.5 s | 0.76 s | 4.39 s | 29.2% | 128 / 128 | 54 / 64 |
| 32 | **CA** | **41.0 s** | **16.7 s** | **0.82 s** | 6.16 s | **48.8%** | 89 / 167 | 0 / 64 |
| 32 | LB | 44.6 s | 19.3 s | 1.21 s | 6.22 s | 35.7% | 91 / 165 | 41 / 64 |
| 32 | FF | 57.6 s | 24.7 s | 0.96 s | 11.60 s | 28.9% | 118 / 138 | 48 / 64 |
| 32 | RR | 59.3 s | 24.6 s | 2.38 s | 11.03 s | 26.4% | 128 / 128 | 57 / 64 |

**Analysis: why the numbers look like this.**

1. **Cache affinity wins at every load** (RQ1). A miss re-prefills up to ~10k tokens (hundreds of
   ms); a hit prefills only the ~500 new tokens (tens of ms). At these context lengths the saved
   prefill outweighs CA's load imbalance: at concurrency 16, CA's makespan is 35% below RR's and
   its mean workflow time 36% lower.
2. **CA's hit rate still falls with load** (71.8% → 61.7% → 48.8%) **although no workflow ever
   switches device.** The lost hits are evictions. At concurrency 32 there are ~32 live contexts
   of several thousand tokens each, more KV cache than the devices hold (the v5e has about 8 GB
   left after the 4B weights, roughly 50k tokens of KV cache), so a workflow's prefix is often
   evicted during its tool wait. Affinity alone cannot keep the cache once the working set
   exceeds memory, and the smaller-memory device is hit first.
3. **CA becomes unbalanced** (89 TPU vs. 167 GPU calls at concurrency 32) because the home device
   is chosen once, from in-flight counts at the first step, and never revisited. Past
   concurrency 16 the system is saturated: CA's makespan no longer improves (37.5 → 41.0 s) and
   mean workflow time doubles (8.7 → 16.7 s); extra workflows only queue.
4. **RR and FF lose most of the cache** (26–30% hit at concurrency 16 and 32). RR alternates with
   one global counter, so consecutive steps of a workflow usually land on different devices; it
   only hits when it returns to a device that still holds an older prefix, which gets rarer as
   more workflows interleave. FF's `gpu` is above its threshold (*T* = 4) almost all the time at
   concurrency ≥ 8, so 40–45% of calls spill to the TPU regardless of where the cache is.
5. **FF and RR are throughput-bound** (makespan ≈ 57 s at every concurrency). With ~70% of prompt
   tokens recomputed, both devices spend their capacity on redundant prefill; more concurrency
   only adds queueing (TTFT p99 ≈ 11 s at 32).
6. **LB is second.** It balances load well (makespan within 9% of CA at 32) but loses 13–20 points
   of hit rate against CA, which shows up as 44–58% higher median TTFT.

Together, (2) and (3) are the opening for GapShift: the best baseline keeps the cache but cannot
move a workflow off an overloaded or memory-starved device, while the policies that do move
workflows throw the cache away.

**Limitations.** One repetition per setting (p99 is close to the single slowest call). The GPU
path includes a 199 ms network round trip from Colab to PSC, a placement artifact rather than a
hardware property. The eviction explanation is inferred from hit rates and memory sizes; we have
not yet collected vLLM's prefix-cache metrics. PS was not run in the cross-device setup, and
contexts stop at ~10.5k tokens.

## 5. Remaining Work and Timeline

| Dates | Work |
|---|---|
| 11/5–11/9 | Long-context group: both servers at `max_model_len` 32768, contexts growing to ~27k, concurrency 2/4/8 (notebook ready). Remove the network confound: run the TPU in a GCP zone near PSC (or the course cluster); report RTT-corrected TTFT. Add PS to the cross-device runs. Device profiles: cold/warm TTFT for 1k–32k tokens and decode speed on both devices (Keyi). |
| 11/10–11/16 | HotpotQA ReAct traces (Hannah, 11/10). 2–3 repetitions of all sweeps; collect vLLM prefix-cache metrics to confirm evictions; figures for workflow completion time and latency breakdown. |
| 11/17–11/26 | Implement GapShift (cost model from the profiles, pre-warm during gaps); ablations over gap length, context length and concurrency. |
| 11/27–12/4 | Final experiments, report and presentation (12/4). |

Roles: Yanze, system and experiments (vLLM on both devices, connection, router and baselines,
sweeps, GapShift); Keyi, device profiling and figures; Hannah, workloads and the replay client.
