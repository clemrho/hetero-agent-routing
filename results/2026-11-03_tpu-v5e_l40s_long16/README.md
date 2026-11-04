# 2026-11-03 · TPU v5e + L40S · long context (16 workflows, contexts to ~27k, concurrency 2 / 4 / 8)

Colab TPU v5e + PSC L40S over an SSH tunnel (~199 ms RTT), Qwen3-4B, bf16, prefix caching on,
**`max_model_len` 32768 on both servers**. No failed calls. Raw output: [raw_output.txt](raw_output.txt).
Per-call logs: Google Drive `MyDrive/har/long16_c2-8/`.

## Commands

```bash
python traces/gen_synthetic.py --max_model_len 32768 --out <trace> \
    --steps 3 5 --init_ctx 8192 16384 --per_combo 4 \
    --obs_min 1500 --obs_max 3000 --out_min 20 --out_max 60 --tool_min 0.5 --tool_max 2.0
python replay/replay.py --trace <trace> --sweep --policies LB FF CA RR \
    --concurrency_list 2 4 8 --reps 1 --warmup ... --log_dir logs/long16_c2-8
```

Trace: 16 workflows (3 or 5 steps, start 8k or 16k tokens, 1.5–3k tokens added per step, up to
27,124 tokens), 64 LLM calls, 62 s of tool wait in total. Cache-hit upper bound: **70.1%**.

## Two repetitions

The log folder holds two complete sweeps of the same trace (same generator seed; each run uses its
own prompt salt, so runs cannot share cache):

- **Rep A** (`long16/`, device label `gpu`): branch `yanze/colab-tpu-psc`, all 12 runs.
- **Rep B** (`long16_c2-8/`, device label `a100`): same policies run from `main`, where the GPU
  was still labelled `a100`; the routing logic is identical. LB at concurrency 2 is missing: an
  empty log left by an earlier failed attempt made replay skip it.

## Results (rep A; rep B in brackets where it exists)

| Conc. | Policy | Makespan | Mean wf time | TTFT p50 | TTFT p99 | Cache hit | Calls tpu/gpu | Wf switched |
|---|---|---|---|---|---|---|---|---|
| 2 | **CA** | **71.9 s** (69.7) | 8.6 s | **0.47 s** | 1.43 s | 70.0% (70.0) | 27 / 37 | 0/16 |
| 2 | FF | 80.4 s (80.6) | 9.7 s | 0.57 s | 1.52 s | 70.1% (70.1) | 0 / 64 | 0/16 |
| 2 | LB | 79.2 s (–) | 9.3 s | 0.59 s | 1.81 s | 51.9% (–) | 25 / 39 | 13/16 |
| 2 | RR | 76.7 s (76.7) | 8.9 s | 0.52 s | 1.55 s | 58.1% (58.1) | 32 / 32 | 9/16 |
| 4 | **CA** | **42.4 s** (43.6) | **9.3 s** | **0.43 s** | 1.93 s | 70.0% (70.0) | 31 / 33 | 0/16 |
| 4 | FF | 54.9 s (55.3) | 12.5 s | 0.64 s | 3.35 s | 70.1% (70.1) | 0 / 64 | 0/16 |
| 4 | LB | 50.1 s (50.9) | 10.9 s | 0.68 s | 2.55 s | 51.2% (42.7) | 29 / 35 | 11/16 |
| 4 | RR | 51.2 s (53.7) | 11.6 s | 0.80 s | 2.60 s | 46.5% (41.3) | 32 / 32 | 12/16 |
| 8 | **CA** | **34.5 s** (34.4) | **14.7 s** | **0.73 s** | 4.34 s | **59.0%** (59.0) | 27 / 37 | 0/16 |
| 8 | FF | 41.1 s (41.6) | 16.6 s | 1.42 s | 4.90 s | 45.7% (38.7) | 21 / 43 | 10/16 |
| 8 | LB | 41.0 s (37.6) | 17.3 s | 1.30 s | 6.23 s | 40.9% (42.9) | 26 / 38 | 13/16 |
| 8 | RR | 47.5 s (49.5) | 20.1 s | 1.66 s | 6.92 s | 34.9% (28.1) | 32 / 32 | 14/16 |

## Why the numbers look like this

1. **CA wins at every concurrency, by more than with short contexts.** At concurrency 8 CA's
   makespan is 27% below RR's (15% in the short-context sweep) and RR's median TTFT is 2.3× CA's
   (1.5× with short contexts). A miss now re-prefills 8k–27k tokens instead of 2k–10k, so every
   lost hit costs more: this is the effect the TA predicted for long contexts.
2. **Keeping the cache is not enough; both devices must be used.** At concurrency 2 and 4, FF never
   exceeds its threshold (4 in flight), so it sends all 64 calls to the GPU. It reaches the
   hit-rate bound like CA, but CA's makespan is 11–23% lower, because FF leaves the TPU idle and
   every call pays the 199 ms tunnel RTT.
3. **Evictions start at lower concurrency.** CA never switches devices, yet its hit rate drops from
   70.0% to 59.0% at concurrency 8 (in the short-context sweep it held until 16). Eight contexts of
   14k–27k tokens need far more KV cache than the v5e has (~8 GB left, roughly 50k tokens), so
   prefixes are evicted during tool waits. Longer contexts hit the memory limit sooner.
4. **CA's placement is deterministic, the others' is not.** CA and FF (at 2 and 4) give the same
   placement and nearly the same numbers in both repetitions; LB, RR and FF at 8 decide from
   in-flight counts at each call, so their hit rates vary by up to 9 points between repetitions.
5. **RR does better at concurrency 2 (58%) than at higher load.** With only two workflows
   interleaving, the global alternation often sends a workflow back to the same device
   (only 9/16 switched); with more workflows interleaved this alignment disappears.

## Caveats

- Two repetitions for 11 of 12 settings, one for LB at concurrency 2.
- GPU path includes ~199 ms network RTT. Eviction explanation inferred from hit rates and memory
  sizes (vLLM prefix-cache metrics not collected).
