# 2026-11-03 · TPU v5e + L40S · short context, high load (64 workflows, concurrency 8 / 16 / 32)

Same setup as [quick12](../2026-11-03_tpu-v5e_l40s_quick12/README.md) (Colab TPU v5e + PSC L40S
over an SSH tunnel with ~199 ms RTT; Qwen3-4B, bf16, `max_model_len` 16384), with 64 workflows so
that concurrency 32 really has 32 workflows in flight. One repetition, no failed calls.
Raw output: [raw_output.txt](raw_output.txt). Per-call logs: Google Drive `MyDrive/har/short64_c8-32/`.

## Commands

```bash
python traces/gen_synthetic.py --max_model_len 16384 --out logs/short64.jsonl \
    --steps 3 5 --init_ctx 2048 8192 --per_combo 16 \
    --out_min 20 --out_max 60 --tool_min 0.2 --tool_max 0.8
python replay/replay.py --trace logs/short64.jsonl --sweep --policies LB FF CA RR \
    --concurrency_list 8 16 32 --reps 1 --warmup 1024 2048 4096 8192 12288 --log_dir logs/short64_c8-32
```

Trace: 64 workflows (3 or 5 steps, start 2k or 8k tokens, up to 10.6k), 256 LLM calls, 95 s of
tool wait in total. Cache-hit upper bound: **71.9%**.

## Results

| Conc. | Policy | Makespan | Mean wf time | TTFT p50 | TTFT p99 | Cache hit | Calls tpu/gpu | Wf switched |
|---|---|---|---|---|---|---|---|---|
| 8 | **CA** | **48.6 s** | **5.7 s** | **0.19 s** | 1.44 s | **71.8%** | 128 / 128 | 0/64 |
| 8 | LB | 54.3 s | 6.5 s | 0.27 s | 1.44 s | 52.1% | 120 / 136 | 49/64 |
| 8 | FF | 57.0 s | 6.7 s | 0.30 s | 1.53 s | 49.0% | 110 / 146 | 52/64 |
| 8 | RR | 57.4 s | 6.9 s | 0.27 s | 1.47 s | 45.9% | 128 / 128 | 62/64 |
| 16 | **CA** | **37.5 s** | **8.7 s** | **0.25 s** | 3.12 s | **61.7%** | 113 / 143 | 0/64 |
| 16 | LB | 43.9 s | 10.2 s | 0.40 s | 3.10 s | 43.8% | 113 / 143 | 49/64 |
| 16 | FF | 57.7 s | 13.3 s | 0.80 s | 4.88 s | 30.1% | 107 / 149 | 52/64 |
| 16 | RR | 57.4 s | 13.5 s | 0.76 s | 4.39 s | 29.2% | 128 / 128 | 54/64 |
| 32 | **CA** | **41.0 s** | **16.7 s** | **0.82 s** | 6.16 s | **48.8%** | 89 / 167 | 0/64 |
| 32 | LB | 44.6 s | 19.3 s | 1.21 s | 6.22 s | 35.7% | 91 / 165 | 41/64 |
| 32 | FF | 57.6 s | 24.7 s | 0.96 s | 11.60 s | 28.9% | 118 / 138 | 48/64 |
| 32 | RR | 59.3 s | 24.6 s | 2.38 s | 11.03 s | 26.4% | 128 / 128 | 57/64 |

## Why the numbers look like this

1. **CA wins at every load.** A cache miss re-prefills up to ~10k tokens (hundreds of ms); a hit
   prefills only the ~500 new tokens (tens of ms). At these context lengths the saved prefill is
   worth more than CA's load imbalance. At concurrency 16, CA's makespan is 35% below RR and its
   mean workflow time is 36% lower.
2. **CA's hit rate still falls with load (71.8% → 61.7% → 48.8%) although no workflow ever
   switches device.** The lost hits are evictions: at concurrency 32 there are ~32 live contexts of
   several thousand tokens each, more KV cache than the devices hold (the v5e has ~8 GB left after
   the 4B weights, roughly 50k tokens), so a workflow's prefix is often evicted during its tool
   wait. Affinity alone cannot keep the cache once the working set exceeds memory.
3. **CA also gets unbalanced at high load (89 TPU vs 167 GPU calls at concurrency 32).** CA picks
   a workflow's device once, by in-flight count at its first step, and never revisits it. Its
   mean workflow time doubles from 16 to 32 (8.7 → 16.7 s) while makespan gets slightly worse
   (37.5 → 41.0 s): past concurrency 16 the system is saturated and extra workflows only queue.
4. **RR and FF lose most of the cache (26–30% hit at 16 and 32).** RR alternates devices with one
   global counter across all workflows, so consecutive steps of a workflow usually land on
   different devices; it only gets a hit when it comes back to a device that still holds an
   older prefix, which gets rarer as more workflows interleave. FF's GPU is past its threshold
   (4 in flight) almost all the time at concurrency ≥ 8, so 40–45% of calls spill to the TPU with
   no regard for where the workflow's cache is (52/64 workflows switch already at 8).
5. **FF and RR are throughput-bound (makespan ~57 s at every concurrency).** With ~70% of prompt
   tokens recomputed, both devices spend their time on redundant prefill, so adding concurrency
   only adds queueing: TTFT p99 grows to 11 s at 32.
6. **LB is second.** It balances load well (makespan within 9% of CA at 32) but loses
   13–20 points of hit rate against CA, which shows up as 44–58% higher median TTFT.

## Caveats

- One repetition. The GPU path includes ~199 ms of network RTT (Colab ⇄ PSC).
- The eviction explanation in (2) is inferred from the hit rates and memory sizes; vLLM's
  prefix-cache metrics (`/metrics`) were not collected in this run.
