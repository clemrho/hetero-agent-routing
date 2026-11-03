# 2026-11-03 · TPU v5e + L40S · quick baseline (12 workflows)

First baseline run across two different accelerators. Single repetition, so treat the
numbers as a first look, not final results. Raw output: [raw_output.txt](raw_output.txt).
The per-call `.jsonl` logs were saved to Google Drive (`MyDrive/har/tpu_gpu_quick12/`) and
still need to be copied here.

## Setup

| | |
|---|---|
| Code | branch `yanze/colab-tpu-psc` @ `8de3674`, notebook `scripts/colab/tpu_psc_smoke.ipynb` |
| Slow/fast labels | `tpu` = Colab TPU v5e (1 chip, 16 GB), `gpu` = PSC Bridges-2 L40S (48 GB), node `gl003` |
| Serving | vLLM-TPU on Colab; latest vLLM on PSC (V1 engine). Both: `Qwen/Qwen3-4B`, bfloat16, prefix caching on, `max_model_len=16384` |
| Router | `router/router.py` in the Colab notebook; the L40S is reached over an SSH tunnel through `bridges2.psc.edu` |
| Trace | `traces/gen_synthetic.py --steps 3 5 --init_ctx 2048 8192 --per_combo 3 --out_min 20 --out_max 60 --tool_min 0.2 --tool_max 0.8 --max_model_len 16384`: 12 workflows, 48 LLM calls, contexts up to 10,461 tokens, 20 s of tool wait in total |
| Replay | `replay/replay.py --sweep --policies LB FF CA RR --concurrency_list 4 8 --reps 1 --warmup 1024 2048 4096 8192 12288` (closed loop) |

## Device check (3k-token prompt, 1 output token, single request)

| | TPU v5e (local) | L40S (through tunnel) | L40S minus 199 ms RTT |
|---|---|---|---|
| cold (no cache) | 216 ms | 469 ms | ~270 ms |
| warm (prefix hit) | 74 ms | 227 ms | ~28 ms |
| network RTT | 0.6 ms | 198.7 ms | |

## Baseline results

Cache-hit upper bound for this trace (every step reuses the whole previous prompt): **71.9%**.

| Policy | Conc. | Makespan | Mean wf time | TTFT p50 | TTFT p99 | Cache hit | Calls tpu/gpu | Wf switched |
|---|---|---|---|---|---|---|---|---|
| CA | 4 | **18.7 s** | **5.2 s** | **0.174 s** | 1.261 s | 71.8% | 19 / 29 | 0/12 |
| FF | 4 | 19.8 s | 6.0 s | 0.225 s | 1.376 s | 71.9% | 0 / 48 | 0/12 |
| LB | 4 | 19.5 s | 5.5 s | 0.265 s | 1.241 s | 52.1% | 23 / 25 | 10/12 |
| RR | 4 | 19.4 s | 5.6 s | 0.229 s | 1.266 s | 50.9% | 24 / 24 | 10/12 |
| CA | 8 | **12.4 s** | **5.9 s** | **0.181 s** | 1.859 s | 71.8% | 24 / 24 | 0/12 |
| FF | 8 | 13.4 s | 6.8 s | 0.269 s | 1.422 s | 54.7% | 21 / 27 | 9/12 |
| LB | 8 | 12.7 s | 6.6 s | 0.265 s | 1.856 s | 53.3% | 24 / 24 | 9/12 |
| RR | 8 | 14.4 s | 7.2 s | 0.343 s | 2.000 s | 48.7% | 24 / 24 | 10/12 |

## Takeaways

1. **Cache affinity wins at this load.** CA reaches the cache-hit bound and has the lowest median
   TTFT (about half of RR at concurrency 8). LB and RR move workflows between devices and lose
   ~20 points of cache hit rate, paying for re-prefill.
2. **FF spills and loses cache once the "fast" device is busy.** At concurrency 4 everything stays
   on the GPU; at 8 the GPU passes FF's threshold, 9/12 workflows get split across devices and the
   hit rate drops to 54.7%.
3. **The two devices are not a simple fast/slow pair.** Cold prefill is about equal (v5e slightly
   faster); on a cache hit the L40S is ~2.6x faster. Seen from the router, the 199 ms tunnel makes
   the GPU the *slower* endpoint, which is also why FF at concurrency 4 (all GPU) has a higher
   median TTFT than CA.

## Caveats

- One repetition; p99 is close to the single slowest call.
- The GPU path includes ~199 ms of network round trip (Colab ⇄ PSC). This is placement, not hardware.
- Light load: makespan is dominated by tool waits, so makespan differences are small.
- No Production Stack (PS) baseline in this run: its router is not installed on the Colab side.
