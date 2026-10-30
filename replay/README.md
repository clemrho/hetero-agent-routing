# Replay (owner: Hannah)

`replay.py` reads a trace and drives `router.Router` exactly as the trace says:
call the LLM with the step's context, wait `tool_s`, call again with the longer context, and so on.
Every policy replays the same workflows with the same lengths and waits.

## Run

```bash
export TPU_URL=http://localhost:8000 A100_URL=http://localhost:8001   # A100 via SSH tunnel

# one run
python replay/replay.py --trace traces/synthetic.jsonl --policy CA --concurrency 4 --rep 0

# the whole baseline matrix (2 reps); warm up both devices first so XLA compiles aren't timed
python replay/replay.py --trace traces/synthetic.jsonl --sweep \
    --policies LB FF CA --concurrency_list 1 4 8 16 --reps 2 \
    --warmup 512 1024 2048 4096 8000
```

Output in `logs/<trace name>/`:
- `CA_c4_r0.jsonl`: one line per LLM call (written by the Router: 4 timestamps, device, cached_tokens)
- `CA_c4_r0.meta.json`: settings, salt, wall time
- `CA_c4_r0.errors.jsonl`: failed calls, only if any (each call is retried once; a workflow whose call
  still fails is dropped, and the run's summary line says so)

An existing log is skipped, so an interrupted sweep can just be restarted; `--overwrite` redoes it.

## Details worth knowing

- **Concurrency** = number of workflows in flight (closed loop). `--mode open` starts workflows at
  `arrival_s` instead.
- **Prompts are token-id lists**, so token counts are exact. Step k's prompt is a prefix of step k+1's.
  The part standing in for the model's own output is also synthetic, so the KV of what the model
  actually generated is never reused. That's 50–200 tokens out of thousands per step.
- **No cache leaks between runs**: every run uses a fresh random salt for its token ids (same lengths,
  different content), so you don't need to restart vLLM between policies.
- `--limit 4` runs only the first 4 workflows, for a quick check.

## Testing without a TPU or A100

`mock_vllm.py` is a fake vLLM (streaming, prefix cache on 16-token blocks, `cached_tokens` in usage)
for checking the pipeline. Its numbers mean nothing.

```bash
python replay/mock_vllm.py --port 8001 --prefill_tps 20000 --tpot 0.004 &   # "A100"
python replay/mock_vllm.py --port 8000 --prefill_tps 6000  --tpot 0.008 &   # "TPU"
python traces/gen_synthetic.py --tool_min 0.05 --tool_max 0.2 --per_combo 2 --out /tmp/fast.jsonl
python replay/replay.py --trace /tmp/fast.jsonl --sweep --policies LB FF CA RR --concurrency_list 1 4 --reps 1
python analysis/analyze.py logs/fast/*.jsonl
```
