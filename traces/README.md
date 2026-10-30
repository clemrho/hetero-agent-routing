# Traces (owner: Hannah)

One JSON object per line, **one line per workflow**. Only token counts and timings are
stored, not text: prefix caching only depends on *which tokens are shared and how many*,
so `replay/replay.py` rebuilds prompts from these numbers (as token ids) and both devices
see identical input.

```json
{
  "wf_id": "syn-s5-c4096-2",
  "source": "synthetic",
  "arrival_s": 0.0,
  "steps": [
    {"prompt_tokens": 4096, "output_tokens": 120, "tool_s": 1.4, "obs_tokens": 512},
    {"prompt_tokens": 4728, "output_tokens": 95,  "tool_s": 0.0, "obs_tokens": 0}
  ]
}
```

| Field | Meaning |
|---|---|
| `prompt_tokens` | full context sent at this step; must never shrink (earlier steps are a prefix of later ones) |
| `output_tokens` | tokens generated at this step (replay sends `max_tokens` = this, with `ignore_eos`) |
| `tool_s` | tool wait after this call, the gap GapShift uses; 0 on the last step |
| `obs_tokens` | tool output appended before the next step (informational; `prompt_tokens` is what replay uses) |
| `arrival_s` | start time for `--mode open`; ignored by the default closed-loop replay |

Required: `wf_id`, `steps[].prompt_tokens`, `steps[].output_tokens`, `steps[].tool_s`. Anything else is metadata.

## Synthetic

```bash
python traces/gen_synthetic.py                         # 32 workflows -> traces/synthetic.jsonl
python traces/gen_synthetic.py --max_model_len 16384   # if vLLM runs with --max-model-len 16384
```

Plan parameters: steps {3, 5, 7, 9} × initial context {1k, 4k}, 4 workflows per combo,
400–600 tokens of tool output per step, 0.5–2 s tool wait, 50–200 output tokens.

**`--max_model_len` must match vLLM.** At 8192, the 9-step × 4k workflows don't fit and get cut to
7 steps (the generator prints a table of what was cut). Running vLLM with `--max-model-len 16384`
on both devices keeps every workflow at full length.

## HotpotQA (due 11/10)

Same format, `"source": "hotpotqa"`. Run ReAct on the questions and record, per step: the token
count of the full prompt, the model's output length, and how long the tool (local paragraph
retrieval) took. Real token counts come from the Qwen3 tokenizer.
