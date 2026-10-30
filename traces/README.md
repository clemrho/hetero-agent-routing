# Traces

Draft format (to be agreed — owner: Hannah). One JSON per line, one line per workflow:

```json
{
  "wf_id": "swe-0001",
  "arrival_s": 0.0,
  "steps": [
    {"prompt_tokens": 2100, "output_tokens": 180, "tool_s": 1.4},
    {"prompt_tokens": 2950, "output_tokens": 95,  "tool_s": 0.3}
  ]
}
```

- `prompt_tokens` is the full context at that step (it grows; earlier steps are a prefix of later ones).
- `tool_s` is the tool time after the LLM call (the "gap" GapShift uses). Last step can be 0.
- Replay builds a synthetic prompt with the same token counts and a unique per-workflow prefix,
  so prefix caching behaves like the real workload.
