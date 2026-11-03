# Results

One folder per experiment run, named `<date>_<devices>_<what>`. Each has a `README.md`
(setup, exact commands, table, takeaways, caveats) and the raw output. Large per-call logs
(`logs/` is git-ignored) go here only for runs we report.

| Date | Run | Devices | What |
|---|---|---|---|
| 2026-11-03 | [tpu-v5e_l40s_quick12](2026-11-03_tpu-v5e_l40s_quick12/README.md) | Colab TPU v5e + PSC L40S | First cross-device baseline: LB / FF / CA / RR, concurrency 4 and 8, 12 synthetic workflows, Qwen3-4B |
