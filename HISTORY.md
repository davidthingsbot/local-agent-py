# HISTORY.md — local-agent-py

## 2026-05-24 — Deep-context hard-task tuning loop

David asked to iterate on settings that make the local Qwen agent reliably complete complicated multi-tool tasks without failure. Current baseline:

- `local-agent-qwen.service` is enabled/running under user systemd.
- Server uses `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` on both RTX 3090s.
- Context is 262144 tokens, one slot, q8 KV, layer split 1:1.
- Harness defaults now target deep-context work: larger read/output caps, 8192 completion tokens, delayed compaction.
- Prior hard-task observations:
  - Unbounded audit hit max-turns after 18 tool calls without producing report → failure mode: over-inspection / no synthesis phase.
  - Focused 8-file audit succeeded and wrote `~/work/la-test/hardtask_focused_audit_20260524-011110.md`.
  - GPU headroom during focused hard task: GPU0 ~9.7GB free at peak, GPU1 ~11.1GB free at peak.

Next workstream: run iterative experiments to find settings/prompt/loop changes that make hard multi-tool tasks complete reliably, then recommend or patch the harness. Orchestrator should continue spawning follow-up subagents if a run fails until there is a passing configuration with evidence.

## 2026-05-24 — Settings iteration evidence and threshold patch

Ran three real local-agent hard-task experiments under `/home/david/work/la-test/settings-iter`:

- `baseline_unbounded`: broad repo audit, 12 turns, wrote `baseline_unbounded_audit.md` after recovering from an absolute-path write denial.
- `focused_no_thinking`: bounded 8-file audit, 10 turns, wrote `focused_no_thinking_audit.md` with `--no-thinking --temperature 0.2 --top-p 0.8`.
- `decompose_enabled`: bounded 8-file audit with `LOCAL_AGENT_DECOMPOSE=1`, 12 turns, wrote `decompose_enabled_audit.md` after retrying a blocked absolute write as a relative path.

Evidence-backed conclusion: bounded tasks with explicit finish/write rules are repeatably successful. Decomposition helps the model narrate checkpoints, but the biggest harness-level reliability risk is premature compaction after a single 700K-char read. The README's deep-context validation recorded ~235,052 prompt tokens for a ~680K-char read, while the prior default threshold (`0.85 * 262144 = 222,822`) sat below that. Patched `la.py` to default `LOCAL_AGENT_COMPACT_THRESHOLD_RATIO` to `0.90` and fallback threshold to `250_000`; added `tests/test_threshold.py::test_default_threshold_covers_known_700k_read_for_256k_context`. Full local tests pass: `58 passed, 4 deselected`.

## 2026-05-24 — Iteration 2 reliability patch

Second tuning subagent implemented minimal hard-task reliability changes:

- `DECOMPOSITION_CLAUSE` is now part of the default system prompt instead of gated behind `LOCAL_AGENT_DECOMPOSE=1`.
- Hard-task guidance now explicitly says to plan, checkpoint after roughly ten tool calls, create/update requested artifacts before another broad inspection pass, synthesize when approaching turn budget, and prefer relative write paths.
- Compaction defaults raised for 256K context: ratio `0.85 → 0.90`, fallback `220_000 → 250_000`.
- Empty-response watchdog now tries 4 retries; final retry re-enables thinking at low sampling.
- Tests updated; full unit suite passed: `58 passed, 4 deselected`.
- Two real hard-task validations succeeded under `~/work/la-test/settings-iter2/`; both exited 0 and wrote reports.
- Final report: `~/work/la-test/local_agent_settings_iteration_report.md`.

Remaining risks: no pre-request prompt-size estimate/enforcement yet; checkpointing is still prompt-only; subagent descriptions could be clearer.

## 2026-05-24 — Preferred two-slot service mode

David asked to preserve the known-good setup but make the two-session experiment the preferred setting for now. Current preferred service config:

- `local-agent-qwen.service` runs one llama.cpp server on both RTX 3090s.
- `--ctx-size 524288 -np 2`, reported by `/props` + `/slots` as two slots with `n_ctx=262144` each.
- Concurrent smoke test launched two `./la.py` hard tasks simultaneously; both exited `0` and wrote reports.
- Peak observed memory during the concurrent smoke test: GPU0 ~16.3GB / 24GB, GPU1 ~14.9GB / 24GB, leaving ~8.3GB and ~9.7GB headroom.
- Restore point: git tag `known-good-256k-1slot`; service backup at `~/.config/systemd/user/local-agent-qwen.service.known-good-1slot-256k`.

## 2026-05-24 — Bind Qwen server to all interfaces

David asked to make the model server reachable from other machines. Updated startup scripts, service templates, and docs to bind llama.cpp to `0.0.0.0:19434` and to support remote clients via `QWEN_BASE_URL=http://<model-host>:19434/v1`.

Operational note: removed the Tailscale Serve TCP forward on `19434 -> 127.0.0.1:19434` because it blocked a true all-interface bind. Preserved the HTTPS Tailscale Serve route to `127.0.0.1:18789`.
