# SETUP_STATUS

Living record. Newest step last within each section.

## Completed

| # | step | outcome |
|---|---|---|
| 1 | Environment audit | macOS 15.5, Apple M2 arm64, 17.2 GB RAM, no CUDA. Python 3.12.7 (anaconda base). git 2.43.1. |
| 2 | Disk freed | 1.8 GB -> ~9 GB available. Sufficient for toolchain + 0.5B smoke model; NOT for a local 7B. |
| 3 | Plunkett repo audited | Read-only clone in scratchpad, never modified. Scenario pool consumed as input only. |
| 4 | Design frozen | 16 contexts, 4 adapters, randomised balanced non-blocked cell assignment, seed 20260831. |
| 5 | Scaffold created | configs/, src/selfinterp/, tests/, data/, analysis/, runs/, reference/. |
| 6 | Data generated | 15,040 distinct option pairs; 4 adapter-specific training sets; 3 common held-out sets. |
| 7 | Integrity suite v1 | 28 passed, 1 skipped. |
| 8 | Design flaw found | Unrestricted independent draws gave chance-collinear nonshared pairs (C15 A2-B1 cos=+0.961). v1 manifest ARCHIVED, not deleted. |
| 9 | Constrained regeneration | \|cos\| <= 0.50 for all nonshared pairs; 42 redraws over 16 contexts; worst nonshared \|cos\| now 0.487. 36 of 64 policy vectors changed. |
| 10 | Flip stratification added | Exactly 35 flips per order per weak adapter x context cell (was 70 unstratified). |
| 11 | Integrity suite v2 | 35 passed, 0 skipped. |
| 12 | venv + model stack | .venv (864 MB), torch 2.5.1 / transformers 4.46.3 / trl 0.12.1 / peft 0.13.2 / accelerate 1.1.1 / datasets 3.1.0. Anaconda base untouched. |
| 13 | Base revision pinned | Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28 |
| 14 | Tokenization gate | PASSED. A=32, B=33, single distinct tokens, template prefix-stable over 200 prompts. |
| 15 | Loss-masking test | PASSED. 178.9 prompt tokens masked / 3.0 response tokens supervised per example (98.35% masked). |
| 16 | Pipeline built | data_access (reserved-split guard), run_utils, collator, train_lora, evaluate, run_eval, check_reload. Same code path for smoke and 7B; only configs/smoke.yaml overlays differ. |
| 17 | 7B LoRA targets verified | All 7 target modules x 28 layers present, proven from the 28 KB shard index. No weights downloaded. |
| 18 | Collator vs reference | 500/500 EXACT token-for-token. TRL DataCollatorForCompletionOnlyLM (token-subsequence search) vs selfinterp.masking (prefix differencing) - independent implementations. |
| 19 | Smoke training | COMPLETED, exit 0. 50 steps, 1874.8s, final loss 0.2318, no NaN. Adapter saved (17 MB). Reserved-split guard clean. |

## Decisions on record

- Common (not per-adapter) held-out sets for policy_val / self_pred / relational, so shared-vs-independent
  adapters can be compared on identical prompts. Training remains adapter-specific.
- Held-out labels always come from the UNCORRUPTED target policy. The 35% flip touches training labels only.
- Predictability is primary for strong-vs-weak; target accuracy is secondary.
- Malformed outputs stay in the denominator of the primary match statistic and never count as a match.
- Same-order match primary; cross-order underlying-option consistency is the robustness measure.
- Pooled CIs must use a hierarchical bootstrap clustering on the 16 contexts.
- bf16 LoRA on a 48 GB A40. QLoRA only if downsized to 24 GB.
- Rescue attempt, if needed: 3 -> 4 epochs. Nothing else changes. One attempt only.
- No nested git repo created (~/Desktop already sits inside a repo rooted at ~).

## Environment used for generation

anaconda base interpreter, read-only. NOTHING was installed. numpy 1.26.4 / PyYAML 6.0.1 / pytest 7.4.4
were already present. Recorded in data/design/manifest.json.

## Condition labelling (required disclosure)

Nonshared cells contain **separately trained, LOW-OVERLAP policies**, not unrestricted independent
draws. Rejection sampling on |cos| <= 0.50 truncates the joint distribution; acceptance statistics
per context are in data/design/manifest.json under `policy_constraint.acceptance`. Selection used
target weights ONLY -- no model behaviour, no held-out performance.

## Blockers

- 7B weights NOT downloaded (only tokenizer, 23 MB). No cloud resources, no paid services.
- Disk: 8.2 GB free. Enough for the 0.5B smoke model (~1 GB); NOT enough for local 7B weights (~15 GB).
- TRL 0.12.1 has no `assistant_only_loss` kwarg; training will use the verified
  selfinterp.masking reference or a collator asserted to match it.

## Smoke run COMPLETE

Run id: `runs/20260901_112326_smoke_qwen05b`. All 11 pipeline checks passed.
Reload determinism was BIT-IDENTICAL (max |delta logit_diff| = 0.0 over 80 prompts).
Base sampling produced 8/128 malformed outputs, all of the form `'B:'` -- recorded as MALFORMED,
never coerced. This is the exact failure mode the Plunkett audit flagged as silently mis-scored.

## Open local issue

Disk fell from 9 GB to 4 GB across the eval run (swap file growth, not project artifacts --
run outputs total ~17 MB). Needs attention before further local work; does not affect the cloud run.

## Cloud harness remediation (2026-09-02) -- COMPLETE

| fix | evidence |
|---|---|
| single `set -euo pipefail` cloud script | `scripts/run_cloud_a1.sh`, 16 fatal gates, `bash -n` clean |
| checkpointing + true resume | SIGKILL test: resumed at step 3, finished at 12, only 9 steps trained, optimizer/scheduler/RNG restored, rotation to [9,12] honoured |
| sampling params wired from frozen config | `tests/test_sampling_params_wired.py`, 7 tests |
| context-level reporting | `analysis/context_level_report.py` + `tests/test_context_statistics.py`, run on smoke outputs |
| artifact preservation + verification | IMPLEMENTED in `run_cloud_a1.sh` s8 + runbook s4-5. LOCALLY TESTED on the smoke run (26 files, tamper detected, tarball round-trip OK). NOT verified pod-to-laptop -- no cloud run has occurred. |
| pre-launch billing gate | typed `I CONFIRM` for auto-recharge off / balance <= $25 / script exit != billing stop |

Config hashes after these edits: design/data/prompt UNCHANGED (so all 16 generated data files
still re-verify, 16/16). model.yaml CHANGED (revision pinned), train.yaml CHANGED (checkpoint keys),
eval.yaml CHANGED (operational sizes moved in). No scientific value altered.

Two bugs were found by the fixed test harness and repaired:
  1. `resumed_from` was read AFTER training, but save_total_limit had already rotated that
     checkpoint away -> FileNotFoundError. Now captured before training.
  2. `scripts/test_resume.sh` piped the run through `grep`, which swallowed tracebacks and made
     a real failure look like silence. Now tee'd to a log with explicit exit-code checking.

## Statistical audit (2026-09-02)

Two clustering violations found and fixed in the reporting layer:
  1. rate / agreement CIs used Wilson over ALL generations, treating repeated samples of one
     prompt as independent. Replaced with a stimulus-level cluster bootstrap.
  2. strong/weak and pooled CIs resampled CELLS (context x order), so each context contributed
     two correlated cells. Replaced with a hierarchical contexts -> stimuli bootstrap.
Estimators moved to `analysis/estimators.py` with an explicit L0..L4 unit hierarchy.
Wilson retained ONLY as a clearly-labelled diagnostic to expose how anticonservative it is.

## Recipe v2 (2026-09-03)

Gates frozen in GATES_v2.md before any v2 model existed. v2 changes supervision
mode ONLY (`loss_mode: choice_only`).

| step | outcome |
|---|---|
| v2 configs | `configs/smoke_v2.yaml`, `configs/a2v2_cloud.yaml`, `configs/a1v2_cloud.yaml` |
| v2 smoke, attempt 1 | `runs/20260903_143858_smokev2_qwen05b` - INTERRUPTED_BY_HOST_RESTART (host booted 14:41:56 mid-training). No error in log, no checkpoint possible at 50 steps with save_steps=100. Preserved unchanged. |
| v2 smoke, attempt 2 | `runs/20260903_145754_smokev2_qwen05b` - **PASSED all 9 stages** from step 0. 71 tests; tokenization gate; collator 3200/3200 exact with exactly 1 supervised token/example; 50 steps / 574.4 s / final_loss 0.6828; base + adapter eval in fresh processes; reload determinism bit-identical; context-level report; 39 artifacts hash-verified; reserved-split guard clean. |
| local smoke driver | `scripts/run_smoke_v2_local.sh` - mirrors `run_cloud_a2.sh` gate for gate, minus billing gate and CUDA check |
| v2 cloud scripts | `scripts/run_cloud_a2_v2.sh`, `scripts/run_cloud_a1_v2.sh` - `bash -n` clean |
| v2 claim verified mechanically | resolved-config diff v1 -> v2 is exactly `run_tag` + `train.loss_mode` for BOTH adapters; seed/epochs/lr/batch/rank/contexts/revision identical |

Loss scale confirmed on real output: v2 reads on the ln 2 = 0.693147 scale
(smoke final_loss 0.6828) against v1's ~0.231 scale. Diagnosability only - the
smoke shows no acquisition (0.469 vs base 0.523), which is expected for a 0.5B
base with r=8 on 200 examples and carries no scientific weight.

## Block 2 - A2-v2 (2026-09-03)

Run `20260903_195207_a2v2_qwen7b`. **FAILS GATE 1** on D1, D2 and D3.
Verdict: `runs/20260903_195207_a2v2_qwen7b/metrics/A2_V2_VERDICT.json`.

| | result |
|---|---|
| D1 improvement over base | +0.0427, CI [-0.058, +0.146] vs threshold +0.20 - **FAIL** |
| D2 breadth | 9/16 and 0/16 vs 12/16 and 10/16 - **FAIL** |
| D3 output validity | adapter greedy malformed **1.000** (base 0.000) - **FAIL** |
| D4 position materiality | PASS (vacuously; both quantities near zero) |
| D5 strong vs weak | 0/4 - FAIL (non-blocking) |
| choice-token loss | 0.6902 vs chance ln 2 = 0.693147; S3 margin 0.0029 < 0.02 - **FAIL** |

Two findings, pulling in opposite directions:

1. **v2's prediction held.** A2-v2 reproduces A2-v1's choice-scale loss to within
   0.0001. Supervision mode was never the cause of A2's failure to acquire.
2. **v2's "diagnosability only" claim is falsified.** Masking `<|im_end|>` removed
   the constraint holding termination in place; the adapter now emits `'BBBBBBBB'`
   and never stops. A token carrying negligible LOSS was still carrying a
   CONSTRAINT. The choice-only loss cannot see this, so the same masking that
   caused the failure also hid it.

The 0.5B smoke could not have caught it (adapter freegen malformed 0.000, equal to
base) - the defect is scale- and duration-dependent.

## Exact next action

**STOP for a scientific decision.** GATE 2 cannot pass (S1 needs both adapters
through GATE 1). **A1-v2 deliberately NOT run** - it would spend on a recipe now
known to be defective.

Recommended, not yet approved: abandon the masking change and get diagnosability
by REPORTING choice-token cross-entropy alongside v1 completion-mode supervision.
Zero change to gradients or trained behaviour. Alternatively mask only the
trailing newline, keeping `<|im_end|>` supervised.

The original question - why A1 acquired and A2 did not on an identical recipe -
remains open and is now known NOT to be about supervision mode.

Pod: `root@69.30.85.97 -p 22005`, running. `/workspace/.ssh_backup/authorized_keys`
persists across restarts. Artifacts retrieved and hash-verified (25/25); the pod
can be stopped.

Billing gate is unchanged and still applies: confirm RunPod auto-recharge OFF and
balance <= $25 before launching.
