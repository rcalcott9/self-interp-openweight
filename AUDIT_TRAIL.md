# Audit trail

## 2026-09-03 - Criterion 4 implementation error, found and corrected

**Criterion 4 as frozen** (GO_NO_GO_CRITERIA.md, approved 2026-08-31, before any
data existed): blocks only if switching from raw to order-averaged agreement
**REDUCES** the adapter-base improvement by more than 25% relative.

**Implementation error.** `analysis/go_no_go.py` coded `passed=(abs(drop) <= 0.25)`.
The absolute value is not in the criterion and is stricter than it: it also fails a
large INCREASE, which is the opposite of the artefact the criterion targets.

**Observed on A1:** raw improvement +0.2062, order-averaged +0.2635, relative
drop **-0.278** (a 27.8% INCREASE). The erroneous form reported FAIL; the
criterion as written gives PASS, because no reduction occurred.

**Why the increase happens:** base letter bias P(first)=0.661 (+0.161) vs adapter
-0.041. Order-averaging strips the base's bias, lowering its score more than the
adapter's, widening the gap. Position noise was masking real acquisition.

**Disposition.** The erroneous output is preserved verbatim at
`runs/20260902_230634_a1_qwen7b/metrics/go_no_go_ERRONEOUS_absdrop.json`. The
code was corrected to the directional form on 2026-09-03 AFTER the discrepancy
was disclosed to and adjudicated by the user. The criterion itself was NOT
amended. No other criterion was touched.

## 2026-09-03 - Stage 3 feasibility analysis revised BEFORE A2 was observed

**Superseded conclusion.** The first feasibility analysis
(`analysis/transfer_gate_power_v1_SUPERSEDED.py`,
`metrics/transfer_gate_power_v1_SUPERSEDED.json`) concluded that the A1-A2
transfer contrast was "sub-threshold": it required delta >= 0.10 and found a
ceiling of ~0.096 even under perfect policy transfer.

**Why that was wrong.** The 0.10 bar was derived from the SPACING of the original
Stage 3 payoff grid (10-point wager, integer costs k in {0,1,2,3,5}, so the
smallest cost step needed 10*delta > 1). That conflates the size of the transfer
effect with the coarseness of the grid used to probe it. A shared-different
contrast well below 0.10 still changes behaviour whenever the two match
probabilities fall on OPPOSITE SIDES of a wager threshold. Decision-relevance is
a property of where the probabilities sit relative to a threshold, not of the
gap between them.

**Correction.** Feasibility is now assessed on a finer threshold grid:
coordination wager 40 points if the predicted Worker choice is correct, 0 if
incorrect, against safe payoffs of 20/21/22/23/24 - i.e. indifference
probabilities .500/.525/.550/.575/.600. This is an integer representation of a
10/0 wager against safe payoffs 5.00-6.00; it does NOT inflate the wager relative
to the safe option, it only makes the threshold grid finer.

**Timing.** This revision was made on 2026-09-03, BEFORE adapter A2 existed and
before any A1-A2 comparison was computed. No A2 outcome informed it. The
superseded analysis and its outputs are retained unmodified.

**Consequence.** The gate is now split into three separately reported components:
scientific transfer, strategic usefulness (threshold crossing), and a minimum
practical effect justified independently of grid spacing.

## 2026-09-03 - v2 smoke interrupted by host restart; rerun from step 0

**Not a training failure.** Run `20260903_143858_smokev2_qwen05b` stopped after
`[train] starting` with no further output. Log inspection found **no traceback,
no exception, no nonfinite loss, no OOM and no kill signature** - the full 10-line
log contains zero error matches. The host booted at **14:41:56** (`sysctl
kern.boottime`) against a last artifact write at **14:39**, and `last reboot`
shows that reboot with **no paired shutdown entry** (unlike the 2026-08-29 pair),
i.e. an unexpected restart ~2-3 minutes into training.

No valid checkpoint existed, and none could have: `save_steps: 100` while the
smoke overlay yields 200 examples / batch 4 / accum 1 / 1 epoch = **50 optimizer
steps**. Checkpoint absence is expected here and is not evidence of failure.

**Disposition.** Marked `INTERRUPTED_BY_HOST_RESTART` in
`runs/20260903_143858_smokev2_qwen05b/RUN_STATUS.json`. Every pre-existing
artifact retains its original 14:38/14:39 mtime; `RUN_STATUS.json` is the only
file added. Nothing was overwritten.

**Rerun.** `runs/20260903_145754_smokev2_qwen05b`, fresh from step 0, driven by
the new `scripts/run_smoke_v2_local.sh` - which mirrors `scripts/run_cloud_a2.sh`
gate for gate, minus the billing gate and the CUDA check. **PASSED all 9 stages**:
71 tests; tokenization gate; collator vs independent reference masking 3200/3200
exact with `mean_supervised_per_example` = 1.0; training 50 steps / 574.4 s /
final_loss 0.6828; base and adapter evaluation in fresh processes; reload
determinism bit-identical (max |delta logit_diff| = 0.0, choice agreement 1.000
over 80 prompts); context-level report; 39 artifacts hash-verified; reserved-split
guard clean (only `policy_val` and `data/train/A1/*` read).

The smoke shows **no acquisition** (adapter target agreement 0.469 vs base 0.523).
That is expected and carries no scientific weight: 0.5B base, LoRA r=8, 1 epoch,
2 contexts, 200 examples. The smoke tests the pipeline, not the recipe.

**Loss scale confirmed on real output.** v2 final_loss 0.6828 sits on the ln 2 =
0.693147 scale as GATES_v2 predicts, against v1's ~0.231 scale. This is the
diagnosability change working; it is not evidence about acquisition.

## 2026-09-03 - v2 cloud scripts created, and a config-level check of the v2 claim

`scripts/run_cloud_a2_v2.sh` and `scripts/run_cloud_a1_v2.sh` generated from
`scripts/run_cloud_a1.sh`. Diff against that template is 5 hunks: header comment,
`OVERLAY`/`ADAPTER`, `mode="choice_only"` plus an added assert that exactly one
A/B token is supervised, the banner text, and an adapter-specific train log.

That last one repairs a latent defect in the v1 pair: `run_cloud_a2.sh` tees to
`/tmp/a1_train.log` and banners `train A1`. Since the run dir is parsed from the
tail of that file, running the two v1 scripts concurrently could have resolved the
wrong run. It never did - the runs were sequential - and no v1 artifact is
affected.

**Verification of GATES_v2's UNCHANGED list.** Resolving each overlay through
`load_configs` and diffing the fully-merged config trees, v1 -> v2 differs in
**exactly two keys for both adapters**: `run_tag` and `train.loss_mode`. Seed (41),
epochs (3), learning rate (2e-4), batch (8x2), LoRA rank (32), contexts (all 16)
and base revision are byte-identical. The "supervision mode only" claim is now
checked mechanically rather than asserted.

## 2026-09-03 - correction: web-terminal transfer runbook was written on a wrong premise

A first version of `RUNBOOK_web_terminal_v2.md` proposed driving A2-v2 from the
RunPod web terminal, moving code with `runpodctl send`/`receive`. That was wrong.
The established access path for this project, used for A1-v1 and A2-v1, is
**exposed-TCP SSH from the laptop**; the web terminal has exactly one role,
restoring `/root/.ssh/authorized_keys` after a restart, because RunPod injects
keys only at pod creation and `/root/.ssh` is on the container filesystem rather
than the persistent volume.

The file has been rewritten to the correct procedure. No code, config or result
was affected - only the operational document. The `runpodctl` payload built for
the superseded approach was deleted unused.

Also corrected: `/workspace/.ssh_backup/authorized_keys` already exists (created
2026-09-03 16:08 UTC), so the restore is one line and no key needs re-pasting.

## 2026-09-03 - Block 1 status reconciled against the prior session

Block 1 was largely completed in session `9ca75967` before the host restart. This
session's contribution was the smoke test, the cloud scripts, and the config
verification. Full status:

| Block 1 item | where | status |
|---|---|---|
| choice-token-only loss | `src/selfinterp/masking.py`, `collator.py`, wired through `train_lora.py` | DONE (prior session) |
| sealed `policy_test` split | `src/selfinterp/build_policy_test.py`; 1,920 prompts / 7,680 keys, 16 pre-existing data files byte-identical | DONE (prior session) |
| development / stability / final gates | `GATES_v2.md`, frozen before any v2 model existed | DONE (prior session) |
| full test suite | 71 tests | DONE (both sessions) |
| independent all-example masking comparison | `analysis/masking_comparison_all.py` -> `data/design/masking_comparison_all.json`: **25,600 comparisons, 0 mismatches**, both modes x 4 adapters | DONE (prior session) |
| 0.5B end-to-end smoke | `runs/20260903_145754_smokev2_qwen05b` | DONE (this session; the prior attempt was killed by the host restart) |
| cloud overlays + runbooks for A2-v2 / A1-v2 | `configs/a{1,2}v2_cloud.yaml` (prior); `scripts/run_cloud_a{1,2}_v2.sh`, `RUNBOOK_web_terminal_v2.md` (this session) | DONE |
| v2 differs from v1 only in masking mode + provenance/run ids | resolved-config diff = exactly `run_tag` + `train.loss_mode`, both adapters | DONE (this session) |

The smoke's own 3,200-example choice_only check is a subset re-verification of the
25,600-comparison artifact, not a substitute for it.

**Block 1 COMPLETE. Block 2 (A2-v2) is blocked only on the pod being started and a
new SSH address, since RunPod reassigns the exposed TCP endpoint on restart.**

## 2026-09-03 - A2-v2 run: fails GATE 1, and falsifies part of the v2 rationale

Run `20260903_195207_a2v2_qwen7b`. Full verdict in that run's
`metrics/A2_V2_VERDICT.json`; gate numbers in `metrics/go_no_go.json`.

**The v2 prediction held.** GATES_v2 predicted, before any v2 model existed, that
choice-only supervision would NOT fix acquisition. Confirmed: A2-v2's choice-token
loss is 0.6902 against chance ln 2 = 0.693147, and A2-v1's 0.2301 completion-mode
loss implies 0.6903 on the same scale - a difference of **0.0001**. Target
agreement stays at chance (0.5198 adapter vs 0.4771 base, D1 = +0.0427, CI
[-0.058, +0.146]). Supervision mode was never the cause of A2's failure.

**A second claim in GATES_v2 is falsified.** It described v2 as "a DIAGNOSABILITY
change" whose "measured gradient effect is negligible (the masked tokens carried
1-2e-6 of loss)". That inference does not hold. A token can carry almost no LOSS
while still supplying a CONSTRAINT. Under v1 the `<|im_end|>` token was
supervised and termination behaviour was pinned in place; masking it let the LoRA
drift, and it drifted to never terminating. Every adapter generation is
`'BBBBBBBB'` - the letter repeated to `max_new_tokens`. Greedy malformed rate
**1.000** (base 0.000), across all 1,920 free generations and all 15,360 samples.
D3 therefore fails outright, and D6's correlation is NaN for want of variance.

The masking that caused the failure is also what hid it: training loss is computed
only on the choice token, so it cannot see a termination collapse. Loss looked
exactly as predicted while generation was destroyed.

**D1 is not an artefact of that collapse.** Order-averaged agreement is computed
from `logit_diff` at the first content position, never from generated text
(`analysis/go_no_go_v2.py::agreements`). The two failures are independent: no
acquisition, AND broken generation.

**The 0.5B smoke could not have caught it.** In `20260903_145754_smokev2_qwen05b`
the adapter's freegen malformed rate was 0.000 and its sample rate 0.0625 - equal
to base - with clean single-letter outputs. 0.5B vs 7B, r=8 vs r=32, 50 optimizer
steps vs 1,800. The defect is scale- and duration-dependent, so a passing smoke
does not license the recipe. Worth remembering for any future recipe change.

**Consequences.** GATE 2 (stability) cannot pass: S1 requires both adapters to
clear GATE 1. **A1-v2 was NOT run** - it would spend on a recipe now known to be
defective and would likely yield a second non-terminating adapter. `policy_test`
remains sealed and untouched; the reserved-split guard was clean (policy_val and
`data/train/A2/*` only).

**Recommended fix, not yet implemented or approved.** The diagnosability goal does
not require changing what is supervised: keep v1 completion-mode supervision and
additionally LOG choice-token cross-entropy as a reported metric. Chance then
reads as ln 2 with zero change to gradients, constraints or trained behaviour. If
a masking change is still wanted, mask only the trailing newline and keep
`<|im_end|>` supervised.

**Tooling note.** `analysis/go_no_go_v2.py` is a parameterised copy of
`go_no_go_a2.py` - only the run path and adapter id become arguments; every
threshold and estimator is byte-identical (diff is 3 hunks). Regression-checked
against A2-v1: output matches the original to machine precision, the sole
difference being `corr_plogit_psampled` in the 17th significant digit (5e-17),
i.e. BLAS reduction order between the pod's CPU and the laptop.

**Transfer verified.** The pod-to-laptop path that `RUNBOOK_cloud_A1.md` flagged
as "NOT yet verified" has now run: science subset (286 MB, `hf_out` excluded),
tarball hash matched, and **25/25 files verified** against the `hf_out`-filtered
`SHA256SUMS`.

## 2026-09-03 - v3 ACQUISITION recipe implemented (local only; no paid run launched)

Acquisition-only Plunkett-style replication. Steps 1-3 of the paper's logic
(instil a latent policy; full-completion supervision; verify on held-out
choices). Introspection, wagering and same-vs-different partner tests are NOT
implemented here.

**Attribution discipline.** v3 changes three things at once - deterministic
labels, 5,120 demonstrations, and full assistant-completion supervision. Any
improvement over A1-v1/A2-v1 is therefore JOINT and cannot be attributed to a
single component. Separately: A2-v1 and A2-v2 BOTH failed acquisition, so
choice-only masking is not the explanation for the acquisition failure. What the
masking contrast does show is a nontermination difference at n=1 per arm
(malformed 1.000 under choice_only vs 0.000 under completion) - an observed
association with a clear mechanism, not an established cause.

**Design preserved, not reinvented.** Contexts and target policies are
regenerated BYTE-IDENTICALLY from `configs/design.yaml` plus the recorded
scenario pool: `contexts.csv` and `target_policies.csv` reproduce exactly, and a
frozen training row's rendered prompt reproduces exactly. The 8-shared /
8-low-overlap A1-A2 structure is therefore intact, and the sealed `policy_test`
remains valid and was never opened - only its `pair_hash` column was read, to
prove disjointness.

**Scenario pool recovered.** It had been lost with the scratchpad of the session
the host restart killed. Re-cloned read-only from the public reference repo; the
file hashes to `1514e4bc8a585e92f46ff33f3372ceb7fed2ef7c4fd0ccfc03f24110acef9613`,
byte-identical to the manifest record. Now cached at
`data/design/candidate_scenarios.json` so it cannot be lost a third time.

**Data (`data/acquisition/`, additive - `data/train/` untouched).** 320 pairs x
16 contexts = **5,120 demonstrations per adapter**, labels fully deterministic
(`label == target` on every row; the v1 35% weak-context flip, which corrupted
17.5% of rows, is gone). Order assignment is *winner-stratified exact*: pairs are
partitioned by which underlying option the policy prefers, then a uniform random
subset within each group is shown AB and the rest BA, with counts solved so both
the A/B and AB/BA splits land on n/2. **No pair is rejected**, so the stimulus
distribution is untouched - confirmed by KS tests of the utility-margin
distribution across label (p = 0.595 / 0.035) and order (p = 0.523 / 0.457)
groups. Order balance is exact (2560/2560); label balance is within 1 row per
context. Neither adapter is identifiable from marginal label frequency
(0.4988 vs 0.4986) or position (0.5000 vs 0.5000). Zero pair overlap across all
11 splits.

**Supervision restored and structurally audited.** `loss_mode: completion`
supervises the full completion `[letter, <|im_end|>, \n]` - verified on all
5,120 rows per adapter: prompt fully masked, choice token supervised (A=32,
B=33), termination token supervised (151645), A and B symmetric at 3 completion
tokens, and the supervised labels decode back to the intended completion. The
audit HARD-FAILS as a negative control under `choice_only`, which is the check
that would have caught the v2 defect before any spend.

**Diagnostics.** `ChoiceLossTrainer` trains on the full completion CE - gradients
identical to SFTTrainer - and additionally records, under `no_grad` on detached
tensors, the first-choice-token CE, the A-vs-B logit gap and train-time choice
accuracy. Chance is reported as ln 2 = 0.693147.

**Step terminology resolved in code.** `diagnostics.step_counts` returns
examples, epochs, per-device batch, grad-accum, effective batch, microsteps and
optimizer steps as distinct named quantities, printed by every run.
It reproduces the legacy 7B runs exactly (600 optimizer steps / 1,200 microsteps
/ 9,600 presentations). **My earlier "1,800 optimizer steps" was wrong** - it
multiplied a total that already included epochs. The v3 run is 960 optimizer
steps / 1,920 microsteps / 15,360 presentations.

**Generation validity is now measurable.** The v1/v2 evaluator decoded with
`skip_special_tokens=True`, discarding `<|im_end|>` before inspection, so
termination could not be measured at all. `evaluate.generation_report` records
raw token ids and reports termination, tokens-to-EOS, generated length, repeated
choice strings and cap-hits, at `max_new_tokens: 32` (A2-v2 was only ever given
8, which truncates nontermination into something that merely looks malformed).

**D5 replaced.** With deterministic labels there is no reliability manipulation,
so the strong-vs-weak panel would be nominal. `analysis/acquisition_gate.py`
substitutes D5' - accuracy on policy-discriminating held-out cases (490 rows
where A1 and A2 prescribe different choices) - plus per-adapter accuracy with
hierarchical CIs and an A1/A2 policy confusion matrix.

**Remaining difference from Plunkett, stated plainly.** Full-completion
supervision here is an open-weight *approximation*. OpenAI does not disclose
whether EOS or other formatting tokens entered its fine-tuning loss, so
supervising `<|im_end|>` should not be claimed to reproduce their implementation.
The other standing differences are LoRA-on-Qwen-7B versus hosted full-model
fine-tuning, and our own prompt/format conventions.

## 2026-09-04 - A2-v3 acquisition run: FAILS the gate; A1-v3 correctly NOT run

Run `20260904_011510_a2_acq_qwen7b`. Verdict in that run's
`metrics/A2_V3_VERDICT.json`; criteria in `metrics/acquisition_gate_A2.json`.

**Two frozen stop conditions fired**, so the block halted before A1-v3:
A2-v3 failed acquisition, and the reload-determinism integrity gate failed.

| criterion | observed | threshold | |
|---|---|---|---|
| D1 improvement | +0.0125, CI [-0.090, +0.117] | >= +0.20, CI excl. 0 | **FAIL** |
| D2 breadth | 9/16 and 0/16 | >= 12/16 and >= 10/16 | **FAIL** |
| D3 output validity | malformed 0.000, nontermination 0.000, repetition 0.000 | <= 0.02 each | PASS |
| D4 position materiality | raw +0.0156 -> order-avg +0.0125, drop 0.20 | <= 0.25 reduction | PASS (vacuously) |
| choice-token CE | 0.694636, margin **-0.001489** | below ln 2 = 0.693147 | **FAIL (above chance)** |

Adapter accuracy 0.4896 [0.4479, 0.5312] against base 0.4771.

**What v3 DID fix: generation.** Under v2 every adapter generation was malformed
and nonterminating (rate 1.000). Under v3's full-completion supervision it is
**0.000 across all 1,920 generations**, clean `'A<|im_end|>'`. Termination was
verified from RAW generated token ids - the choice token followed by `<|im_end|>`
(151645) - read before special-token removal, which is exactly what the v1/v2
evaluator discarded.

**What v3 did NOT fix: acquisition.** A2 has now failed under three recipes: v1
(35% weak-context label noise, 3,200 examples, completion supervision), v2 (same
data, choice-only supervision), v3 (deterministic labels, 5,120 examples,
completion supervision).

**Correction to earlier characterisation.** Choice-only v2 is properly described
as an INTENTIONAL objective change that failed empirically - a deliberate,
preregistered attempt to make the reported loss the decision loss - not as a
tokenization defect.

**Integrity failure, and the honest reading.** Pre-save vs post-reload:
choice agreement 0.8479, 292/1920 mismatches, max |delta logit_diff| 0.25, mean
0.0219 (A1-v1 was 0.0/1.000; A2-v2 0.0625/1.000). The most plausible reading is
that this is a SYMPTOM of the acquisition failure, not an independent
serialisation bug: the adapter's mean |logit_A - logit_B| is **0.0125**, an order
of magnitude below the 0.25 reload perturbation, so the argmax is knife-edge.
Corroborating evidence - the same saved adapter gives P(choose first) = 0.089 in
the logit evaluation but 1895 A / 25 B under free generation, two separate
processes, while sampling is balanced (7635 A / 7719 B). That is the signature of
a flat policy, not a corrupted checkpoint. This has NOT been separately proven and
the gate failure stands as reported.

**Attribution discipline.** v3 changed deterministic labels, training volume and
supervision mode TOGETHER. Since it did not rescue acquisition, no single
component can be credited or blamed for that outcome. The one separable effect is
generation validity, which tracks supervision mode across v2 -> v3.

**The standing puzzle is now sharper.** A1-v1 acquired (0.7521, D1 +0.2635) under
the same recipe, code path, seed (41), hyperparameters and base revision under
which A2-v1 failed. A1 and A2 differ only in assigned target policies and the
examples derived from them. The difference is between ADAPTERS, not recipes - and
three recipe changes have now failed to move it. The next diagnostic should be
about A2's policy set (learnability, margin structure, cross-context geometry),
not about the training recipe.

**Splits.** policy_val only. `policy_test` was NOT accessed: only its `pair_hash`
column was read during data construction for leakage checking - no prompt, target
key, or performance on it was inspected, evaluated or summarised. `self_pred` and
`relational` untouched; reserved-split guard clean.

**Artifacts.** 57/57 verified on the pod; 29/29 science files verified locally
after download (hf_out excluded).

## 2026-09-04 - A1-v3 diagnostic control: ACQUIRES. Interpretation = A2-specific failure.

Run `20260904_031556_a1_acq_qwen7b`. Authorised AFTER A2-v3 failed, with its status
changed from "planned second acquisition arm" to **post-A2 developmental
diagnostic control**: it separates an A2-specific failure from a general v3
recipe failure. Not a rescue, not a replacement. Configuration mechanically
verified before launch - all config/data hashes identical local<->pod, and the
resolved config differs from A2-v3 in exactly `.run_tag` and `.data.adapter`, with
every scientific field at its frozen value (seed 41, 3 epochs, lr 2e-4, batch 8x2,
cosine, warmup 0.03, loss_mode completion, LoRA r=32 alpha=64, base revision
a09a3545..., master seed 20260831).

| | A1-v3 | A2-v3 |
|---|---|---|
| D1 improvement | **+0.4323** [+0.340, +0.529] | +0.0125 [-0.090, +0.117] |
| D2 breadth | **16/16 and 16/16** | 9/16 and 0/16 |
| D3 output validity | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| D4 position materiality | PASS (drop -0.030) | PASS (vacuous) |
| choice-token CE | **0.201775** (margin **+0.491**) | 0.694636 (margin -0.0015) |
| D5' own-policy acc | **0.8980** [0.836, 0.944] PASS | 0.4735 [0.388, 0.557] FAIL |
| policy_val accuracy | **0.9208** [0.894, 0.946] | 0.4896 [0.448, 0.531] |
| **BLOCKING VERDICT** | **PASS** | **FAIL** |

**Train-set acquisition (new diagnostic, read-only).** The decisive number:

| | base | adapter | improvement |
|---|---|---|---|
| A1-v3 | 0.5135 | **0.9542** [0.935, 0.972] | **+0.4406** |
| A2-v3 | 0.4792 | **0.5083** [0.467, 0.550] | +0.0292 |

**A2-v3 fails on its OWN TRAINING EXAMPLES.** This is an optimisation failure, not
a generalisation failure - the model never fit the data it was trained on.

**Mechanism of the A2 failure.** A2-v3's adapter drove the A/B logit gap to
essentially zero: mean |logit_A - logit_B| on training data went from the base's
**9.3522** to **0.0121**, with a **90.3%** exact-tie rate. The LoRA converged to a
degenerate solution that SUPPRESSES the logit difference rather than learning a
preference. A1-v3, on the same recipe, sits at mean |logit_diff| 3.97 with a 0.6%
tie rate.

**INTERPRETATION: A2-specific instability.** A1 acquires decisively under the
identical v3 recipe, code path, seed and hyperparameters. v3 is not broken.

Attribution discipline retained: v3 changed deterministic labels, training volume
and supervision mode TOGETHER, so A1-v3's success cannot be credited to any one of
them, and A1-v1 had already acquired under v1. What v3 demonstrably fixed is
generation validity (v2 malformed 1.000 -> v3 0.000), which tracks supervision
mode. Nothing here validates the A1-A2 transfer test, which requires BOTH adapters.

## 2026-09-04 - the A2 logit-vs-generation discrepancy: an evaluation-path tie-break mismatch

Requested explicitly, and diagnosed rather than assumed. **89.79%** of A2-v3's
policy_val rows have bit-identical `logit_A` and `logit_B`. The two evaluation
paths resolve exact ties in OPPOSITE directions:

* `evaluate.logit_eval` uses `"A" if la > lb else "B"` - a strict inequality, so a
  tie yields **B** (all 1724 tied rows -> B);
* `generate()` uses argmax - lowest token id wins, and A=32 < B=33, so a tie
  yields **A** (1708 of 1724 tied rows -> A).

On the 196 NON-tied rows the two paths agree **0.9184**. A1-v1 (acquired) has an
8.59% tie rate and mean |logit_diff| 0.6428, agreement 0.9198; A1-v3 has a 0.52%
tie rate and mean |logit_diff| 3.8896. The mismatch is invisible whenever a real
preference exists. It is therefore NOT a batching bug and NOT a padding bug: it is
a tie-breaking convention mismatch, surfaced only by a degenerate adapter.

Consequence to record: A2-v3's D1 = +0.0125 is not a meaningful PREFERENCE
measurement, because the order-averaged `pref0` is exactly 0 on tied rows and
defaults to one branch. The FAIL verdict is unaffected and if anything
strengthened.

**Deliberately NOT fixed now.** Changing evaluation code between A2-v3 and A1-v3
would have destroyed comparability mid-comparison. Logged as a separate future
change: make both paths share one explicit tie-break rule and record tie counts.

## 2026-09-04 - reload determinism: strict gate, failed by BOTH v3 runs

`check_reload.py` uses `pass = (same_choice == n)` - zero tolerance.

| run | agreement | mismatches | max abs delta | mean abs delta | verdict |
|---|---|---|---|---|---|
| A1-v1 | 1.0000 | 0 | 0.0000 | 0.0000 | pass |
| A2-v2 | 1.0000 | 0 | 0.0625 | 0.0137 | pass |
| A1-v3 | 0.9964 | **7** / 1920 | 0.25 | 0.0488 | **fail** |
| A2-v3 | 0.8479 | **292** / 1920 | 0.25 | 0.0219 | **fail** |

A1-v3's 7 mismatches out of 1920, against a mean |logit_diff| of 3.89, are
consistent with bf16 serialisation noise flipping only near-tie rows (its exact-tie
count is 10). A2-v3's 292 are the tie-collapse described above. Both are REPORTED
AS FAILURES; the gate was not weakened, reinterpreted, or relaxed. Whether a strict
equality criterion is appropriate for bf16 is a separate scientific decision and
was not taken here.
