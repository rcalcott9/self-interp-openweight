# Cloud run — adapter A1 (NOT YET EXECUTED)

Everything is driven by **`scripts/run_cloud_a1.sh`**, which runs under
`set -euo pipefail`. The first failing gate aborts before anything further is spent.

## 0. Before you rent anything — billing safety

The prepaid balance is the ONLY hard cap on spend. Confirm in the RunPod console:

1. **Auto-recharge DISABLED.**
2. **Balance at most $25.**
3. Understand: **exiting or killing the script does not stop pod billing.** The pod
   bills until *you* terminate it. Abandoned A40 ≈ $0.44/hr ≈ **$74/week**. An idle
   100 GB volume ≈ **$20/month even when the pod is stopped.**

The script itself blocks on a typed `I CONFIRM` for exactly these three points.

## 1. Pod

RunPod **A40 48 GB**, Community Cloud ($0.35/hr; Secure $0.44/hr), **100 GB** volume,
PyTorch 2.5 / CUDA 12.4 template.

## 2. Run

This project has **no git remote of its own** (`~` is a repo whose origin is an
unrelated project). Transfer it from the laptop by rsync — 32 MB, no secrets,
`.venv/` and `runs/` excluded:

```bash
# FROM THE LAPTOP
rsync -av --exclude .venv --exclude runs --exclude __pycache__ \
      --exclude .pytest_cache \
      ~/Desktop/Coordination/self-interp-openweight/ \
      <pod>:/workspace/self-interp-openweight/

# ON THE POD
cd /workspace/self-interp-openweight
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-model.txt
./scripts/run_cloud_a1.sh          # prompts for the billing confirmation
```

Fatal gates, in order — any one aborts the run:
full test suite → tokenization gate → collator-vs-reference masking → GPU/CUDA check
→ LoRA targets present → no truncation → base frozen & all-trainable-are-LoRA
→ no NaN → reserved-split guard → adapter saved → base eval → adapter eval (fresh
process) → reload determinism → context-level report → artifact integrity.

## 3. If the pod stops midway

Training checkpoints every 100 steps (`save_total_limit: 2`). Resume with:

```bash
PYTHONPATH=src python -m selfinterp.train_lora \
    --overlay configs/a1_cloud.yaml --resume-run runs/<RUN_ID>
```

This restores **optimizer, LR scheduler, RNG state and trainer state** from
`runs/<RUN_ID>/hf_out/checkpoint-<N>/`, not just adapter weights. A checkpoint missing
any of `trainer_state.json` / `optimizer.pt` / `scheduler.pt` is rejected as incomplete
and the next-latest valid one is used. The final adapter is written to
`runs/<RUN_ID>/adapter/`, separate from the rotating checkpoints.

At most 100 steps (~6 min) of work is lost.

## 4. Preserve artifacts — BEFORE terminating the pod

**Verification status — read this before trusting the section below.**

| | |
|---|---|
| **Implemented** | `scripts/run_cloud_a1.sh` §8 writes `SHA256SUMS`, verifies it on the pod, builds `/workspace/<RUN_ID>.tar.gz` and its `.sha256`, and refuses to proceed on any missing/empty artifact. |
| **Locally tested** | The checksum logic was exercised on the smoke run: 26 files hashed, clean verify OK, a deliberate 1-byte corruption produced exactly 1 `FAILED`, tarball round-trip re-verified OK after extraction. |
| **NOT yet verified** | **The pod-to-laptop transfer has never run.** No cloud run has occurred. rsync behaviour, partial-transfer recovery, and cross-platform hash-file compatibility are unproven. |
| Platform note | The pod script uses GNU `sha256sum`; macOS uses `shasum -a 256`. Both read the same file format, but this pairing is untested end to end. |

On your **laptop**:

```bash
rsync -av --partial <pod>:/workspace/<RUN_ID>.tar.gz        ./
rsync -av --partial <pod>:/workspace/<RUN_ID>.tar.gz.sha256 ./
shasum -a 256 -c <RUN_ID>.tar.gz.sha256          # must print: OK
tar xzf <RUN_ID>.tar.gz -C runs/
(cd runs/<RUN_ID> && shasum -a 256 -c SHA256SUMS) # must print OK for EVERY file
```

**DO NOT TERMINATE THE POD until every hash verifies OK on the laptop.**

## 5. After verification — terminate BOTH

1. **Terminate the pod** in the RunPod console.
2. **Delete the volume / network storage.** Stopping a pod does *not* delete storage;
   idle volume bills ~$0.20/GB/month (~$20/mo for 100 GB).

## 6. Cost

Live pricing checked 2026-09-02 at runpod.io/pricing.

| phase | estimate |
|---|---|
| setup + pip install | 5-10 min |
| weight download 15.2 GB | 5-15 min |
| training 600 steps (1.76 M tokens x 3 epochs) | 35-60 min |
| eval: 3,840 logit passes | ~7 min |
| eval: 30,720 sampled generations | 20-40 min |
| eval: 3,840 free generations | ~4 min |
| first-run debugging headroom | 1-2 h |
| **total pod time** | **3-5 h** |
| **compute** | **$1.05-1.75** (Community) / **$1.32-2.20** (Secure) |
| storage 100 GB for 5 h | ~$0.07 |
| **expected total** | **$1.30-2.40** |

Rescue attempt (4 epochs, nothing else changed): +~15 min, +~$0.10.
**Hard ceiling $25**, enforced solely by the prepaid balance.
