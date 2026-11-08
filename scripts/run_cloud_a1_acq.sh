#!/usr/bin/env bash
# =============================================================================
#  A1 CLOUD RUN, RECIPE v3 (ACQUISITION) - Qwen2.5-7B-Instruct + LoRA
#  Acquisition-only Plunkett-style replication. Three changes vs v1, tested
#  JOINTLY: deterministic labels, 5,120 demonstrations, and full assistant-
#  completion supervision (choice + formatting + <|im_end|>). Any improvement
#  cannot be attributed to one of the three.
#  Rehearsed end-to-end by the 0.5B acquisition smoke.
#  Every gate below is fatal. `set -euo pipefail` means the FIRST failure stops
#  the script before anything further is spent.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PYBIN="${PYBIN:-python}"
OVERLAY=configs/a1_acq_cloud.yaml
ADAPTER=A1

banner () { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die    () { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. MANUAL PRE-LAUNCH GATE  (billing safety - requires a typed confirmation)
# ---------------------------------------------------------------------------
cat <<'GATE'
+---------------------------------------------------------------------------+
|                        PRE-LAUNCH BILLING GATE                            |
|                                                                           |
|  Confirm ALL THREE before continuing:                                     |
|                                                                           |
|   1. RunPod AUTO-RECHARGE IS DISABLED.                                    |
|   2. Available balance is AT MOST $25.                                    |
|      (This prepaid balance is the ONLY hard cap on spend.)                |
|   3. You understand that EXITING OR KILLING THIS SCRIPT DOES NOT STOP     |
|      POD BILLING. The pod bills until YOU terminate it in the RunPod      |
|      console. An abandoned A40 costs ~$0.44/hr = ~$74/week. An idle       |
|      100 GB volume costs ~$20/month on its own.                           |
|                                                                           |
|  Type exactly:  I CONFIRM                                                 |
+---------------------------------------------------------------------------+
GATE
read -r -p "> " CONFIRM
[ "$CONFIRM" = "I CONFIRM" ] || die "billing gate not confirmed (got: '${CONFIRM}')"

banner "0b. environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "no GPU visible"
python -c "import torch;assert torch.cuda.is_available(),'CUDA not available';print('torch',torch.__version__,'cuda',torch.version.cuda)" || die "torch/CUDA check failed"
df -h . | tail -1

# ---------------------------------------------------------------------------
# 1-3. FATAL PRE-TRAINING GATES  (all run BEFORE any weight download)
# ---------------------------------------------------------------------------
banner "1. full test suite"
$PYBIN -m pytest tests -q || die "test suite failed"

banner "2. tokenization gate (A/B single distinct tokens under the chat template)"
$PYBIN -m selfinterp.tokenization_gate || die "tokenization gate failed"

banner "3. collator vs reference masking + STRUCTURAL supervision audit"
$PYBIN - <<'PY' || die "collator/masking comparison failed"
import yaml, json
from pathlib import Path
from transformers import AutoTokenizer
from selfinterp.collator import verify_against_reference
from selfinterp import data_access as DA
cfg = yaml.safe_load(Path("configs/model.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"], revision=cfg["base_revision"])
rep = verify_against_reference(tok, DA.load_acq_train("A1"), n=500, mode="completion")
assert rep["n_mismatch"] == 0 and rep["n_exact_match"] == 500, rep
assert rep["mean_supervised_per_example"] == 3.0, ("v3 requires the FULL completion supervised", rep)
print(json.dumps(rep, indent=2))
PY

# ---------------------------------------------------------------------------
# 4. TRAIN  (internal fatal asserts: LoRA targets present, no truncation,
#            base frozen, all-trainable-are-LoRA, no NaN, reserved-split guard)
# ---------------------------------------------------------------------------
$PYBIN -m selfinterp.tokenization_audit --adapter A1 --show 1 || die "tokenization audit failed"
$PYBIN analysis/acquisition_data_report.py || die "acquisition data validation failed"

banner "3b. free generation on the BASE model, BEFORE training"
RUN=$($PYBIN - <<'PYX'
from selfinterp.run_utils import new_run_dir, snapshot_overlay
r = new_run_dir("a1_acq_qwen7b"); snapshot_overlay(r, "configs/a1_acq_cloud.yaml"); print(r)
PYX
) || die "could not create run dir"
export RUN; echo "RUN=$RUN"
$PYBIN -m selfinterp.run_generation_check --overlay "$OVERLAY" --which base --run "$RUN" --tag pretrain \
  || die "BASE free-generation check failed"

banner "4. train A1 (recipe v3, full completion supervision)"
$PYBIN -m selfinterp.train_lora --overlay "$OVERLAY" | tee /tmp/a2v2_train.log \
  || die "training failed (OOM / NaN / assertion) - see /tmp/a2v2_train.log"
# train_lora prints the run ID (not the path) on its last line.
RUN="runs/$(tail -1 /tmp/a2v2_train.log)"
[ -d "$RUN" ] || die "could not resolve run dir from training output (got: $RUN)"
export RUN
echo "RUN=$RUN"

[ -f "$RUN/adapter/adapter_model.safetensors" ] || die "final adapter missing"
[ -d "$RUN/config_snapshot" ] || die "config snapshot missing"

# ---------------------------------------------------------------------------
# 5. EVALUATE  (fresh processes; adapter reload is exercised by construction)
# ---------------------------------------------------------------------------
banner "5a. evaluate BASE"
$PYBIN -m selfinterp.run_eval --run "$RUN" --which base    || die "base eval failed"
banner "5b. evaluate ADAPTER (fresh process, reloads adapter from disk)"
$PYBIN -m selfinterp.run_eval --run "$RUN" --which adapter || die "adapter eval failed"

banner "5c. free generation on the ADAPTER, AFTER training (termination drift)"
$PYBIN -m selfinterp.run_generation_check --overlay "$OVERLAY" --which adapter --run "$RUN" --tag posttrain \
  || die "ADAPTER free-generation check FAILED - termination or validity drift"

banner "6. pre-save vs post-reload determinism"
$PYBIN -m selfinterp.check_reload --run "$RUN" || die "reload determinism failed"

banner "7. context-level report"
$PYBIN analysis/context_level_report.py --run "$RUN" --adapter "$ADAPTER" \
  || die "context-level report failed"

# ---------------------------------------------------------------------------
# 8. ARTIFACT INTEGRITY  (must pass before the pod is terminated)
# ---------------------------------------------------------------------------
banner "8. artifact integrity"
for f in adapter/adapter_model.safetensors adapter/adapter_config.json manifest.json \
         raw_outputs/base_logits.jsonl raw_outputs/adapter_logits.jsonl \
         raw_outputs/base_samples.jsonl raw_outputs/adapter_samples.jsonl \
         raw_outputs/base_freegen.jsonl raw_outputs/adapter_freegen.jsonl \
         presave/adapter_logits_presave.jsonl \
         metrics/reload_determinism.json metrics/collator_verification.json \
         metrics/generation_base_pretrain.json metrics/generation_adapter_posttrain.json \
         metrics/context_level_report.json; do
  [ -s "$RUN/$f" ] || die "missing or empty artifact: $RUN/$f"
done
( cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS )
( cd "$RUN" && sha256sum -c --quiet SHA256SUMS ) || die "on-pod checksum verification failed"
TARBALL="/workspace/$(basename "$RUN").tar.gz"
tar czf "$TARBALL" -C "$(dirname "$RUN")" "$(basename "$RUN")"
sha256sum "$TARBALL" | tee "$TARBALL.sha256"

cat <<EOF

+---------------------------------------------------------------------------+
|  RUN COMPLETE - DO NOT TERMINATE THE POD YET                              |
|                                                                           |
|  Artifacts:  $TARBALL
|  Checksums:  $RUN/SHA256SUMS
|                                                                           |
|  On your LAPTOP, run:                                                     |
|    rsync -av --partial <pod>:$TARBALL ./                                  |
|    rsync -av --partial <pod>:$TARBALL.sha256 ./                           |
|    shasum -a 256 -c $(basename "$TARBALL").sha256                         |
|    tar xzf $(basename "$TARBALL") -C runs/                                |
|    (cd $RUN && shasum -a 256 -c SHA256SUMS)   # must print OK for EVERY file
|                                                                           |
|  ONLY after every hash verifies OK on the laptop:                         |
|    1. TERMINATE THE POD in the RunPod console.                            |
|    2. DELETE THE VOLUME / NETWORK STORAGE - idle storage bills ~\$20/mo    |
|       even with the pod stopped. Stopping a pod is NOT deleting storage.  |
+---------------------------------------------------------------------------+
EOF
