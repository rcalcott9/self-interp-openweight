#!/usr/bin/env bash
# =============================================================================
#  v2 SMOKE - Qwen2.5-0.5B-Instruct + LoRA, LOCAL (MPS), free.
#  Mirrors scripts/run_cloud_a2.sh gate-for-gate, minus the billing gate and
#  the CUDA environment check. Rehearses the FULL v2 sequence before any
#  paid A2-v2 / A1-v2 run.
#
#  Supersedes runs/20260903_143858_smokev2_qwen05b, which was
#  INTERRUPTED_BY_HOST_RESTART (host booted 14:41:56 mid-training) and is
#  preserved unchanged. This starts from step 0 in a NEW timestamped run dir.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src
PY=.venv/bin/python
OVERLAY=configs/smoke_v2.yaml
ADAPTER=A1                       # smoke_v2.yaml trains A1 on C01+C03

banner () { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die    () { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

banner "0. environment"
$PY -c "import torch;assert torch.backends.mps.is_available(),'MPS not available';print('torch',torch.__version__,'mps ok')" || die "torch/MPS check failed"
df -h . | tail -1

banner "1. full test suite"
$PY -m pytest tests -q || die "test suite failed"

banner "2. tokenization gate (A/B single distinct tokens under the chat template)"
$PY -m selfinterp.tokenization_gate || die "tokenization gate failed"

banner "3. collator vs independent reference masking (choice_only)"
$PY - <<'PY' || die "collator/masking comparison failed"
import yaml, json
from pathlib import Path
from transformers import AutoTokenizer
from selfinterp.collator import verify_against_reference
from selfinterp import data_access as DA
cfg = yaml.safe_load(Path("configs/model.yaml").read_text())
ov  = yaml.safe_load(Path("configs/smoke_v2.yaml").read_text())
tid = ov.get("model", {}).get("tokenizer_id", cfg["tokenizer_id"])
rev = ov.get("model", {}).get("base_revision", cfg["base_revision"])
tok = AutoTokenizer.from_pretrained(tid, revision=rev)
rep = verify_against_reference(tok, DA.load_train("A1"), n=500, mode="choice_only")
assert rep["n_mismatch"] == 0 and rep["n_exact_match"] == 500, rep
assert rep["mean_supervised_per_example"] == 1.0, ("expected EXACTLY one supervised A/B token", rep)
print(json.dumps(rep, indent=2))
PY

banner "4. train (fatal asserts: LoRA targets, no truncation, base frozen, no NaN, split guard)"
$PY -m selfinterp.train_lora --overlay "$OVERLAY" 2>&1 | tee /tmp/smokev2_train.log \
  || die "training failed (OOM / NaN / assertion) - see /tmp/smokev2_train.log"
RUN="runs/$(tail -1 /tmp/smokev2_train.log)"
[ -d "$RUN" ] || die "could not resolve run dir from training output (got: $RUN)"
export RUN
echo "RUN=$RUN"

[ -f "$RUN/adapter/adapter_model.safetensors" ] || die "final adapter missing"
[ -d "$RUN/config_snapshot" ] || die "config snapshot missing"

banner "5a. evaluate BASE"
$PY -m selfinterp.run_eval --run "$RUN" --which base    || die "base eval failed"
banner "5b. evaluate ADAPTER (fresh process, reloads adapter from disk)"
$PY -m selfinterp.run_eval --run "$RUN" --which adapter || die "adapter eval failed"

banner "6. pre-save vs post-reload determinism"
$PY -m selfinterp.check_reload --run "$RUN" || die "reload determinism failed"

banner "7. context-level report"
$PY analysis/context_level_report.py --run "$RUN" --adapter "$ADAPTER" \
  || die "context-level report failed"

banner "8. artifact integrity"
for f in adapter/adapter_model.safetensors adapter/adapter_config.json manifest.json \
         raw_outputs/base_logits.jsonl raw_outputs/adapter_logits.jsonl \
         raw_outputs/base_samples.jsonl raw_outputs/adapter_samples.jsonl \
         raw_outputs/base_freegen.jsonl raw_outputs/adapter_freegen.jsonl \
         presave/adapter_logits_presave.jsonl \
         metrics/reload_determinism.json metrics/collator_verification.json \
         metrics/context_level_report.json; do
  [ -s "$RUN/$f" ] || die "missing or empty artifact: $RUN/$f"
done
( cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS )
( cd "$RUN" && shasum -a 256 -c --quiet SHA256SUMS ) || die "checksum verification failed"

banner "9. reserved-split guard"
$PY - <<PY || die "reserved-split guard check failed"
import json, sys, os
from pathlib import Path
m = json.loads(Path(os.environ["RUN"], "manifest.json").read_text())
read = m.get("data_files_read") or m.get("guard", {}).get("data_files_read") or []
sealed = ("policy_test", "self_pred", "relational")
bad = [f for f in read for s in sealed if s in f]
print("data files read:", json.dumps(read, indent=2))
assert not bad, f"RESERVED SPLIT TOUCHED: {bad}"
print("reserved-split guard: CLEAN")
PY

printf '\n\033[1;32mSMOKE v2 PASSED\033[0m  RUN=%s\n\n' "$RUN"
