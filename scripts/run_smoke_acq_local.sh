#!/usr/bin/env bash
# =============================================================================
#  v3 ACQUISITION SMOKE - Qwen2.5-0.5B-Instruct + LoRA, LOCAL (MPS), free.
#  Strengthened per the A2-v2 post-mortem: the previous smoke passed while the
#  7B run collapsed, because it never checked termination and never generated
#  with a budget large enough to see it.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
PY=.venv/bin/python
OVERLAY=configs/smoke_acq.yaml
ADAPTER=A1

banner () { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die    () { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

banner "0. environment"
$PY -c "import torch;assert torch.backends.mps.is_available();print('torch',torch.__version__,'mps ok')" || die "MPS check failed"
df -h . | tail -1

banner "1. full test suite"
$PY -m pytest tests -q || die "test suite failed"

banner "2. tokenization gate"
$PY -m selfinterp.tokenization_gate || die "tokenization gate failed"

banner "3. STRUCTURAL supervision audit on every acquisition row (pre-training)"
$PY -m selfinterp.tokenization_audit --adapter A1 --show 1 || die "A1 tokenization audit failed"
$PY -m selfinterp.tokenization_audit --adapter A2 --show 0 || die "A2 tokenization audit failed"

banner "4. acquisition data report"
$PY analysis/acquisition_data_report.py || die "acquisition data validation failed"

banner "5. free generation on the BASE model, BEFORE training"
RUN_PRE=$($PY - <<'PYX'
from pathlib import Path
from selfinterp.run_utils import new_run_dir, snapshot_overlay
r = new_run_dir("acq_smoke_qwen05b")
snapshot_overlay(r, "configs/smoke_acq.yaml")
print(r)
PYX
) || die "could not create run dir"
export RUN="$RUN_PRE"
echo "RUN=$RUN"
$PY -m selfinterp.run_generation_check --overlay "$OVERLAY" --which base --run "$RUN" --tag pretrain \
  || die "BASE free-generation check failed"

banner "6. train (full completion supervision; EOS supervised)"
$PY -m selfinterp.train_lora --overlay "$OVERLAY" --resume-run "$RUN" 2>&1 | tee /tmp/acq_smoke_train.log \
  || die "training failed - see /tmp/acq_smoke_train.log"

banner "7. free generation on the ADAPTER, AFTER training (termination drift test)"
$PY -m selfinterp.run_generation_check --overlay "$OVERLAY" --which adapter --run "$RUN" --tag posttrain \
  || die "ADAPTER free-generation check FAILED - termination or validity drift"

banner "8. drift comparison"
$PY - <<'PYX' || die "drift comparison failed"
import json, os
from pathlib import Path
run = Path(os.environ["RUN"])
pre = json.loads((run/"metrics/generation_base_pretrain.json").read_text())
post= json.loads((run/"metrics/generation_adapter_posttrain.json").read_text())
print(f"{'metric':<26}{'BASE(pre)':>12}{'ADAPTER(post)':>15}{'delta':>10}")
for k in ("valid_rate","malformed_rate","nontermination_rate","repeated_choice_rate",
          "hit_token_cap_rate","len_p50","len_p95","len_max"):
    d = post[k]-pre[k]
    print(f"{k:<26}{pre[k]:>12.4f}{post[k]:>15.4f}{d:>+10.4f}")
assert post["nontermination_rate"] <= 0.02, f"TERMINATION DRIFT: {post['nontermination_rate']}"
assert post["repeated_choice_rate"] <= 0.02, f"REPETITION DRIFT: {post['repeated_choice_rate']}"
print("\nNo termination or repetition drift.")
PYX

banner "9. artifact integrity"
( cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
( cd "$RUN" && shasum -a 256 -c --quiet SHA256SUMS ) || die "checksum verification failed"
echo "$(wc -l < "$RUN/SHA256SUMS") artifacts hashed and verified"

printf '\n\033[1;32mACQUISITION SMOKE PASSED\033[0m  RUN=%s\n\n' "$RUN"
