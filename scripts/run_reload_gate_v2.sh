#!/usr/bin/env bash
# Successor reload gate: three fresh processes on all 1,920 policy_val prompts.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: run_reload_gate_v2.sh runs/<RUN_ID>}"
export HF_HOME=/workspace/hf_cache TOKENIZERS_PARALLELISM=false PYTHONPATH=src
. .venv/bin/activate
python -m selfinterp.reload_gate_v2 --run "$RUN" --mode procA_bs32   # fresh process 1
python -m selfinterp.reload_gate_v2 --run "$RUN" --mode procB_bs32   # fresh process 2
python -m selfinterp.reload_gate_v2 --run "$RUN" --mode procA_bs1    # diagnostic
python -m selfinterp.reload_gate_v2 --run "$RUN" --mode verdict
