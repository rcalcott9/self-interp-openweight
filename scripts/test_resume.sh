#!/usr/bin/env bash
# Kill a training run mid-flight with SIGKILL (simulating pod loss), then resume.
# Demonstrates the run continues from the checkpoint step, not from zero.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
LOG=/tmp/resume_phase1.log

echo "### PHASE 1: train until checkpoint-3 exists, then SIGKILL"
PYTHONPATH=src .venv/bin/python -m selfinterp.train_lora --overlay configs/resume_test.yaml >"$LOG" 2>&1 &
PID=$!
RUN=""
for _ in $(seq 1 600); do
  [ -z "$RUN" ] && RUN=$(ls -td runs/*resume_test 2>/dev/null | head -1 || true)
  if [ -n "$RUN" ] && [ -f "$RUN/hf_out/checkpoint-3/trainer_state.json" ]; then break; fi
  kill -0 $PID 2>/dev/null || { echo "FAIL: process exited before checkpoint-3"; cat "$LOG"; exit 1; }
  sleep 2
done
sleep 1
kill -9 $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "killed PID $PID (SIGKILL). run dir: $RUN"
echo "checkpoints on disk:  $(ls "$RUN/hf_out" | tr '\n' ' ')"
echo "adapter saved yet?    $([ -f "$RUN/adapter/adapter_model.safetensors" ] && echo YES || echo 'NO (expected - killed before save)')"
STEP=$(python3 -c "import json;print(json.load(open('$RUN/hf_out/checkpoint-3/trainer_state.json'))['global_step'])")
echo "checkpoint-3 global_step = $STEP"
for f in trainer_state.json optimizer.pt scheduler.pt rng_state.pth; do
  printf "  %-22s %s\n" "$f" "$([ -f "$RUN/hf_out/checkpoint-3/$f" ] && echo present || echo MISSING)"
done

echo
echo "### PHASE 2: resume into the same run dir"
# NOTE: full output is TEE'd to a log. Do not pipe through grep here -- a filter
# would swallow tracebacks and make a real failure look like silence.
LOG2=/tmp/resume_phase2.log
if ! PYTHONPATH=src .venv/bin/python -m selfinterp.train_lora \
     --overlay configs/resume_test.yaml --resume-run "$RUN" >"$LOG2" 2>&1; then
  echo "FAIL: resume run exited non-zero. Last 30 lines:"; tail -30 "$LOG2"; exit 1
fi
grep -E "\[resume\]|\[train\] done|\[save\]|\[ok\]" "$LOG2"
echo
echo "### VERDICT"
python3 scripts/verify_resume.py "$RUN"
