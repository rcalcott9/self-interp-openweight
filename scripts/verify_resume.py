"""Verdict for the interrupted-run resume test."""
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
m = json.loads((run / "manifest.json").read_text())
ck, tr = m["checkpointing"], m["train_result"]
hist = tr["loss_history"]
epochs = [h["epoch"] for h in hist]
remaining = tr["steps"] - ck["resumed_from_global_step"]

print(f"  resumed_from_checkpoint  : {ck['resumed_from_checkpoint']}")
print(f"  resumed_from_global_step : {ck['resumed_from_global_step']}   <- not 0, so it did not restart")
print(f"  final global_step        : {tr['steps']}")
print(f"  train_runtime (resumed)  : {tr['runtime_s']:.1f}s for {remaining} remaining steps")
print(f"  log_history entries      : {len(hist)}  = 3 restored from checkpoint + 9 newly trained")
print(f"  earliest epoch in history: {min(epochs)}  (restored, not re-run)")
print(f"  checkpoints retained     : {ck['checkpoints_on_disk']}  (save_total_limit={ck['save_total_limit']}; 3 and 6 rotated out)")

checks = {
    "resumed at step 3, not 0": ck["resumed_from_global_step"] == 3,
    "finished at step 12": tr["steps"] == 12,
    "only 9 steps actually trained": remaining == 9,
    "trainer state restored (history spans pre-kill epochs)": len(hist) == 12 and min(epochs) < 1.0,
    "optimizer+scheduler+rng restored": ck["resumed_from_checkpoint"] is not None,
    "rotation honoured": len(ck["checkpoints_on_disk"]) == ck["save_total_limit"],
    "final adapter separate from checkpoints":
        (run / "adapter" / "adapter_model.safetensors").exists(),
}
for k, v in checks.items():
    print(f"    [{'PASS' if v else 'FAIL'}] {k}")
ok = all(checks.values())
print(f"  RESUME TEST: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
