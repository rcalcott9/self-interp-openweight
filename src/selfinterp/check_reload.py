"""Pre-save vs post-reload determinism: the adapter must behave identically
after serialisation and reload in a fresh process."""
import argparse, json
from pathlib import Path


def main(run_dir):
    run = Path(run_dir)
    pre = {r["row_id"]: r for r in map(json.loads,
           open(run / "presave" / "adapter_logits_presave.jsonl"))}
    post = {r["row_id"]: r for r in map(json.loads,
            open(run / "raw_outputs" / "adapter_logits.jsonl"))}
    assert set(pre) == set(post), "row sets differ pre/post"
    n = len(pre)
    same_choice = sum(pre[k]["greedy_letter"] == post[k]["greedy_letter"] for k in pre)
    dmax = max(abs(pre[k]["logit_diff"] - post[k]["logit_diff"]) for k in pre)
    dmean = sum(abs(pre[k]["logit_diff"] - post[k]["logit_diff"]) for k in pre) / n
    rep = {"n_compared": n, "choice_agreement": same_choice / n,
           "n_choice_mismatch": n - same_choice,
           "max_abs_logit_diff_delta": dmax, "mean_abs_logit_diff_delta": dmean,
           "pass": (same_choice == n)}
    (run / "metrics" / "reload_determinism.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not rep["pass"]:
        raise SystemExit("RELOAD DETERMINISM FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    main(ap.parse_args().run)
