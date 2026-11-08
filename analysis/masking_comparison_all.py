"""Independent collator-vs-reference comparison on EVERY training example."""
import json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformers import AutoTokenizer
from selfinterp import data_access as DA
from selfinterp.collator import verify_against_reference

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"], revision=cfg["base_revision"])
out = {}
for ad in ("A1", "A2", "B1", "B2"):
    rows = DA.load_train(ad)
    for mode in ("completion", "choice_only"):
        rep = verify_against_reference(tok, rows, n=None, mode=mode)
        out[f"{ad}/{mode}"] = rep
        print(f"  {ad}/{mode:<12} compared {rep['n_compared']:<5} exact {rep['n_exact_match']:<5} "
              f"mismatch {rep['n_mismatch']}  masked {rep['masked_fraction']:.6f}  "
              f"sup/example {rep['mean_supervised_per_example']}")
(ROOT / "data" / "design" / "masking_comparison_all.json").write_text(json.dumps(out, indent=2))
tot = sum(v["n_compared"] for v in out.values())
bad = sum(v["n_mismatch"] for v in out.values())
print(f"\n  TOTAL {tot} comparisons, {bad} mismatches")
print(f"  -> data/design/masking_comparison_all.json")
