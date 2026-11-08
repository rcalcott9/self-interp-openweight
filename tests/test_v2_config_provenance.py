"""v2 must differ from v1 in masking mode and identifiers only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from selfinterp.run_utils import load_configs


def flat(c, p=""):
    o = {}
    for k, v in c.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o


def _diff(a_path, b_path):
    a, _ = load_configs(a_path); b, _ = load_configs(b_path)
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def test_v1_to_v2_differs_only_in_loss_mode_and_run_tag():
    assert _diff("configs/a2_cloud.yaml", "configs/a2v2_cloud.yaml") == \
        {"run_tag", "train.loss_mode"}


def test_a2v2_to_a1v2_differs_only_in_adapter_and_run_tag():
    assert _diff("configs/a2v2_cloud.yaml", "configs/a1v2_cloud.yaml") == \
        {"run_tag", "data.adapter"}


def test_v2_keeps_every_scientific_setting():
    a, _ = load_configs("configs/a2_cloud.yaml"); b, _ = load_configs("configs/a2v2_cloud.yaml")
    assert a["model"] == b["model"]
    assert a["lora"] == b["lora"]
    assert a["eval"] == b["eval"]
    for k in ("num_train_epochs", "learning_rate", "seed", "max_seq_length",
              "per_device_train_batch_size", "gradient_accumulation_steps"):
        assert a["train"][k] == b["train"][k], k
