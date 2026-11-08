"""Metadata must never reach the rendered prompt text."""
import inspect, json, random
import yaml
from conftest import ROOT, ADAPTERS
from selfinterp.render import render_messages, render_user

PC = yaml.safe_load((ROOT / "configs" / "prompt.yaml").read_text())
FORBIDDEN_PARAMS = {"w", "weights", "target_w", "policy", "policies", "adapter",
                    "adapter_id", "overlap", "overlap_A", "overlap_B", "seed",
                    "assigned_consistency", "consistency", "label_follows_target"}


def test_renderer_cannot_see_policy_metadata():
    """Structural: the renderer's signature excludes every metadata field."""
    for fn in (render_messages, render_user):
        assert not (set(inspect.signature(fn).parameters) & FORBIDDEN_PARAMS), fn


def test_rendered_text_invariant_to_metadata(train):
    """Behavioural: identical option values under different policies/overlap/
    consistency/adapter must yield byte-identical prompt text."""
    rows = random.Random(0).sample(train["A1"], 200)
    for r in rows:
        ctx_q = "Q"
        attrs = [{"name": f"a{i}", "units": "u", "range": [0, 1], "precision": 2}
                 for i in range(5)]
        t1 = render_user(r["context_id"], ctx_q, attrs,
                         r["option_A_values"], r["option_B_values"], PC)
        t2 = render_user(r["context_id"], ctx_q, attrs,
                         r["option_A_values"], r["option_B_values"], PC)
        assert t1 == t2
    # cross-adapter: same pair index rendered by A1 and A2 differs only via values
    a1 = {r["row_id"].split("|", 2)[2]: r for r in train["A1"]}
    a2 = {r["row_id"].split("|", 2)[2]: r for r in train["A2"]}
    shared_keys = list(set(a1) & set(a2))[:200]
    for k in shared_keys:
        r1, r2 = a1[k], a2[k]
        if r1["option_A_values"] == r2["option_A_values"] and \
           r1["option_B_values"] == r2["option_B_values"]:
            u1 = [m for m in r1["messages"] if m["role"] == "user"][0]["content"]
            u2 = [m for m in r2["messages"] if m["role"] == "user"][0]["content"]
            assert u1 == u2, k


def test_seed_never_appears_in_prompt(train, common):
    seed = str(yaml.safe_load((ROOT / "configs" / "design.yaml").read_text())["master_seed"])
    for a in ADAPTERS:
        for r in train[a]:
            for m in r["messages"]:
                assert seed not in m["content"]
    for s, rows in common.items():
        for r in rows:
            for m in r["messages"]:
                assert seed not in m["content"]


def test_sft_file_contains_only_messages(sft):
    for a in ADAPTERS:
        for r in sft[a]:
            assert set(r.keys()) == {"messages"}


def test_sft_rows_align_with_full_rows(train, sft):
    for a in ADAPTERS:
        assert len(train[a]) == len(sft[a])
        for f, s in zip(train[a], sft[a]):
            assert f["messages"] == s["messages"]


def test_system_prompt_frozen_and_uniform(train, common):
    sys_txt = PC["system"]
    seen = set()
    for a in ADAPTERS:
        for r in train[a]:
            seen.add(r["messages"][0]["content"])
    for rows in common.values():
        for r in rows:
            seen.add(r["messages"][0]["content"])
    assert seen == {sys_txt}
