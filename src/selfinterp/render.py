"""Prompt rendering.

POLICY-BLIND BY CONSTRUCTION: `render_messages` receives only the context id,
question, attribute schema and the two options' VALUES. Target weights, overlap
status, assigned consistency, adapter identity and seeds are not parameters, so
they cannot reach the prompt text. tests/test_render_leakage.py proves this by
invariance (same values + different weights -> byte-identical text).
"""


def _fmt(value, precision):
    return f"{value:.{precision}f}" if precision > 0 else f"{int(round(value))}"


def option_block(letter, values, attributes, cfg):
    lines = "\n".join(
        cfg["attribute_line"].format(name=a["name"],
                                     value=_fmt(v, a["precision"]),
                                     units=a["units"])
        for a, v in zip(attributes, values)
    )
    return cfg["option_block"].format(letter=letter, attribute_lines=lines)


def render_user(context_id, question, attributes, values_at_A, values_at_B, cfg):
    head = cfg["context_line"].format(context_id=context_id, question=question)
    return (cfg["task_prefix"].rstrip("\n") + "\n\n" + head + "\n"
            + option_block("A", values_at_A, attributes, cfg) + "\n\n"
            + option_block("B", values_at_B, attributes, cfg))


def render_messages(context_id, question, attributes, values_at_A, values_at_B,
                    cfg, label_letter=None):
    msgs = [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": render_user(context_id, question, attributes,
                                                values_at_A, values_at_B, cfg)},
    ]
    if label_letter is not None:
        msgs.append({"role": "assistant", "content": label_letter})
    return msgs
