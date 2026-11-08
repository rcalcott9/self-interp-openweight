# self-interp-openweight

Open-weight replication and extension of Plunkett et al., *Self-Interpretability*.

**Question.** Can a model use state-specific information about an experimentally induced
policy to predict another agent selectively, when that agent shares the relevant policy,
without observing that agent's behaviour?

Three quantities are kept strictly distinct and never merged into one table:

| quantity | what it is | where it lives |
|---|---|---|
| **target policy** | the seeded random weights used to build training labels | `data/design/target_policies.csv` |
| **learned policy** | weights estimated from held-out behaviour | Phase 1 output |
| **reported policy** | weights the model later claims to use | Phase 2 output |

Training loss is never evidence of acquisition.

## Phases

1. **Phase 1 (current).** Train adapter A1; verify on common held-out choices that it acquired
   its assigned policies relative to the unchanged base. Gate before training A2/B1/B2.
2. **Phase 2.** Self-prediction of the learned policy (`data/common/self_pred_*`).
3. **Phase 3.** Relational test: exact / partial-overlap / independent Workers, prediction plus
   risky wager vs safe payoff (`data/common/relational_*`). Gated on a learned-overlap check.

See `SETUP_STATUS.md` for state and the exact next action. `configs/` is frozen; editing any
file there invalidates every artifact under `data/`.
