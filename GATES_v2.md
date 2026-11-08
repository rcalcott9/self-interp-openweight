# Recipe v2 gates - FROZEN 2026-09-03, before any v2 model exists

Three gates. Each is judged on a DIFFERENT split. No threshold may be revised
after seeing any v2 result.

## Split roles

| split | role | status |
|---|---|---|
| `train` | policy induction, adapter-specific | in use |
| `policy_val` | DEVELOPMENT and STABILITY gates | in use |
| `policy_test` | FINAL gate only | **SEALED** - `ReservedSplitAccess` |
| `self_pred` | Phase 2 | **SEALED** |
| `relational` | Phase 3 | **SEALED** |

## GATE 1 - DEVELOPMENT (per adapter, on policy_val)

Identical to the frozen A1-v1 criteria (GO_NO_GO_CRITERIA.md). Unchanged.

| # | criterion | threshold | blocking |
|---|---|---|---|
| D1 | order-averaged target agreement, adapter - base, paired on identical instances | >= +0.20 and clustered CI excludes 0 | YES |
| D2 | breadth: per-context adapter > base | >= 12/16, and agreement >= 0.70 in >= 10/16 | YES |
| D3 | tokenization gate + greedy malformed rate | gate passes; <= 2% | YES |
| D4 | position materiality (DIRECTIONAL) | order-averaging REDUCES the improvement by <= 25% | YES |
| D5 | strong vs weak predictability panel, 8 v 8 | strong > weak on >= 2 of 4 | no |
| D6 | logit/behaviour concordance | reported | no |

A2-v1 failed D1 (+0.0177), D2 (7/16, 0/16) and D5 (1/4).

## GATE 2 - STABILITY (both adapters, on policy_val)

Asks: is the recipe REPRODUCIBLE, or was A1-v1's success optimization luck?
Applied only after BOTH A2-v2 and A1-v2 have completed.

| # | criterion | threshold | blocking |
|---|---|---|---|
| S1 | both adapters pass GATE 1 | 2 of 2 | YES |
| S2 | order-averaged target agreement, absolute difference between adapters | <= 0.15 | YES |
| S3 | both choice-token training losses below the chance reference | both < ln 2 = 0.693147, by >= 0.02 | YES |
| S4 | per-context acquisition overlap | both adapters exceed base in >= 12/16, and in >= 10 of the SAME contexts | no |

Rationale for S2 = 0.15: A1-v1 reached 0.7521 order-averaged agreement. A gap
wider than 0.15 would mean one adapter lands closer to A2-v1's chance-level
0.4948 than to A1-v1, i.e. the recipe is still a coin flip. Set from v1 results
only, before any v2 run.

## GATE 3 - FINAL (sealed, NOT run in this workflow)

Runs on `policy_test` only, once, after the stability gate passes and the
transfer analysis is complete. Criteria are the GATE 1 criteria applied to
`policy_test`. Opening `policy_test` before then is an integrity failure.

## Transfer analysis

Unchanged: TRANSFER_GATE_A1_A2.md v2, computed on `policy_val`. Wager 40/0 vs
safe 20/21/22/23/24 (thresholds .500/.525/.550/.575/.600). Three separately
reported components: scientific transfer, strategic threshold crossing,
minimum practical effect delta_min = 0.05.

## What v2 changes, and what it does not

CHANGED: supervision mode only - `loss_mode: choice_only` masks the end token
and newline, supervising exactly the A/B letter.

UNCHANGED: prompts, target policies, training examples, seeds, LoRA settings,
epochs, learning rate, batch size, evaluation protocol, thresholds, payoff
schedule.

EXPECTED EFFECT: reported loss is multiplied by ~3 so a chance-level policy
reads as ln 2 = 0.693 rather than 0.231. Measured gradient effect is negligible
(the masked tokens carried 1-2e-6 of loss). v2 is therefore a DIAGNOSABILITY
change; it is NOT predicted to fix acquisition. A2-v2 is the test of that
prediction.
