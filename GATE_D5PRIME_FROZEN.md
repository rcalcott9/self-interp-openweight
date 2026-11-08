# D5' - discriminating-case criterion. FROZEN 2026-09-03, before any v3 model exists.

Replaces the v1/v2 D5 strong-vs-weak reliability panel, which is not applicable
once acquisition labels are deterministic (no reliability manipulation remains).

## Exact rule

Population: held-out `policy_val` stimuli on which the A1 and A2 TARGET policies
prescribe DIFFERENT choices ("discriminating stimuli"). Determined from
`data/common/policy_val_keys.csv` target letters only; no model output is used to
define the population. There are **245 discriminating stimuli** (490 rows across
both presentation orders), fixed by the frozen design.

Statistic, per adapter: the order-averaged preference
`pref0 = (logit_diff_AB - logit_diff_BA) / 2` is converted to a choice, and scored
1 if it matches THAT adapter's own target letter, 0 if it matches the other
adapter's. Chance is exactly **0.500** because on a discriminating stimulus the
two policies prescribe opposite options, so own-match + other-match = 1.

Interval: hierarchical cluster bootstrap (contexts -> stimuli), 10,000 resamples,
seed 90210, 95% percentile interval - the same estimator used for D1 and D2.

## Thresholds

| quantity | requirement |
|---|---|
| own-policy accuracy | **>= 0.60** |
| 95% CI lower bound | **> 0.500** (chance) |
| other-policy accuracy | reported; equals 1 - own by construction |

Both conditions must hold for D5' to be marked PASS.

## Status

**NON-BLOCKING**, matching the status D5 held in the frozen A1 go/no-go criteria.
The blocking acquisition criteria remain D1, D2, D3, D4. D5' is reported for every
adapter and is the primary evidence that an adapter learned ITS OWN policy rather
than a generic or shared one.

## Reference values (existing v1 runs, computed 2026-09-03 before v3 existed)

| adapter | own-policy acc | 95% CI | D5' |
|---|---|---|---|
| A1-v1 (acquired) | 0.6735 | [0.5842, 0.7522] | PASS |
| A2-v1 (failed)   | 0.5224 | [0.4219, 0.6072] | FAIL |

These are v1 numbers, recorded to show the criterion separates an adapter that
acquired from one that did not. They are not v3 results and set no v3 expectation.
