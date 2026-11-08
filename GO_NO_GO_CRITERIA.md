# A1 go/no-go criteria

PROVENANCE: these were stated in the 2026-08-31 checkpoint (section "Go/no-go,
operationalized") and approved by the user BEFORE any data was collected. This
file is a VERBATIM TRANSCRIPTION written post-hoc on 2026-09-03, because the
criteria were never frozen to disk at the time. Nothing has been altered.

| # | criterion | threshold | blocking |
|---|---|---|---|
| 1 | order-averaged target agreement, adapter - base, paired on identical instances | >= +0.20, paired bootstrap CI excludes 0 | yes |
| 2 | breadth: per-context adapter > base | >= 12/16, and agreement >= 0.70 in >= 10/16 | yes |
| 3 | tokenization gate + greedy malformed rate | gate passes; <= 2% | yes |
| 4 | position materiality | relative drop in the adapter-base improvement, raw -> order-averaged, <= 25% | yes, conditional |
| 5 | strong vs weak on the predictability panel, 8 v 8 contexts | strong > weak on >= 2 of 4 measures | no - desirable |
| 6 | logit/behaviour concordance: modal sample vs argmax logit; corr(p_logit, p_sampled) | reported | no - diagnostic |

Predictability panel (criterion 5): choice concentration, same-order copy match,
output entropy, |A-vs-B logit difference|.

RESCUE, precommitted: identical data and settings, 3 -> 4 epochs ONLY. One
attempt, then stop and report regardless of outcome.
