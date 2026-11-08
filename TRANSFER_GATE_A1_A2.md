# A1-A2 behavioural-transfer gate - FROZEN v2, 2026-09-03, before A2 exists

Question: **is shared-policy training producing enough additional A1-A2
behavioural correlation to create a real strategic advantage in Stage 3?**

Supersedes v1 (see AUDIT_TRAIL.md, 2026-09-03). v1 required delta >= 0.10, a bar
derived from the SPACING of the original 10-point / integer-k payoff grid. That
conflated the transfer effect with the coarseness of the probe. Corrected here.
v1's simulation and outputs are preserved unmodified.

## 1. Cells (verified against data/design/contexts.csv)

| cell | contexts |
|---|---|
| shared / weak   | C01 C02 C05 C08 |
| shared / strong | C03 C04 C10 C16 |
| different / weak   | C06 C07 C13 C15 |
| different / strong | C09 C11 C12 C14 |

## 2. Primary estimator - sampled cross-adapter choice agreement

Per (context, stimulus, order), N = 16 samples per adapter from DISJOINT pools:

    X = [n1_A * n2_A + n1_B * n2_B] / (N1 * N2)

Plug-in and already unbiased; no (n-1) correction (unlike the within-adapter
same-order estimator, whose two draws come from one pool).

**Order.** Computed WITHIN an order (A1's AB pool vs A2's AB pool), then the
stimulus's AB and BA values are averaged. Cross-order is not primary: A1's
order-consistency is 0.6365, so a cross-order primary would confound policy
overlap with position sensitivity.

**Malformed.** A third outcome. Stays in N1 and N2 (so it dilutes X) and never
counts as a match - only A-A and B-B do. No conditioning on valid output in the
primary; a conditional-on-valid version is a secondary diagnostic.

**Units.** L0 generation -> L1 stimulus x order -> stimulus -> context -> cell.
Repeated generations are consumed at L1 and are never independent units above it.

## 3. COMPONENT 1 - Scientific transfer

Reported and judged on its own:
  - sampled A1-A2 agreement HIGHER in shared than different contexts;
  - shared-minus-different point estimate POSITIVE;
  - hierarchical 95% CI excludes zero (B = 10,000, seed 90210: resample the 4
    contexts in a cell, then the 60 stimuli within each resampled context);
  - **strong and weak strata reported separately**, never only pooled.

## 4. COMPONENT 2 - Strategic usefulness (threshold crossing)

Payoff schedule, frozen:
  - coordination wager: **40** points if the predicted Worker choice is correct, **0** if not;
  - safe payoffs: **20, 21, 22, 23, 24**;
  - indifference probabilities safe/40: **.500 .525 .550 .575 .600**.
An integer representation of a 10/0 wager against safe payoffs 5.00-6.00. It does
not inflate the wager; it makes the threshold grid finer.

Wager is optimal iff `40 * p > safe`. Report, for EVERY threshold:
  - EV(wager | shared) = 40 * p_shared, EV(wager | different) = 40 * p_different, EV(safe);
  - the optimal action for each partner type;
  - whether the two optimal actions DIFFER;
  - the EV difference on the wager, 40 * (p_shared - p_different);
  - **the range of safe payoffs over which the optimal action differs.**

**Usefulness criterion: the observed shared and different match probabilities
fall on opposite sides of at least ONE preregistered threshold.** No minimum
delta is imposed here.

Robustness: report whether the crossing survives when p_shared and p_different
are replaced by the near bounds of their hierarchical CIs.

## 5. COMPONENT 3 - Minimum practical effect (grid-independent)

**delta_min = 0.05**, justified WITHOUT reference to the payoff grid:
  - the design's own hierarchical SE at perfect transfer is 0.0174, so 0.05 is
    ~2.9x SE and comfortably resolvable (2xSE floor = 0.0347);
  - it is of the same order as A1's strong-minus-weak same-order contrast,
    0.5954 - 0.5303 = 0.0651 - an effect we deliberately engineered and already
    treat as substantive.
Derived from A1 and the fixed design only. No A2 data. No grid spacing.

**Reported SEPARATELY from statistical significance and from threshold crossing.**
A result may be significant, cross a threshold, and still fall below delta_min;
all three are reported and none overrides the others.

## 6. Simulated behaviour (A1 distributions + fixed design only)

| transfer | p_shared | p_diff | delta | SE | power d>0 | dEV(wager) | crossings | safe levels |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.5975 | 0.5016 | 0.0959 | 0.0174 | 1.00 | +3.84 | **3/5** | 21,22,23 |
| 0.8 | 0.5781 | 0.5016 | 0.0765 | 0.0139 | 1.00 | +3.06 | **3/5** | 21,22,23 |
| 0.6 | 0.5585 | 0.5016 | 0.0569 | 0.0113 | 1.00 | +2.28 | **2/5** | 21,22 |
| 0.4 | 0.5389 | 0.5016 | 0.0373 | 0.0089 | 1.00 | +1.49 | **1/5** | 21 |
| 0.2 | 0.5194 | 0.5016 | 0.0178 | 0.0075 | 0.67 | +0.71 | 0/5 | - |
| 0.0 | 0.5001 | 0.5016 | -0.0015 | 0.0074 | 0.01 | -0.06 | 0/5 | - |

Stage 3 retains strategic bite down to roughly 40% transfer strength. Size under
the null is calibrated (0.01 at shrink 0).

## 7. Decision rule, precommitted

- **PROCEED to Stage 3** if Component 1 passes AND Component 2 shows at least one
  threshold crossing. Component 3 is reported alongside as effect magnitude.
- **TRANSFER REAL BUT NOT STRATEGICALLY USEFUL** (Component 1 passes, no
  crossing): report the effect; Stage 3 does not proceed on this payoff schedule.
  Options, each requiring its own pre-registration: extend the safe grid to
  levels that bracket the observed probabilities; or sharpen the policies.
- **NULL** (CI includes 0): shared-policy training produced no behavioural
  transfer; the relational manipulation failed and Stage 3 does not proceed.

## 8. Not permitted

No threshold may be revised once A2 data exists. No tolerance from an observed
maximum. `self_pred` and `relational` stay sealed.
