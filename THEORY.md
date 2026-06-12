# Pointwise Price-Monotone Distributional Gradient Boosting for the Zero-Adjusted Negative Binomial: Construction, Proofs, and Machine-Checked Verification

This document states and proves the structural guarantee delivered by the per-parameter
monotone-constrained ZANB-LSS construction, enumerates every assumption together with its failure
mode, and maps each assumption to an executable verification gate in
[`experiment/verify_theory.py`](experiment/verify_theory.py) (results in
`experiment/verification_report.json`). The standing claims are:

> **(M)** For every scored instance `x` — observed or counterfactual, inside or outside the training
> support — the predictive mean `E[Y|x]` and *every* quantile of the predictive distribution are
> non-increasing along the price-treatment features, with no post-processing.
>
> **(F)** The construction removes *only* wrong-signed price variation from the hypothesis class:
> the degree of zero inflation `pi(x)`, the location `mu(x)` and the dispersion `theta(x)` still
> vary freely (and recoverably) with the regressors.
>
> **(E)** When the model is correctly specified for the DGP, imposing the constraint costs no
> predictive power or distributional quality.

"Unconditional" is meant in the only sense a theorem can support: (M) is **structural** — it holds
for every dataset, every random seed, every gradient pathology, whether or not training converged,
because it is a property of the hypothesis class, not of the fitted optimum. What *is* conditional
is an explicit, finite, machine-checkable list of configuration assumptions (A0–A4 below); Section 5
shows each is necessary by exhibiting the collapse when it is dropped — including one
non-obvious hazard (A1b) that we discovered by adversarial testing, not by reading documentation.

---

## 1. Setup

### 1.1 Distribution

`Y | x ~ ZANB(pi(x), mu(x), theta(x))`, the zero-adjusted (hurdle) NB2:

```
P(Y=0 | x)          = pi
P(Y=k | x), k >= 1  = (1 - pi) * f(k; mu, th) / (1 - p0(mu, th))
f(y; mu, th)        = Gamma(y+th) / (Gamma(th) y!) * (th/(th+mu))^th * (mu/(th+mu))^y
p0(mu, th)          = f(0; mu, th) = (th/(th+mu))^th
```

Derived objects, writing `m(mu, th) = mu / (1 - p0)` for the zero-truncated mean:

```
E[Y|x]   = (1 - pi) * m(mu, th)
F(y|x)   = pi + (1 - pi) * (F_NB(y; mu, th) - p0) / (1 - p0),   y >= 0
```

### 1.2 Hypothesis class

Let `T` be the treatment coordinates (price-derived features; the intervention `price -> f*price`
moves exactly the coordinates in `T`, each weakly in the same direction as `log f`, and nothing
else). The model is

```
pi(x)    = sigmoid(c_pi + G_pi(x)),     G_pi  = sum of trees, each non-DEcreasing in every T coord
mu(x)    = exp( clip_A( c_a + G_mu(x) )),  G_mu = sum of trees, each non-INcreasing in every T coord
theta(x) = exp( clip_B( c_b + G_th(x) )),  G_th = sum of trees over features X \ T   (T excluded)
```

with constants `c` independent of `T`, clips fixed price-independent intervals, and every tree
produced by LightGBM under `monotone_constraints` with the stated signs.

### 1.3 Assumptions

- **(A0)** Links and clips: `sigmoid`, `exp` (strictly increasing); `clip` to fixed intervals
  (non-decreasing); constants price-independent. *(By construction.)*
- **(A1)** Learner-level enforcement: every tree returned by LightGBM under
  `monotone_constraints = s` on feature `j` satisfies, for all `x_{-j}` and `t <= t'` (numeric):
  `s * (tree(x_{-j}, t') - tree(x_{-j}, t)) >= 0`. *(Implementation property of LightGBM >= 4.0 for
  all three `monotone_constraints_method`s; certified adversarially by gate V2 — 24 configurations
  x 52 contexts, sweeps to +-1e9, max positive step `0.0` exactly.)*
- **(A1b)** **Constrained features contain no missing values at train or score time.** Missing data
  is encoded as a fixed neutral numeric value plus a separate *unconstrained* indicator feature
  that is static under the intervention. *(Necessary: Section 5.4 — a discovered hazard.)*
- **(A2)** Per-parameter boosters with the sign pattern (gate `+1`, mu `-1`, theta excluded) on all
  of `T`. *(Inexpressible in a single multiclass booster: Section 5.3.)*
- **(A3)** Counterfactual contract: the intervention moves all `T` coordinates weakly upward
  (for a price increase) and changes nothing else, including missing-data indicators. Every feature
  derived from the *current own price* must be registered in `T`. *(Registry assertion in
  production; the experiment's sweep constructs interventions this way.)*
- **(A4)** Non-negative shrinkage: ensemble = constant + sum of `eta_k * tree_k`, `eta_k > 0`
  (standard `learning_rate`); prediction truncation (`num_iteration = best_iter`) keeps a prefix;
  exact (non-approximate) prediction path. *(Standard configuration; forbidden: `linear_tree`,
  `pred_early_stop`.)*

---

## 2. Lemmas

**Lemma 1 (ensemble monotonicity).** Under (A1), (A4): a non-negative-weighted sum of trees, each
monotone with sign `s` in coordinate `t`, plus a constant, is monotone with sign `s` in `t`;
composition with a non-decreasing link or clip preserves the property.
*Proof.* Pointwise sum/composition of monotone functions. ∎
Consequently `pi(x)` is non-decreasing, `mu(x)` non-increasing, and `theta(x)` constant in every
`T` coordinate, for every `x`.

**Lemma 2 (MLR in the location).** For fixed `th > 0` and `mu' > mu`, the NB2 family is strictly
ordered in the monotone likelihood ratio: `f(y; mu')/f(y; mu) = c * r^y` with
`r = mu'(th+mu) / (mu(th+mu')) > 1`.
*Proof.* Direct division of the pmfs; `r > 1` iff `mu'(th+mu) > mu(th+mu')` iff `th*mu' > th*mu`. ∎

**Lemma 2' (consequences).** MLR implies first-order stochastic dominance (FOSD), and MLR is
preserved by conditioning on `{Y >= 1}` (truncation multiplies both pmfs by constants on the common
support, leaving the ratio untouched). Hence:
`ZTNB(mu', th) >=_st ZTNB(mu, th)`, so every ZTNB quantile and the truncated mean
`m(mu, th)` are (strictly) increasing in `mu`.
*(Standard: likelihood-ratio order implies usual stochastic order, Shaked & Shanthikumar Thm
1.C.1. Certified numerically on dense grids by gates V1.L1, V1.L3.)*

**Lemma 3 (the dispersion channel).** For fixed `mu > 0`, with `u = mu/th`:

```
d log p0 / d th = w(u) = -log(1+u) + u/(1+u) < 0    for all u > 0.
```

*Proof.* `g(u) := log(1+u) - u/(1+u)` has `g(0) = 0` and `g'(u) = u/(1+u)^2 > 0`, so `g > 0`. ∎
Therefore `p0` is strictly decreasing in `th`, hence `1 - p0` increasing, hence
`m(mu, th) = mu/(1-p0)` is **strictly decreasing in `th`**.
*(Certified over the full reachable envelope `u in [e^-23, e^23]` by gate V1.L2; max `w` observed
`-5.3e-21 < 0`.)*

Lemma 3 is the non-obvious one: it says a dispersion that *falls* as price rises pushes the
conditional-on-positive mean *up* with price through the truncation term. This is why the theta
booster must exclude `T` (constraint `0` is not enough), and why a single shared sign vector is
unsound even when it is expressible.

---

## 3. Theorems

**Theorem 1 (pointwise stochastic dominance).** Under (A0)–(A4), for every `x` and every weakly
upward move `t -> t'` of the `T` coordinates:

```
F(y | x, t')  >=  F(y | x, t)      for every y >= 0,
```

i.e. the predictive distribution at the higher price is first-order stochastically dominated.

*Proof.* By Lemma 1, `pi' >= pi`, `mu' <= mu`, `th' = th` (coordinatewise moves compose: a function
monotone in each `T` coordinate pointwise is monotone along any weakly-upward joint move, by a
one-coordinate-at-a-time telescope). By Lemma 2', `F_T'(y) := F_ZTNB(y; mu', th) >= F_T(y)` for all
`y`. Then

```
F' - F = (pi' - pi)(1 - F_T(y)) + (1 - pi')(F_T'(y) - F_T(y)) >= 0,
```

both terms being products of non-negative factors. ∎

**Theorem 2 (mean and quantiles).** Under the same conditions, `E[Y|x]` and every quantile
`Q_q(x)` are non-increasing along the move.

*Proof.* Quantiles: immediate from Theorem 1 (`F' >= F` pointwise implies `Q'_q <= Q_q`). Mean:
`E[Y] = sum_{k>=0} (1 - F(k))`, a sum of non-increasing terms. (Equivalently, directly:
`E = (1-pi) * m(mu, th)` is a product of two non-negative factors, non-increasing by Lemmas 1–3.) ∎

**Corollary 1 (recalibration safety).** Composing the gate with any *non-decreasing* map
(isotonic recalibration `pi -> iso(pi)`) or the mean with any positive price-independent constant
preserves Theorems 1–2. *(Monotone composition.)*

**Corollary 2 (validity at every iteration).** The guarantee holds for the model truncated at any
iteration prefix — in particular at the early-stopped `best_iter`, and mid-training. Convergence,
gradient stabilisation, hessian flooring, bagging, feature subsampling and seed choice are all
irrelevant to (M). *(Each tree is feasible at construction; prefixes of feasible sums are
feasible.)*

**Remark (floating point).** In IEEE-754 the per-tree leaf values are exactly ordered; the ensemble
sum incurs rounding `<= n_trees * eps * max|partial sum|` (~1e-12 for 1000 trees at score scale 10),
which the audits bound *measured*: every audit in V2/V3 reports max positive step `0.0` exactly.

---

## 4. Flexibility: the constraint removes nothing else

**Proposition 1 (density).** Let `K` be a compact rectangle of feature space, and let
`g : K -> R` be continuous and coordinatewise non-increasing in `T`. Then for every `eps > 0` there
is a finite sum of trees, each satisfying the `-1` constraint on `T`, within `eps` of `g` uniformly
on `K`. (Same statement with `+1`/non-decreasing; and unconstrained trees on `X \ T` are dense in
the continuous functions that do not depend on `T`.)

*Proof sketch.* Partition `K` into a product grid: cells `C` over the `X \ T` coordinates times a
grid over `T`. Define the staircase `s(x) = g(corner of the cell of x)`, choosing the corner rule so
that within each `X\T`-cell the staircase is non-increasing in the `T` coordinates (inherited from
`g`). Uniform continuity gives `|s - g| < eps` for a fine enough grid. `s` is realisable as a single
tree (split first on the `X\T` cells, then on `T` thresholds with ordered leaf values): the
monotone constraint requires order only within fixed `x_{-T}` paths, which the construction
satisfies. ∎

Consequences: the hypothesis class is exactly *(all correctly-signed monotone functions)* x
*(all `T`-free dispersion surfaces)*, up to approximation error that vanishes with tree count. The
zero-inflation degree and the positive-part shape retain full dependence on every regressor — the
only thing removed is wrong-signed price variation. **Empirical counterpart:** gate V4a — on a DGP
whose truth lies in this class, the fitted surfaces recover the truth (correlations ~0.97 / ~0.99 /
~0.9 for logit-pi / log-mu / log-theta) with `theta` exactly price-flat; gate V3 split census —
the constrained model keeps using price in gate and mu (hundreds of splits), i.e. monotone
*and informative*, not monotone-by-deletion.

**Proposition 2 (no power loss under correct specification).** If the truth lies in the constrained
class, the constrained MLE cannot be asymptotically worse than the unconstrained MLE, and is
typically strictly better in finite samples (imposing true restrictions reduces variance; the
constrained estimator coincides with the unconstrained one whenever the latter already satisfies
the constraints, and improves on it otherwise by projection). We do not lean on an asymptotic
theorem for boosted trees; instead the claim is verified directly: gate **V6** runs a paired
multi-seed study (constrained vs unconstrained, identical capacity, correctly-specified DGP) and
requires: test NLL and oracle-mean RMSE not worse within 2 standard errors; parameter-recovery
parity; and — because correct specification removes every excuse — *absolute* calibration of the
constrained model: uniform PIT and nominal 80% coverage.

**Proposition 3 (misspecification view).** If the truth is *not* in the class (e.g. a pocket where
the true mean rises with price), likelihood boosting within the class targets the KL projection of
the truth onto the class: the fitted model is the closest price-monotone distribution. This is the
*requested* behaviour (the inductive bias overrides the data), and its cost is quantified on the
adversarial panel (trap units) in `experiment/results.json`. Two corollaries verified in code:
the fitted dispersion is *conditional* dispersion — under feature-insufficiency it absorbs
unexplained location variance (the per-unit theta-MLE correlation with the structural `theta` is
positive given the *true* location and vanishes given the fitted one; gate V4b attributes this
with an oracle-location diagnostic — a property of likelihood estimation itself, not of the
constraints) — and conditional calibration of the constrained model is at least as good as the
unconstrained one's on the same panel (relative gates, V4b: PIT-by-tier 0.164 vs 0.219).

---

## 5. Necessity of the assumptions: the failure atlas

Each row drops one assumption and exhibits the collapse — by proof, by constructed counterexample,
or by measured violation.

| dropped | collapse | evidence |
|---|---|---|
| theta excluded from `T` (A2) | `m` strictly decreasing in `th` (Lemma 3): a theta falling with price pushes the conditional mean **up**; with near-flat location elasticity the total mean rises with price. Constraint `0` (unconstrained) leaves the channel open; shared `-1` invites it. | numeric example: `mu=1`: `th: 2 -> 0.5` lifts `m` from 1.80 to 2.37; trap-unit DGP truth is non-monotone on 8.1% of sweeps via exactly this channel |
| per-parameter signs (A2) | the gate needs `+1` while mu needs `-1` and theta needs *exclusion*; one multiclass booster has ONE global vector: with `[-1]`, P(zero) is *forbidden from rising with price* (channel deleted, zero rate miscalibrated — observed under every configuration), and the theta booster is *invited* to track the falling price–dispersion signal, re-opening the truncation channel of Lemma 3 | V2 num_class demo: both outputs forced non-increasing (max step 0.0, 0.0), the +slope target crushed (RMSE 1.73 vs 0.004); model B: 0 gate price-splits in every run; 87% sweep violations when its theta booster finds the price signal (240 splits), 0% when it misses it (its positive part then coincides with the correctly-constructed C) — monotone only by accident |
| learner-level constraint (A1) — replaced by penalised likelihood | trees are piecewise constant: the monotonicity functional has no usable gradient (0 a.e., Dirac at thresholds); finite-difference penalties bind only at their evaluation points while violations live at split thresholds that move every iteration; soft penalties select within an *unconstrained* class and certify nothing pointwise | the library's own `penalize_crossing` for expectiles, with the authors' caveat that crossing still occurs; unconstrained model A violates on 99.6% of test sweeps |
| NaN-free constrained features (A1b) | **discovered hazard**: with 40% NaN in one constrained feature, LightGBM's enforcement breaks at O(1e-2) on *numeric* sweeps — including sweeps of the *other*, NaN-free constrained feature (bound propagation through the missing branch poisons subtree constraints) | V5: hazard reproduced (max step 0.030 with NaN; 0.0 without); safe encoding (neutral value + unconstrained static indicator) certified at 0.0 |
| counterfactual contract (A3) | a feature derived from current own price but left out of `T` (unregistered ratio, rank, or price-derived imputation) moves under the intervention without a constraint — monotone composition fails | by construction; enforced by the feature-registry assertion |
| positive shrinkage / exact predict (A4) | a negative weight flips a tree's direction; `pred_early_stop` returns approximate sums; `linear_tree` leaves are unconstrained linear models | configuration ban; never used |
| isotonic replaced by non-monotone post-hoc map | composition with a non-monotone calibrator destroys Theorem 1 | Corollary 1 delimits exactly which recalibrations are safe |

A final non-assumption worth naming: **convergence is not assumed anywhere.** A diverged, stalled,
or absurdly-early-stopped model still satisfies (M) (Corollary 2). What convergence affects is (F)
and (E) — quality — which is why those are gated separately (V4, V6).

---

## 6. Assumption-to-gate map

| assumption / claim | gate(s) | measured |
|---|---|---|
| analytic score & information of ZTNB | V0 | max rel. err vs FD ~1e-7 (grad), ~1e-9 (hess); score identity at truth within MC error |
| Lemma 2'/3 numerics over the full clip envelope | V1.L1–L5 | strict signs hold; `max w = -5.3e-21`; FOSD crossings `<= 0` |
| (A1) enforcement, all methods, NaN-free | V2 | max positive raw step **0.0** over 24 configs x 52 contexts, sweeps to +-1e9 |
| single-vector multiclass inexpressibility | V2 demo | both outputs forced same sign; +slope RMSE 1.73 vs 0.004 |
| (M) per scored instance: mean | V3 | **0.0** over 7,200 instances x 89-point grid incl. out-of-support |
| (M) channels: pi up, mu down, theta flat | V3 | min/max steps 0.0; theta step exactly 0 |
| (M) quantiles q10..q99 and full CDF | V3 | max positive quantile step 0.0; max CDF drop 0.0 |
| no channel deletion (informative monotone) | V3 census | constrained model: hundreds of price splits in gate & mu |
| (F) recovery on identifiable DGP | V4a | corr(logit pi, log mu, log theta) ≈ 0.97 / 0.99 / 0.9; theta exactly price-flat; spread matched |
| (F)/(E) conditional calibration parity | V4b | constrained >= unconstrained on variance-rank, slope, PIT-by-tier |
| dispersion-absorbs-misfit attribution | V4b | per-unit theta-MLE corr vs structural theta: +0.19 given the true location, ≈0 (−0.01) given the fitted one — misfit absorption, not a trainer defect |
| (A1b) hazard + safe encoding | V5 | hazard 0.03; safe encoding 0.0 |
| (E) no power loss under correct spec | V6 | paired multi-seed: dNLL, dRMSE within 2 SE of 0 or better; PIT uniform; coverage 0.80 +- 0.02 |

All gates and their exact measured values: `experiment/verification_report.json`. The suite exits
non-zero on any failure; it is the executable form of this document.

---

## 7. Scope and honest limitations

1. **The guarantee is about the model, not the world.** (M) certifies the *predictive surface*
   responds monotonically to price; it does not make the price effect causal. Confounded training
   data biases the *size* of the response (and the projection in Prop. 3 governs pockets where the
   observational truth is non-monotone), never the sign or the guarantee.
2. **Dispersion is conditional dispersion.** Under feature-insufficiency, `theta(x)` is the
   dispersion *given x*, which absorbs unexplained location heterogeneity (V4b attribution). This
   is a property of likelihood estimation itself, shared exactly by the unconstrained model.
3. **Optimisation is field-standard, not theorem-grade.** Two-phase fitting (location boosting with
   a profiled scalar dispersion, then cyclic per-parameter Newton steps accepted only on validation
   gain) has no global convergence theorem on the non-convex NLL — the same status as gamboostLSS /
   XGBoostLSS / NGBoost. Two implementation hazards are documented and engineered around: pooled
   unconditional start values degenerate to the `theta -> 0` ridge where NB2 gradients die (start
   from a scale-correct location constant and *profile* theta against the conditional fit), and the
   theta booster is kept only if it beats the scalar-theta model on validation NLL (the Z0->Z1
   promotion gate) — quality can only move up.
4. **Statistical claims are sample-verified, not universally quantified.** (M) is proven for all
   inputs; (F)/(E) are established on the designed DGPs (identifiable, adversarial, multi-seed) and
   gated — the appropriate epistemic standard for properties that depend on data.
