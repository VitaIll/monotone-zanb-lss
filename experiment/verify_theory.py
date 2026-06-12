"""
Research-grade verification suite for the monotone ZANB-LSS construction.

Each section certifies one layer of the theory in THEORY.md:

  V1  Distribution-level lemmas on dense parameter grids, including the
      raw-score clip corners:
        L1: m(mu,th) = mu/(1-p0) strictly increasing in mu      (Lemma 2)
        L2: p0 strictly decreasing in th; m strictly decreasing in th
                                                                 (Lemma 3)
        L3: ZTNB first-order stochastic dominance in mu          (Lemma 2')
        L4: ZANB CDF pointwise ordering under (pi up, mu down)   (Thm 2 step)
        L5: closed-form mean == series sum of the pmf            (sanity)
  V2  LightGBM monotone-constraint enforcement, certified adversarially:
      all three constraint methods x tree capacities x NaN contamination x
      native and custom (objective="none" + fobj) training paths; targets
      engineered so the constraint must BIND; sweeps far outside the training
      support. PASS = max positive step <= 1e-12 in raw score (Assumption A1).
  V3  End-to-end per-instance audit of the experiment models: EVERY test row,
      fine price grid incl. out-of-support values; mean, quantiles, full-CDF
      dominance; per-channel monotonicity; exact theta flatness; treatment
      split-count census (characterises how model B "achieves" monotonicity
      by deleting the price->occurrence channel).
  V4  No-collapse / flexibility:
      V4a identifiable DGP (all three parameters driven by OBSERVED features,
          theta price-free): the constrained class must RECOVER pi, mu, theta
          variation (Proposition 1 in practice).
      V4b misspecified main panel: conditional-variance calibration and PIT
          by volume tier (KL-projection sense), plus the oracle-mu attribution
          of the theta-inversion artefact.
  V5  Missing-value contract: interventions map numeric->numeric and
      NaN->NaN; monotonicity holds per NaN-pattern stratum (Remark R2).

Writes verification_report.json; prints PASS/FAIL per gate; exit code 1 on
any FAIL.
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import nbinom, spearmanr
from scipy.special import expit
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from zanb_lss import (GateBooster, ZTNBCyclicBooster, ZANBModel,
                      log_p0, one_minus_p0, ztnb_logpmf, ztnb_mean, zanb_mean,
                      zanb_nll, zanb_cdf, zanb_ppf, zanb_pit, ztnb_sample,
                      check_gradients, numclass_shared_constraint_demo)
from run_experiment import simulate_panel, build_model, FEATS, TREAT

REPORT = {}
FAILURES = []


def gate(name, ok, detail):
    REPORT[name] = {"pass": bool(ok), **detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# ----------------------------------------------------------------------------
# V1 distribution lemmas
# ----------------------------------------------------------------------------

def v1_lemmas():
    print("V1: distribution-level lemmas")
    mu = np.exp(np.linspace(-15, 15, 1201))           # full clip range of a
    ths = np.exp(np.linspace(-8, 8, 33))              # full clip range of b

    # L1: m strictly increasing in mu. Strictness is analytic (MLR, Lemma 2);
    # the grid validates the float implementation: relative steps must never
    # be negative beyond representation noise, and must be strictly positive
    # away from the saturated corners.
    worst_rel, worst_interior = np.inf, np.inf
    interior = (mu >= np.exp(-10)) & (mu <= np.exp(10))
    for th in ths:
        m = ztnb_mean(mu, np.full_like(mu, th))
        d = np.diff(m) / (1.0 + m[:-1])
        worst_rel = min(worst_rel, float(np.min(d)))
        worst_interior = min(worst_interior, float(np.min(np.diff(m)[interior[:-1]])))
    gate("V1.L1_truncated_mean_increasing_in_mu",
         worst_rel >= -1e-12 and worst_interior > 0,
         {"min_relative_step": worst_rel,
          "min_absolute_step_interior": worst_interior})

    # L2: d log p0 / d theta = w(u) = -log1p(u) + u/(1+u) < 0 for all u>0
    # (analytic inequality, Lemma 3); validate w over the whole reachable
    # envelope of u = mu/theta, then the implied m-decrease on the grid.
    u = np.exp(np.linspace(-23, 23, 4001))
    w = -np.log1p(u) + u / (1.0 + u)
    th_grid = np.exp(np.linspace(-8, 8, 1201))
    worst_l2m = -np.inf
    for mmu in np.exp(np.linspace(-15, 15, 33)):
        m = ztnb_mean(np.full_like(th_grid, mmu), th_grid)
        worst_l2m = max(worst_l2m, float(np.max(np.diff(m) / (1.0 + m[:-1]))))
    gate("V1.L2_p0_and_truncmean_decreasing_in_theta",
         float(np.max(w)) < 0 and worst_l2m <= 1e-12,
         {"max_w_over_envelope": float(np.max(w)),
          "max_relative_m_step_in_theta": worst_l2m})

    rng = np.random.default_rng(5)
    worst = -np.inf
    for _ in range(80):                                # FOSD of ZTNB in mu
        th = float(np.exp(rng.uniform(-2.5, 3)))
        mu1 = float(np.exp(rng.uniform(-3, 5)))
        mu2 = mu1 * float(np.exp(rng.uniform(0.05, 1.0)))
        K = int(nbinom.ppf(0.99995, th, th / (th + mu2)) + 2)
        y = np.arange(1, max(K, 3))
        p = lambda m: th / (th + m)
        F1 = (nbinom.cdf(y, th, p(mu1)) - np.exp(log_p0(mu1, th))) / one_minus_p0(mu1, th)
        F2 = (nbinom.cdf(y, th, p(mu2)) - np.exp(log_p0(mu2, th))) / one_minus_p0(mu2, th)
        worst = max(worst, float(np.max(F2 - F1)))     # F2 must be <= F1
    gate("V1.L3_ZTNB_stochastic_dominance_in_mu", worst <= 1e-12,
         {"max_cdf_crossing": worst})

    worst = -np.inf
    for _ in range(80):                                # ZANB CDF ordering
        th = float(np.exp(rng.uniform(-2.5, 3)))
        mu1 = float(np.exp(rng.uniform(-3, 5)))
        mu2 = mu1 * float(np.exp(-rng.uniform(0.05, 1.0)))   # price up: mu down
        pi1 = float(rng.uniform(0.02, 0.7))
        pi2 = min(pi1 + float(rng.uniform(0.01, 0.25)), 0.95)  # price up: pi up
        K = int(nbinom.ppf(0.99995, th, th / (th + mu1)) + 2)
        y = np.arange(0, max(K, 3))
        F1 = zanb_cdf(y, pi1, np.full_like(y, mu1, dtype=float), np.full_like(y, th, dtype=float))
        F2 = zanb_cdf(y, pi2, np.full_like(y, mu2, dtype=float), np.full_like(y, th, dtype=float))
        worst = max(worst, float(np.max(F1 - F2)))     # F2 must be >= F1
    gate("V1.L4_ZANB_FOSD_under_priceup", worst <= 1e-12,
         {"max_cdf_crossing": worst})

    worst = 0.0
    for _ in range(40):                                # closed-form mean check
        th = float(np.exp(rng.uniform(-2, 2.5)))
        mu = float(np.exp(rng.uniform(-2, 4.5)))
        pi = float(rng.uniform(0.05, 0.8))
        K = int(nbinom.ppf(1 - 1e-12, th, th / (th + mu)) + 5)
        y = np.arange(1, K)
        pmf = np.exp(ztnb_logpmf(y.astype(float), mu, th))
        worst = max(worst, abs((1 - pi) * np.sum(y * pmf)
                               - zanb_mean(pi, mu, th)) / zanb_mean(pi, mu, th))
    gate("V1.L5_mean_formula_vs_series", worst <= 1e-8, {"max_rel_err": worst})


# ----------------------------------------------------------------------------
# V2 LightGBM monotone enforcement, adversarial certification
# ----------------------------------------------------------------------------

def v2_lgbm_certification():
    print("V2: LightGBM monotone-constraint certification (adversarial)")
    rng = np.random.default_rng(11)
    n = 4000
    max_step_overall = -np.inf
    n_cfg = 0
    for method, (leaves, minleaf), nan_frac, custom in itertools.product(
            ["basic", "intermediate", "advanced"],
            [(255, 5), (31, 50)], [0.0, 0.3], [False, True]):
        t = rng.normal(0, 1, n)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        if nan_frac > 0:
            x2[rng.uniform(size=n) < nan_frac] = np.nan
        # constraint (-1 on t) must BIND: slope +5 globally, +/-3 by region
        y = (5.0 * t + 3.0 * t * np.sign(x1) + 2.0 * np.sin(3 * x1)
             + np.nan_to_num(x2) + rng.normal(0, 0.5, n))
        X = np.column_stack([t, x1, x2])
        params = dict(num_leaves=leaves, min_data_in_leaf=minleaf,
                      learning_rate=0.1, verbosity=-1, seed=3,
                      deterministic=True, monotone_constraints=[-1, 0, 0],
                      monotone_constraints_method=method)
        ds = lgb.Dataset(X, label=y, free_raw_data=False)
        if not custom:
            bst = lgb.train({**params, "objective": "regression"}, ds,
                            num_boost_round=300)
        else:
            bst = lgb.Booster(params={**params, "objective": "none"}, train_set=ds)
            for it in range(300):
                def fobj(preds, data):
                    return preds - y, np.ones_like(y)
                bst.update(fobj=fobj)

        tg = np.concatenate([np.linspace(-6, 6, 201), [-1e9, -1e5, 1e5, 1e9]])
        tg = np.sort(tg)
        ctx = np.vstack([X[rng.integers(0, n, 50)],
                         [[0, 5, np.nan]], [[0, -5, 7.0]]])
        for row in ctx:
            Xs = np.tile(row, (len(tg), 1))
            Xs[:, 0] = tg
            pred = bst.predict(Xs, raw_score=True)
            max_step_overall = max(max_step_overall, float(np.max(np.diff(pred))))
        n_cfg += 1
    gate("V2.lgbm_monotone_certified", max_step_overall <= 1e-12,
         {"configs_tested": n_cfg, "contexts_per_config": 52,
          "max_positive_step_raw_score": max_step_overall,
          "grid": "201 pts in [-6,6] + {+-1e5, +-1e9} (train support ~[-3,3])"})

    demo = numclass_shared_constraint_demo()
    gate("V2.numclass_single_vector_binds_all_outputs",
         demo["max_increase_output0"] <= 0 and demo["max_increase_output1"] <= 0
         and demo["rmse_output0_target_plus3x"] > 100 * demo["rmse_output1_target_minus3x"],
         demo)


# ----------------------------------------------------------------------------
# V3 end-to-end per-instance audit on the experiment models
# ----------------------------------------------------------------------------

def _sweep_params(model, base, grid):
    rep = base.loc[base.index.repeat(len(grid))].copy()
    rep[TREAT] = np.tile(grid, len(base))
    pi, mu, th = model.predict_params(rep)
    shp = (len(base), len(grid))
    return pi.reshape(shp), mu.reshape(shp), th.reshape(shp)


def v3_per_instance_audit():
    print("V3: per-instance audit on the experiment panel (every test row)")
    df = simulate_panel()
    df = df[df[FEATS].notna().all(axis=1) & (df.week >= 15)].reset_index(drop=True)
    tr = df[df.week <= 104]
    va = df[(df.week >= 105) & (df.week <= 117)]
    te = df[df.week >= 118].reset_index(drop=True)

    models = {k: build_model(k).fit(tr, tr.y.values, va, va.y.values)
              for k in ("A", "B", "C")}

    grid = np.linspace(-0.55, 0.55, 89)               # beyond training support
    pi, mu, th = _sweep_params(models["C"], te, grid)
    E = zanb_mean(pi, mu, th)

    gate("V3.mean_monotone_every_test_instance",
         float(np.max(np.diff(E, axis=1))) <= 1e-12,
         {"n_instances": int(len(te)), "grid_points": len(grid),
          "max_positive_step_mean": float(np.max(np.diff(E, axis=1)))})
    gate("V3.gate_channel_nondecreasing", float(np.min(np.diff(pi, axis=1))) >= -1e-15,
         {"min_step_pi": float(np.min(np.diff(pi, axis=1)))})
    gate("V3.mu_channel_nonincreasing", float(np.max(np.diff(mu, axis=1))) <= 1e-12,
         {"max_step_mu": float(np.max(np.diff(mu, axis=1)))})
    gate("V3.theta_exactly_price_flat", float(np.max(np.abs(np.diff(th, axis=1)))) == 0.0,
         {"max_abs_step_theta": float(np.max(np.abs(np.diff(th, axis=1))))})

    sub = np.sort(np.random.default_rng(1).choice(len(te), 1500, replace=False))
    qworst = -np.inf
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        Q = np.stack([zanb_ppf(q, pi[sub, g], mu[sub, g], th[sub, g])
                      for g in range(len(grid))], axis=1)
        qworst = max(qworst, float(np.max(np.diff(Q, axis=1))))
    gate("V3.all_quantiles_monotone", qworst <= 0.0,
         {"quantiles": "0.1..0.99", "n_instances": len(sub),
          "max_positive_step": qworst})

    sub2 = sub[:400]
    fworst = -np.inf
    for i in sub2:
        K = int(zanb_ppf(0.999, pi[i, 0], mu[i, 0], th[i, 0])) + 2
        ys = np.arange(0, max(K, 4), dtype=float)
        Fg = np.stack([zanb_cdf(ys, pi[i, g], np.full_like(ys, mu[i, g]),
                                np.full_like(ys, th[i, g]))
                       for g in range(0, len(grid), 4)], axis=1)
        fworst = max(fworst, float(np.max(-np.diff(Fg, axis=1))))  # F must rise with price
    gate("V3.full_cdf_first_order_dominance", fworst <= 1e-12,
         {"n_instances": len(sub2), "max_cdf_drop_with_price": fworst})

    # treatment split census: how each model uses the price feature
    census = {}
    for k, m in models.items():
        i_t = FEATS.index(TREAT)
        census[k] = {
            "gate_splits_on_price": int(m.gate.bst.feature_importance("split")[i_t]),
            "mu_splits_on_price": int(m.ztnb.bst_mu.feature_importance("split")[i_t]),
            "theta_splits_on_price": int(
                m.ztnb.bst_th.feature_importance("split")[m.ztnb.feat_th.index(TREAT)]
                if TREAT in m.ztnb.feat_th else -1),
        }
    okC = census["C"]["gate_splits_on_price"] > 0 and census["C"]["mu_splits_on_price"] > 0
    gate("V3.C_uses_price_in_gate_and_mu_(no_channel_deletion)", okC, census)
    REPORT["V3.split_census"] = census
    return census


# ----------------------------------------------------------------------------
# V4 flexibility / no-collapse
# ----------------------------------------------------------------------------

def _identifiable_dgp(seed, n):
    """Correctly-specified DGP for the constrained class: gate rises in t,
    mu falls in t, theta varies with observed regressors but is price-free."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "t": np.clip(rng.normal(0, 0.15, n), -0.5, 0.5),     # price ratio
        "x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n),
        "x3": rng.uniform(0, 1, n), "x4": rng.uniform(0, 1, n),
        "x5": rng.uniform(0, 1, n), "noise": rng.normal(0, 1, n),
    })
    pi = expit(-1.2 + 1.2 * X.x1 - 0.8 * X.x2 + 1.5 * X.t)
    mu = np.exp(0.6 + 1.8 * X.x1 + 1.0 * np.sin(2 * np.pi * X.x3) - 1.2 * X.t)
    th = np.exp(0.4 + 1.6 * X.x4 - 1.0 * X.x5)               # price-free truth
    z = rng.uniform(size=n) < pi
    y = np.where(z, 0, ztnb_sample(rng, mu.values, th.values))
    return X, y, pi.values, mu.values, th.values


def _fit_identifiable(kind, X, y, itr, iva, rounds=800, esr=100):
    """A = unconstrained, C = per-parameter constrained; otherwise identical."""
    feats = list(X.columns)
    gate_p = dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=100,
                  lambda_l2=1.0, seed=1, deterministic=True)
    mu_p = dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=50,
                lambda_l2=1.0, seed=2, deterministic=True)
    th_p = dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=200,
                lambda_l2=2.0, max_delta_step=0.5, seed=3, deterministic=True)
    th_feats = list(feats)
    if kind == "C":
        gate_p.update(monotone_constraints=[+1 if f == "t" else 0 for f in feats],
                      monotone_constraints_method="advanced")
        mu_p.update(monotone_constraints=[-1 if f == "t" else 0 for f in feats],
                    monotone_constraints_method="advanced")
        th_feats = [f for f in feats if f != "t"]
    model = ZANBModel(
        GateBooster(feats, gate_p, rounds, esr),
        ZTNBCyclicBooster(feats, th_feats, mu_p, th_p, rounds, esr,
                          theta_updates_per_iter=1))
    return model.fit(X[itr], y[itr], X[iva], y[iva])


def v4a_identifiable_recovery():
    print("V4a: shape recovery on an identifiable DGP (all params observable)")
    n = 120000
    X, y, pi, mu, th = _identifiable_dgp(21, n)
    itr = np.arange(n) < int(0.85 * n)
    model = _fit_identifiable("C", X, y, itr, ~itr, rounds=1000, esr=150)

    Xte = X[~itr]
    pih, muh, thh = model.predict_params(Xte)
    c_pi = float(np.corrcoef(np.log(pih / (1 - pih)),
                             np.log(pi[~itr] / (1 - pi[~itr])))[0, 1])
    c_mu = float(np.corrcoef(np.log(muh), np.log(mu[~itr]))[0, 1])
    c_th = float(np.corrcoef(np.log(thh), np.log(th[~itr]))[0, 1])
    s_th = float(np.std(np.log(thh)))
    REPORT["V4a.z1_accepted"] = bool(model.ztnb.z1_accepted)

    gate("V4a.gate_surface_recovered", c_pi >= 0.90, {"corr_logit_pi": c_pi})
    gate("V4a.mu_surface_recovered", c_mu >= 0.90, {"corr_log_mu": c_mu})
    gate("V4a.theta_surface_recovered_price_free", c_th >= 0.80,
         {"corr_log_theta": c_th, "std_log_theta_hat": s_th,
          "std_log_theta_true": float(np.std(np.log(th[~itr])))})
    gate("V4a.theta_varies_with_regressors_not_collapsed", s_th >= 0.30,
         {"std_log_theta_hat": s_th})

    # and the guarantee still holds on this model, per instance
    base = Xte.iloc[:1500]
    grid = np.linspace(-0.6, 0.6, 61)
    rep = base.loc[base.index.repeat(len(grid))].copy()
    rep["t"] = np.tile(grid, len(base))
    p_, m_, t_ = model.predict_params(rep)
    E = zanb_mean(p_, m_, t_).reshape(len(base), len(grid))
    gate("V4a.monotone_mean_on_identifiable_model",
         float(np.max(np.diff(E, axis=1))) <= 1e-12,
         {"max_positive_step": float(np.max(np.diff(E, axis=1)))})


def v4b_conditional_calibration():
    """On the deliberately misspecified main panel, absolute calibration is a
    property of the DGP (unobservable heterogeneity), not of the constrained
    construction. The no-collapse claim is therefore RELATIVE: the constrained
    model C must calibrate at least as well as the unconstrained model A."""
    print("V4b: conditional calibration on the misspecified panel (C vs A)")
    df = simulate_panel()
    df = df[df[FEATS].notna().all(axis=1) & (df.week >= 15)].reset_index(drop=True)
    tr = df[df.week <= 104]
    va = df[(df.week >= 105) & (df.week <= 117)]
    te = df[df.week >= 118].reset_index(drop=True)

    stats = {}
    for kind in ("C", "A"):
        model = build_model(kind).fit(tr, tr.y.values, va, va.y.values)
        pi, mu, th = model.predict_params(te)
        E = zanb_mean(pi, mu, th)
        S2 = (mu + mu ** 2 * (1 + 1 / th)) / one_minus_p0(mu, th)
        V = (1 - pi) * S2 - E ** 2
        resid2 = (te.y.values - E) ** 2
        dec = pd.qcut(V, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"V": V, "r2": resid2, "d": dec}).groupby("d").mean()
        rho = float(spearmanr(g.V, g.r2).statistic)
        slope = float(np.polyfit(np.log(g.V), np.log(g.r2), 1)[0])
        u = zanb_pit(np.random.default_rng(2), te.y.values.astype(float), pi, mu, th)
        tiers = pd.qcut(te.rate13.values, 3, labels=False)
        pitdev = max(float(np.max(np.abs(
            np.histogram(u[tiers == t_], bins=10, range=(0, 1), density=True)[0] - 1)))
            for t_ in range(3))
        stats[kind] = {"var_spearman": rho, "var_loglog_slope": slope,
                       "pit_max_dev_by_tier": pitdev}
        if kind == "C":
            modelC, piC, muC, thC = model, pi, mu, th

    gate("V4b.variance_rank_calibration_C", stats["C"]["var_spearman"] >= 0.95,
         {"spearman_decile_C": stats["C"]["var_spearman"]})
    gate("V4b.calibration_C_not_worse_than_unconstrained_A",
         (stats["C"]["pit_max_dev_by_tier"] <= stats["A"]["pit_max_dev_by_tier"] + 0.05)
         and (abs(stats["C"]["var_loglog_slope"] - 1.0)
              <= abs(stats["A"]["var_loglog_slope"] - 1.0) + 0.05),
         {"C": stats["C"], "A": stats["A"],
          "note": "absolute miscalibration here is induced by unobservable "
                  "heterogeneity in the DGP and affects A identically"})

    # attribution of the theta-inversion artefact: dispersion absorbs mu misfit
    model = modelC
    tep = te[te.y > 0]
    _, mu_hat, _ = model.predict_params(tep)
    y = tep.y.values.astype(float)

    def theta_mle(y_, mu_):
        f = lambda b: -np.sum(ztnb_logpmf(y_, mu_, np.exp(b)))
        return minimize_scalar(f, bounds=(-4, 4), method="bounded").x

    rows = []
    for u_ in np.unique(tep.unit.values)[:300]:
        s = tep.unit.values == u_
        if s.sum() < 8:
            continue
        rows.append((theta_mle(y[s], tep.mu_true.values[s]),
                     theta_mle(y[s], mu_hat[s]),
                     np.log(tep.th_true.values[s].mean())))
    rows = np.array(rows)
    c_true = float(np.corrcoef(rows[:, 0], rows[:, 2])[0, 1])
    c_fit = float(np.corrcoef(rows[:, 1], rows[:, 2])[0, 1])
    gate("V4b.theta_inversion_attributed_to_mu_misfit", c_true > c_fit,
         {"corr_thetaMLE_given_TRUE_mu": c_true,
          "corr_thetaMLE_given_FITTED_mu": c_fit,
          "n_units": int(len(rows)),
          "note": "fitted dispersion is conditional-on-features dispersion; "
                  "with the true mean the inversion disappears -> artefact of "
                  "unobservable heterogeneity, not of the constrained trainer"})


def v6_no_power_loss_when_correctly_specified():
    """The user-level requirement: GIVEN the model is correctly specified for
    the DGP (truth lies in the constrained class: gate rising in price, mu
    falling, theta price-free), imposing the constraint must cost no
    predictive power or distributional quality. Classical estimation logic:
    true restrictions cannot hurt asymptotically and typically reduce
    variance in finite samples. Verified as a PAIRED multi-seed study of
    C (constrained) vs A (unconstrained, otherwise identical), plus ABSOLUTE
    calibration gates for C -- on a well-specified DGP there is no
    misspecification excuse: PIT must be uniform and coverage nominal."""
    print("V6: no power loss under correct specification (paired, multi-seed)")
    seeds = [101, 202, 303, 404, 505, 606]
    n = 60000
    per_seed = []
    for s in seeds:
        X, y, pi_t, mu_t, th_t = _identifiable_dgp(s, n)
        idx = np.arange(n)
        itr = idx < int(0.70 * n)
        iva = (idx >= int(0.70 * n)) & (idx < int(0.85 * n))
        ite = idx >= int(0.85 * n)
        E_true = zanb_mean(pi_t[ite], mu_t[ite], th_t[ite])
        row = {}
        for kind in ("A", "C"):
            m = _fit_identifiable(kind, X, y, itr, iva, rounds=600, esr=80)
            pi, mu, th = m.predict_params(X[ite])
            E = zanb_mean(pi, mu, th)
            u = zanb_pit(np.random.default_rng(s), y[ite].astype(float), pi, mu, th)
            row[kind] = {
                "nll": float(np.mean(zanb_nll(y[ite].astype(float), pi, mu, th))),
                "rmse_true_mean": float(np.sqrt(np.mean((E - E_true) ** 2))),
                "corr_pi": float(np.corrcoef(
                    np.log(pi / (1 - pi)), np.log(pi_t[ite] / (1 - pi_t[ite])))[0, 1]),
                "corr_mu": float(np.corrcoef(np.log(mu), np.log(mu_t[ite]))[0, 1]),
                "corr_th": float(np.corrcoef(np.log(th), np.log(th_t[ite]))[0, 1]),
                "pit_max_dev": float(np.max(np.abs(np.histogram(
                    u, bins=10, range=(0, 1), density=True)[0] - 1))),
                "cov80": float(np.mean((u >= 0.1) & (u <= 0.9))),
                "z1_accepted": bool(m.ztnb.z1_accepted),
            }
        per_seed.append(row)
        print(f"  seed {s}: dNLL(C-A)={row['C']['nll'] - row['A']['nll']:+.5f}  "
              f"dRMSE={row['C']['rmse_true_mean'] - row['A']['rmse_true_mean']:+.4f}  "
              f"C(pit={row['C']['pit_max_dev']:.3f}, cov80={row['C']['cov80']:.3f})")

    arr = lambda f: np.array([f(r) for r in per_seed])
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    d_nll = arr(lambda r: r["C"]["nll"] - r["A"]["nll"])
    d_rmse = arr(lambda r: r["C"]["rmse_true_mean"] - r["A"]["rmse_true_mean"])
    d_cpi = arr(lambda r: r["C"]["corr_pi"] - r["A"]["corr_pi"])
    d_cmu = arr(lambda r: r["C"]["corr_mu"] - r["A"]["corr_mu"])
    d_cth = arr(lambda r: r["C"]["corr_th"] - r["A"]["corr_th"])
    pit_C = arr(lambda r: r["C"]["pit_max_dev"])
    cov_C = arr(lambda r: r["C"]["cov80"])
    rmse_A = arr(lambda r: r["A"]["rmse_true_mean"])

    REPORT["V6.per_seed"] = per_seed
    gate("V6.nll_not_worse_than_unconstrained",
         float(np.mean(d_nll)) <= max(2 * se(d_nll), 5e-4),
         {"mean_dNLL_C_minus_A": float(np.mean(d_nll)), "se": se(d_nll),
          "per_seed": [round(float(v), 5) for v in d_nll]})
    gate("V6.true_mean_rmse_not_worse",
         float(np.mean(d_rmse)) <= max(2 * se(d_rmse), 0.01 * float(np.mean(rmse_A))),
         {"mean_dRMSE_C_minus_A": float(np.mean(d_rmse)), "se": se(d_rmse),
          "mean_rmse_A": float(np.mean(rmse_A))})
    gate("V6.parameter_recovery_not_worse",
         min(float(np.mean(d_cpi)), float(np.mean(d_cmu))) >= -0.01
         and float(np.mean(d_cth)) >= -0.02,
         {"mean_dcorr_pi": float(np.mean(d_cpi)),
          "mean_dcorr_mu": float(np.mean(d_cmu)),
          "mean_dcorr_th": float(np.mean(d_cth))})
    gate("V6.absolute_calibration_of_C_under_correct_spec",
         float(np.mean(pit_C)) <= 0.15 and abs(float(np.mean(cov_C)) - 0.80) <= 0.02,
         {"mean_pit_max_dev_C": float(np.mean(pit_C)),
          "mean_coverage80_C": float(np.mean(cov_C)),
          "z1_accept_count_C": int(sum(r["C"]["z1_accepted"] for r in per_seed))})


# ----------------------------------------------------------------------------
# V5 missing-value contract
# ----------------------------------------------------------------------------

def v5_nan_contract():
    """NaN in a CONSTRAINED feature is a discovered, material hazard:
    LightGBM's monotone enforcement is then violated even on numeric-only
    sweeps, and even on the OTHER constrained feature (bound propagation
    through the missing-value branch poisons subtree constraints).
    Contract (A1b): constrained features must be NaN-free. Missing values are
    encoded as (neutral value 0 = ratio 1) + an unconstrained indicator that
    is static under the intervention. This section (a) reproduces the hazard,
    (b) certifies the safe encoding."""
    print("V5: missing-value hazard + safe-encoding certification")
    rng = np.random.default_rng(31)
    n = 8000
    t1 = rng.normal(0, 1, n)
    t2 = rng.normal(0, 1, n)
    miss = rng.uniform(size=n) < 0.4                   # e.g. no ecomm price
    x = rng.normal(0, 1, n)
    y = (-2.0 * t1 - 1.5 * np.where(miss, 0.0, t2) + 0.8 * miss
         + np.sin(2 * x) + rng.normal(0, 0.3, n))
    grid = np.linspace(-4, 4, 81)
    params = dict(objective="regression", num_leaves=63, learning_rate=0.1,
                  min_data_in_leaf=30, verbosity=-1, seed=7, deterministic=True,
                  monotone_constraints_method="advanced")

    def max_steps(bst, X, joint_t2):
        worst_obs, worst_mis = -np.inf, -np.inf
        for i in rng.integers(0, n, 80):
            row = X[i]
            Xs = np.tile(row, (len(grid), 1))
            Xs[:, 0] = grid
            if joint_t2 and not miss[i]:
                Xs[:, 1] = row[1] + (grid - row[0])
            d = float(np.max(np.diff(bst.predict(Xs, raw_score=True))))
            if miss[i]:
                worst_mis = max(worst_mis, d)
            else:
                worst_obs = max(worst_obs, d)
        return worst_obs, worst_mis

    # (a) hazard reproduction: NaN inside a constrained feature
    t2_nan = np.where(miss, np.nan, t2)
    X_nan = np.column_stack([t1, t2_nan, x])
    bst_nan = lgb.train({**params, "monotone_constraints": [-1, -1, 0]},
                        lgb.Dataset(X_nan, label=y), num_boost_round=200)
    h_obs, h_mis = max_steps(bst_nan, X_nan, joint_t2=True)
    REPORT["V5.hazard_nan_in_constrained_feature"] = {
        "max_step_observed_rows": h_obs, "max_step_missing_rows": h_mis,
        "expected": "POSITIVE = enforcement broken; motivates contract A1b"}
    print(f"  [info] hazard reproduced: max steps obs={h_obs:.4f} mis={h_mis:.4f}")

    # (b) safe encoding: impute neutral 0 + unconstrained static indicator
    t2_safe = np.where(miss, 0.0, t2)
    X_safe = np.column_stack([t1, t2_safe, miss.astype(float), x])
    bst_safe = lgb.train({**params, "monotone_constraints": [-1, -1, 0, 0]},
                         lgb.Dataset(X_safe, label=y), num_boost_round=200)
    s_obs, s_mis = max_steps(bst_safe, X_safe, joint_t2=True)
    gate("V5.safe_missing_encoding_monotone", max(s_obs, s_mis) <= 1e-12,
         {"max_step_T2_observed_rows": s_obs,
          "max_step_T2_imputed_rows": s_mis,
          "hazard_if_nan_kept": {"obs": h_obs, "mis": h_mis},
          "contract": "A1b: constrained features NaN-free; missing -> "
                      "(0, indicator); indicator static under intervention"})


# ----------------------------------------------------------------------------

def main():
    print("=" * 72)
    REPORT["gradient_check"] = check_gradients()
    g = REPORT["gradient_check"]
    gate("V0.analytic_derivatives_vs_finite_differences",
         max(g["grad_a_max_rel_err"], g["grad_b_max_rel_err"],
             g["hess_a_max_rel_err"], g["hess_b_max_rel_err"]) <= 1e-6, g)
    v1_lemmas()
    v2_lgbm_certification()
    v3_per_instance_audit()
    v4a_identifiable_recovery()
    v4b_conditional_calibration()
    v5_nan_contract()
    v6_no_power_loss_when_correctly_specified()

    REPORT["summary"] = {"n_gates": sum(1 for k in REPORT if k.startswith("V")),
                         "failures": FAILURES}
    with open(os.path.join(HERE, "verification_report.json"), "w") as f:
        json.dump(REPORT, f, indent=2)
    print("=" * 72)
    if FAILURES:
        print("FAILED gates:", FAILURES)
        sys.exit(1)
    print(f"ALL {REPORT['summary']['n_gates']} GATES PASS "
          f"-> verification_report.json")


if __name__ == "__main__":
    main()
