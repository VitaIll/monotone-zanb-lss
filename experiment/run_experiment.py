"""
Monotone ZANB-LSS experiment.

Synthetic product x pharmacy weekly demand panel with:
  - unit base rates spanning ~4 orders of magnitude,
  - product-level price paths (slow revisions + promo episodes),
  - heterogeneous negative mean-elasticities (including near-zero "trap" products),
  - gate (P(zero)) increasing in price,
  - dispersion theta DECREASING in price (strongly so for trap products) --
    an adversarial channel: through zero-truncation it pushes the conditional
    mean UP with price, so the TRUE mean is locally non-monotone for trap units.

Models (identical 3-booster ZANB architecture, identical hyperparameters):
  A  unconstrained
  B  naive shared sign: lr -> -1 in gate, mu AND theta boosters -- exactly what
     a single global `monotone_constraints` vector does to a stock LightGBMLSS
     multiclass booster (see numclass_shared_constraint_demo)
  C  per-parameter: gate +1, mu -1, theta price-free (ours)

Audit: dense counterfactual price sweeps for every test unit; mean must be
non-increasing in price. C must show exactly zero violations and near-A fit.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zanb_lss import (GateBooster, ZTNBCyclicBooster, ZANBModel,
                      zanb_mean, zanb_cdf, zanb_ppf, zanb_pit, ztnb_mean,
                      ztnb_sample, check_gradients, check_quantiles,
                      numclass_shared_constraint_demo)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

CLR = {"A": "#56B4E9", "B": "#D55E00", "C": "#009E73", "truth": "#000000",
       "oracle": "#888888"}
LBL = {"A": "A: unconstrained",
       "B": "B: shared sign (stock-LSS emulation)",
       "C": "C: per-parameter (ours)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 9.0,
    "axes.titlesize": 10.0, "axes.labelsize": 9.0, "legend.fontsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.unicode_minus": False, "figure.constrained_layout.use": True,
})

FEATS = ["lr", "log_price_level", "sin52", "cos52", "rate13", "zshare13",
         "prod_decile", "pharm_decile", "unit_share_prod", "noise1", "noise2"]
TREAT = "lr"
J, P, W = 40, 15, 130
N_TRAP = 5
LR_GRID = np.linspace(-0.35, 0.35, 57)
SWEEP_WEEKS = (124, 129)


# ----------------------------------------------------------------------------
# Data generating process
# ----------------------------------------------------------------------------

def true_params(bunit, bunit_c, seas, gunit, dunit, t2, lr):
    mu = np.minimum(np.exp(bunit + seas + gunit * lr), 800.0)
    pi = expit(-1.0 - 0.8 * bunit_c + dunit * lr)
    th = np.clip(np.exp(0.15 + 0.45 * bunit_c - t2 * lr), 0.25, 12.0)
    return pi, mu, th


def simulate_panel(seed=20260611):
    rng = np.random.default_rng(seed)

    # products
    log_p0 = rng.uniform(np.log(40), np.log(1500), J)
    is_trap = np.arange(J) < N_TRAP
    bprod = np.where(is_trap, np.log(1.2) + rng.normal(0, 0.2, J),
                     rng.normal(np.log(2.0), 1.1, J))
    gamma = np.where(is_trap, -rng.uniform(0.02, 0.08, J),
                     -rng.uniform(0.3, 2.2, J))
    delta = np.where(is_trap, rng.uniform(0.6, 1.2, J), rng.uniform(0.8, 3.0, J))
    seasA = rng.uniform(0.0, 0.45, J)
    phase = rng.uniform(0, 2 * np.pi, J)
    t2 = np.where(is_trap, 1.6, 0.25)

    # pharmacies
    bph = rng.normal(0.0, 0.7, P)

    # price paths per product
    weeks = np.arange(W)
    price = np.zeros((J, W))
    for j in range(J):
        steps = rng.normal(0, 0.03, W) * (rng.uniform(size=W) < 0.12)
        rw = np.clip(np.cumsum(steps), -0.12, 0.28)
        promo = np.zeros(W)
        t = 0
        while t < W:
            if rng.uniform() < 0.08:
                dur = 1 + min(rng.geometric(0.5), 3)
                depth = rng.uniform(0.12, 0.40)
                promo[t:t + dur] = depth
                t += dur
            t += 1
        price[j] = np.exp(log_p0[j] + rw - promo)

    # reference price: trailing 13w mean, strictly before current week
    pr = pd.DataFrame(price.T)                      # (W, J)
    ref = pr.shift(1).rolling(13, min_periods=8).mean()
    lr_feat = np.log(pr / ref).to_numpy().T         # (J, W)
    log_lvl = np.log(ref).to_numpy().T

    # units
    rows = []
    bunit_all = np.zeros((J, P))
    for j in range(J):
        for p in range(P):
            bunit_all[j, p] = bprod[j] + bph[p] + rng.normal(0, 0.25)
    bmean = bunit_all.mean()

    gunit = np.clip(gamma[:, None] + rng.normal(0, 0.10, (J, P)), -5.0, -0.015)
    dunit = np.clip(delta[:, None] + rng.normal(0, 0.20, (J, P)), 0.2, 5.0)

    for j in range(J):
        seas = seasA[j] * np.sin(2 * np.pi * weeks / 52 + phase[j])
        for p in range(P):
            bu = bunit_all[j, p]
            lr = lr_feat[j]
            pi, mu, th = true_params(bu, bu - bmean, seas, gunit[j, p],
                                     dunit[j, p], t2[j], np.nan_to_num(lr))
            z = rng.uniform(size=W) < pi
            y = np.where(z, 0, ztnb_sample(rng, mu, th))
            rows.append(pd.DataFrame({
                "prod": j, "pharm": p, "unit": j * P + p, "week": weeks,
                "y": y, "lr": lr, "log_price_level": log_lvl[j],
                "sin52": np.sin(2 * np.pi * weeks / 52),
                "cos52": np.cos(2 * np.pi * weeks / 52),
                "seasval": seas,
                "bunit": bu, "bunit_c": bu - bmean,
                "gunit": gunit[j, p], "dunit": dunit[j, p], "t2j": t2[j],
                "pi_true": pi, "mu_true": mu, "th_true": th,
                "is_trap": bool(is_trap[j]), "p0_prod": np.exp(log_p0[j]),
            }))
    df = pd.concat(rows, ignore_index=True)
    df["E_true"] = zanb_mean(df.pi_true.values, df.mu_true.values, df.th_true.values)

    # rolling / hierarchy features (all strictly before current week)
    df = df.sort_values(["unit", "week"]).reset_index(drop=True)
    g = df.groupby("unit", sort=False)
    df["rate13"] = np.log1p(g["y"].transform(
        lambda s: s.shift(1).rolling(13, min_periods=4).mean()))
    df["zero"] = (df.y == 0).astype(float)
    df["zshare13"] = g["zero"].transform(
        lambda s: s.shift(1).rolling(13, min_periods=4).mean())
    df["u13"] = g["y"].transform(lambda s: s.shift(1).rolling(13, min_periods=4).sum())

    ptot = df.pivot_table(index="week", columns="prod", values="y", aggfunc="sum")
    proll = ptot.shift(1).rolling(13, min_periods=4).sum()
    pdec = proll.rank(axis=1, pct=True)
    df = df.merge(pdec.stack().rename("prod_decile").reset_index(), on=["week", "prod"], how="left")

    ftot = df.pivot_table(index="week", columns="pharm", values="y", aggfunc="sum")
    froll = ftot.shift(1).rolling(13, min_periods=4).sum()
    fdec = froll.rank(axis=1, pct=True)
    df = df.merge(fdec.stack().rename("pharm_decile").reset_index(), on=["week", "pharm"], how="left")

    df["unit_share_prod"] = df.groupby(["prod", "week"])["u13"].rank(pct=True)

    rng2 = np.random.default_rng(7)
    df["noise1"] = rng2.normal(size=len(df))
    df["noise2"] = rng2.normal(size=len(df))
    return df


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def mono_vec(feats, signs):
    return [int(signs.get(f, 0)) for f in feats]


def build_model(kind):
    gate_p = dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  lambda_l2=1.0, seed=42, deterministic=True)
    mu_p = dict(num_leaves=31, learning_rate=0.04, min_data_in_leaf=30,
                feature_fraction=1.0, bagging_fraction=0.9, bagging_freq=1,
                lambda_l2=1.0, seed=43, deterministic=True)
    # dispersion: a variance-type parameter -- larger leaves, slower rate and
    # capped Newton steps (hessian can ride the 1e-6 floor on flat cells)
    th_p = dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=150,
                feature_fraction=1.0, bagging_fraction=0.9, bagging_freq=1,
                lambda_l2=2.0, max_delta_step=0.5, seed=44, deterministic=True)
    th_feats = list(FEATS)

    if kind == "A":
        pass
    elif kind == "B":  # one global sign vector, as stock LSS would apply
        for prm in (gate_p, mu_p, th_p):
            prm["monotone_constraints"] = mono_vec(FEATS, {TREAT: -1})
            prm["monotone_constraints_method"] = "advanced"
    elif kind == "C":  # per-parameter signs + price-free theta
        gate_p["monotone_constraints"] = mono_vec(FEATS, {TREAT: +1})
        gate_p["monotone_constraints_method"] = "advanced"
        mu_p["monotone_constraints"] = mono_vec(FEATS, {TREAT: -1})
        mu_p["monotone_constraints_method"] = "advanced"
        th_feats = [f for f in FEATS if f != TREAT]
    else:
        raise ValueError(kind)

    return ZANBModel(
        GateBooster(FEATS, gate_p, num_rounds=1000, early_stopping=150),
        ZTNBCyclicBooster(FEATS, th_feats, mu_p, th_p,
                          num_rounds=1000, early_stopping=150,
                          theta_updates_per_iter=1))


# ----------------------------------------------------------------------------
# Audits
# ----------------------------------------------------------------------------

def sweep_predict(model, base, col_grid):
    """Predict E[Y], params over the lr grid for each base row -> (n, G)."""
    n, Gn = len(base), len(col_grid)
    rep = base.loc[base.index.repeat(Gn)].copy()
    rep[TREAT] = np.tile(col_grid, n)
    pi, mu, th = model.predict_params(rep)
    E = zanb_mean(pi, mu, th)
    shp = (n, Gn)
    return E.reshape(shp), pi.reshape(shp), mu.reshape(shp), th.reshape(shp)


def true_sweep(base, grid):
    out = np.zeros((len(base), len(grid)))
    for i, (_, r) in enumerate(base.iterrows()):
        pi, mu, th = true_params(r.bunit, r.bunit_c, r.seasval,
                                 r.gunit, r.dunit, r.t2j, grid)
        out[i] = zanb_mean(pi, mu, th)
    return out


def violation_stats(E):
    """E: (n_sweeps, G). Violation = increase along grid beyond tolerance."""
    d = np.diff(E, axis=1)
    tol = 1e-8 * (1.0 + np.abs(E[:, :-1]))
    vmask = d > tol
    per_sweep_max = np.where(vmask, d, 0.0).max(axis=1)
    return {
        "n_sweeps": int(E.shape[0]),
        "sweeps_with_violation": int((vmask.any(axis=1)).sum()),
        "violation_rate": float(vmask.any(axis=1).mean()),
        "max_violation": float(per_sweep_max.max()),
        "median_violation_when_violated": float(
            np.median(per_sweep_max[per_sweep_max > 0])) if (per_sweep_max > 0).any() else 0.0,
    }, per_sweep_max


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    results = {"gradient_check": check_gradients(),
               "quantile_property_check": check_quantiles(),
               "numclass_shared_constraint_demo": numclass_shared_constraint_demo()}
    print("[1/6] gradient + quantile + num_class checks done "
          f"(ppf vs brute-force pmf inversion: "
          f"{results['quantile_property_check']['pairs_checked']} pairs, "
          f"{results['quantile_property_check']['mismatches']} mismatches)")

    df = simulate_panel()
    feat_ok = df[FEATS].notna().all(axis=1)
    df = df.loc[feat_ok & (df.week >= 15)].reset_index(drop=True)
    tr = df[df.week <= 104]
    va = df[(df.week >= 105) & (df.week <= 117)]
    te = df[df.week >= 118]
    results["panel"] = {
        "rows_train": len(tr), "rows_val": len(va), "rows_test": len(te),
        "zero_share_train": float((tr.y == 0).mean()),
        "y_max": int(df.y.max()),
        "unit_mean_volume_range": [float(tr.groupby('unit').y.mean().min()),
                                   float(tr.groupby('unit').y.mean().max())],
    }
    print(f"[2/6] panel simulated: {len(df)} rows, "
          f"zero share {results['panel']['zero_share_train']:.2f}")

    models, fit_sec = {}, {}
    results["ztnb_trainer"] = {}
    for kind in ("A", "B", "C"):
        ts = time.time()
        models[kind] = build_model(kind).fit(tr, tr.y.values, va, va.y.values)
        fit_sec[kind] = round(time.time() - ts, 1)
        z = models[kind].ztnb
        results["ztnb_trainer"][kind] = {
            "c_a": z.c_a, "c_b_log_theta0": z.c_b,
            "best_iter_mu": z.best_iter, "best_iter_theta": z.best_iter_th,
            "z0_val_ztnb_nll": z.z0_val_nll, "z1_val_ztnb_nll": z.z1_val_nll,
            "z1_accepted": z.z1_accepted, "phase1_iters": z.phase1_end,
        }
        print(f"[3/6] model {kind} fit in {fit_sec[kind]}s "
              f"(mu trees={z.best_iter}, theta trees={z.best_iter_th}, "
              f"Z0 nll={z.z0_val_nll:.4f} -> Z1 nll={z.z1_val_nll:.4f}, "
              f"Z1 accepted={z.z1_accepted})")
    results["fit_seconds"] = fit_sec

    # ---- sweeps -------------------------------------------------------------
    base = te[te.week.isin(SWEEP_WEEKS)].copy().reset_index(drop=True)
    sweeps, params_sw = {}, {}
    for kind in ("A", "B", "C"):
        E, pi, mu, th = sweep_predict(models[kind], base, LR_GRID)
        sweeps[kind] = E
        params_sw[kind] = (pi, mu, th)
    E_truth = true_sweep(base, LR_GRID)

    viol, vmax = {}, {}
    for kind in ("A", "B", "C"):
        viol[kind], vmax[kind] = violation_stats(sweeps[kind])
    vt, _ = violation_stats(E_truth)
    results["violations"] = {**{k: viol[k] for k in viol}, "truth_DGP": vt}

    # structural channel checks for C: raw scores monotone, theta flat
    piC, muC, thC = params_sw["C"]
    assert np.all(np.diff(piC, axis=1) >= -1e-12), "C gate not non-decreasing"
    assert np.all(np.diff(muC, axis=1) <= 1e-12), "C mu not non-increasing"
    assert np.allclose(np.diff(thC, axis=1), 0.0, atol=1e-12), "C theta moved with price"

    # quantile monotonicity for C (stochastic dominance consequence)
    qviol = {}
    for q in (0.5, 0.9, 0.99):
        Q = np.stack([zanb_ppf(q, piC[:, g], muC[:, g], thC[:, g])
                      for g in range(len(LR_GRID))], axis=1)
        qviol[str(q)] = int((np.diff(Q, axis=1) > 1e-9).sum())
    results["quantile_violations_C"] = qviol

    # extrapolation: guarantee must hold OUTSIDE the training support too
    grid_ext = np.linspace(-0.60, 0.60, 49)
    results["violations_extended_grid"] = {}
    for kind in ("A", "B", "C"):
        E_ext, _, _, _ = sweep_predict(models[kind], base, grid_ext)
        vext, _ = violation_stats(E_ext)
        results["violations_extended_grid"][kind] = vext
    print(f"[4/6] sweeps done: violations A={viol['A']['violation_rate']:.1%} "
          f"B={viol['B']['violation_rate']:.1%} C={viol['C']['violation_rate']:.1%} "
          f"| extended grid C={results['violations_extended_grid']['C']['violation_rate']:.1%}")

    # ---- fit quality --------------------------------------------------------
    rngp = np.random.default_rng(99)
    metrics = {}
    pit = {}
    for kind in ("A", "B", "C"):
        m = models[kind]
        pi, mu, th = m.predict_params(te)
        E = zanb_mean(pi, mu, th)
        metrics[kind] = {
            "test_nll": m.nll(te, te.y.values),
            "rmse_vs_y": float(np.sqrt(np.mean((E - te.y.values) ** 2))),
            "mae_vs_y": float(np.mean(np.abs(E - te.y.values))),
            "rmse_vs_true_mean": float(np.sqrt(np.mean((E - te.E_true.values) ** 2))),
            "rmsle_vs_true_mean": float(np.sqrt(np.mean(
                (np.log1p(E) - np.log1p(te.E_true.values)) ** 2))),
            "rmse_vs_true_mean_trap_units": float(np.sqrt(np.mean(
                (E[te.is_trap.values] - te.E_true.values[te.is_trap.values]) ** 2))),
            # flexibility: parameters must still vary (correctly) with regressors
            "corr_logit_pi": float(np.corrcoef(
                np.log(pi / (1 - pi)),
                np.log(te.pi_true.values / (1 - te.pi_true.values)))[0, 1]),
            "corr_log_mu": float(np.corrcoef(np.log(mu), np.log(te.mu_true.values))[0, 1]),
            "corr_log_theta": float(np.corrcoef(np.log(th), np.log(te.th_true.values))[0, 1]),
            "std_log_theta_hat": float(np.std(np.log(th))),
        }
        pit[kind] = zanb_pit(rngp, te.y.values.astype(float), pi, mu, th)
    from zanb_lss import zanb_nll
    metrics["oracle"] = {"test_nll": float(np.mean(zanb_nll(
        te.y.values.astype(float), te.pi_true.values, te.mu_true.values,
        te.th_true.values)))}
    results["test_metrics"] = metrics

    # implied elasticity at f = 1 (d log E / d lr), from sweeps
    iL = np.argmin(np.abs(LR_GRID + 0.05)); iR = np.argmin(np.abs(LR_GRID - 0.05))
    dlr = LR_GRID[iR] - LR_GRID[iL]
    elas = {k: (np.log(sweeps[k][:, iR]) - np.log(sweeps[k][:, iL])) / dlr
            for k in ("A", "B", "C")}
    elas["truth"] = (np.log(E_truth[:, iR]) - np.log(E_truth[:, iL])) / dlr
    results["unit_elasticity_corr_with_truth"] = {
        k: float(np.corrcoef(elas[k], elas["truth"])[0, 1]) for k in ("A", "B", "C")}

    # product-level elasticity (pooled across pharmacies / sweep weeks)
    prod_of_base = base["prod"].values
    elas_prod = {}
    for k in ("A", "B", "C", "truth"):
        S = sweeps[k] if k != "truth" else E_truth
        agg = pd.DataFrame(S).groupby(prod_of_base).sum().to_numpy()
        elas_prod[k] = (np.log(agg[:, iR]) - np.log(agg[:, iL])) / dlr
    results["product_elasticity_corr_with_truth"] = {
        k: float(np.corrcoef(elas_prod[k], elas_prod["truth"])[0, 1])
        for k in ("A", "B", "C")}

    # aggregate demand curve: normalised mean response across all sweeps
    aggcurve = {k: (sweeps[k] / sweeps[k][:, [len(LR_GRID) // 2]]).mean(axis=0)
                for k in ("A", "B", "C")}
    aggcurve["truth"] = (E_truth / E_truth[:, [len(LR_GRID) // 2]]).mean(axis=0)
    print("[5/6] metrics done")

    # ------------------------------------------------------------------ plots
    fmult = np.exp(LR_GRID)

    # F1: price-response curves, 6 illustrative units
    bs = base.copy()
    bs["volume"] = np.expm1(bs.rate13)
    picks, used = [], set()

    def pick(mask, by, asc=False, label=""):
        cand = bs[mask & ~bs.index.isin(used)]
        if len(cand) == 0:
            cand = bs[~bs.index.isin(used)]
        idx = cand[by].idxmin() if asc else cand[by].idxmax()
        used.add(idx)
        picks.append((idx, label))

    pick((bs.gunit < -0.8) & (bs.week == SWEEP_WEEKS[0]), "volume", False,
         "high volume, strong elasticity")
    pick((bs.volume > 1) & (bs.week == SWEEP_WEEKS[0]), "dunit", False,
         "gate-dominated unit")
    pick(bs.is_trap & (bs.volume > 0.3) & (bs.week == SWEEP_WEEKS[0]), "volume", False,
         "trap unit: near-zero elasticity,\ntheta falls with price (truth non-monotone)")
    pick((bs.zshare13 > 0.5) & (bs.week == SWEEP_WEEKS[0]), "volume", False,
         "intermittent low-volume unit")
    med_vol = bs.volume.median()
    bs["dmed"] = (bs.volume - med_vol).abs()
    pick(bs.week == SWEEP_WEEKS[0], "dmed", True, "median unit")
    pick(bs.is_trap & (bs.week == SWEEP_WEEKS[1]), "dunit", False,
         "trap unit #2 (strong gate)")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    for ax, (idx, label) in zip(axes.ravel(), picks):
        i = base.index.get_loc(idx)
        ax.plot(fmult, E_truth[i], color=CLR["truth"], ls="--", lw=1.4, label="true E[Y]")
        for kind in ("A", "B", "C"):
            ax.plot(fmult, sweeps[kind][i], color=CLR[kind], lw=1.5, label=LBL[kind])
            d = np.diff(sweeps[kind][i])
            bad = np.where(d > 1e-8 * (1 + np.abs(sweeps[kind][i][:-1])))[0]
            if len(bad):
                ax.scatter(fmult[bad + 1], sweeps[kind][i][bad + 1], s=14,
                           color=CLR[kind], marker="x", zorder=5)
        ax.set_title(label, fontsize=8.6)
        ax.set_xlim(fmult[0], fmult[-1])
    for ax in axes[1]:
        ax.set_xlabel("counterfactual price multiplier")
    for ax in axes[:, 0]:
        ax.set_ylabel("E[weekly units]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Counterfactual price response of the predictive mean "
                 "(x = violation of monotonicity)", y=1.12, fontsize=11)
    fig.savefig(os.path.join(FIGDIR, "f1_price_response_curves.png"),
                bbox_inches="tight")
    plt.close(fig)

    # F2: violation audit over all sweeps
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ks = ["A", "B", "C"]
    rates = [viol[k]["violation_rate"] * 100 for k in ks]
    bars = axes[0].bar(ks, rates, color=[CLR[k] for k in ks], width=0.55)
    for b, k in zip(bars, ks):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                     f"{viol[k]['sweeps_with_violation']}/{viol[k]['n_sweeps']}",
                     ha="center", fontsize=8.5)
    axes[0].set_ylabel("% of unit sweeps with any\nmean increase in price")
    axes[0].set_title(f"Monotonicity violations across all {viol['A']['n_sweeps']} "
                      f"test-unit sweeps\n(A unconstrained | B shared sign | C per-parameter)")

    for k in ("A", "B"):
        v = np.sort(vmax[k][vmax[k] > 0])
        if len(v):
            axes[1].plot(v, np.linspace(0, 1, len(v)), color=CLR[k], lw=1.6,
                         label=f"{LBL[k]} (n={len(v)})")
    axes[1].set_xscale("log")
    axes[1].axvline(1e-8, color=CLR["C"], lw=1.2, ls=":")
    axes[1].text(1.4e-8, 0.45, "C: zero violations\n(machine exact)", fontsize=8,
                 color=CLR["C"], rotation=90, va="center")
    axes[1].set_xlabel("largest single-step mean increase within sweep [units]")
    axes[1].set_ylabel("ECDF over violating sweeps")
    axes[1].set_title("Violation magnitudes")
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIGDIR, "f2_violation_audit.png"), bbox_inches="tight")
    plt.close(fig)

    # F3: stochastic dominance for model C (gate-dominated unit).
    # NB the quantile fan of a DISCRETE hurdle law is genuinely a staircase:
    # quantiles live on integers, the median pins at 1 while F(0) < 0.5, q25
    # drops to 0 exactly where P(Y=0) crosses 25%, and the mean may sit below
    # the integer median (zero atom + thin right tail). Verified against
    # brute-force pmf inversion in check_quantiles().
    i3 = base.index.get_loc(picks[1][0])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    qs = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95]
    piC_, muC_, thC_ = (params_sw["C"][j][i3] for j in range(3))
    fan = {q: np.array([zanb_ppf(q, np.array([piC_[g]]), np.array([muC_[g]]),
                                 np.array([thC_[g]]))[0] for g in range(len(LR_GRID))])
           for q in qs}
    axes[0].fill_between(fmult, fan[0.05], fan[0.95], color=CLR["C"], alpha=0.12,
                         step="post", label="90% central interval (q05–q95)")
    axes[0].fill_between(fmult, fan[0.25], fan[0.75], color=CLR["C"], alpha=0.30,
                         step="post", label="50% central interval (q25–q75)")
    for q, ls in [(0.95, ":"), (0.9, "--")]:
        axes[0].plot(fmult, fan[q], color=CLR["C"], ls=ls, lw=1.1,
                     drawstyle="steps-post", label=f"q{int(q * 100)}")
    axes[0].plot(fmult, fan[0.5], color=CLR["C"], lw=1.7,
                 drawstyle="steps-post", label="median")
    axes[0].plot(fmult, sweeps["C"][i3], color="k", lw=1.6, label="mean")
    g25 = int(np.argmax(piC_ > 0.25))
    if piC_[g25] > 0.25:
        axes[0].axvline(fmult[g25], color="#888888", lw=0.9, ls=":")
        axes[0].text(fmult[g25] + 0.008, 0.55 * float(fan[0.95].max()),
                     "P(Y=0) crosses 25%:\nq25 drops to 0", fontsize=7.2,
                     color="#555555")
    axes[0].set_xlabel("price multiplier")
    axes[0].set_ylabel("weekly units")
    axes[0].set_title("Model C: integer-quantile fan, nested central intervals\n"
                      f"every quantile non-increasing; P(Y=0) rises "
                      f"{piC_[0]:.2f}→{piC_[-1]:.2f} so the median pins at 1",
                      fontsize=9.0)
    axes[0].legend(frameon=False, ncol=2, fontsize=7.0)

    for f, c in [(0.75, "#1b7837"), (1.0, "#762a83"), (1.30, "#d73027")]:
        g = np.argmin(np.abs(fmult - f))
        ys = np.arange(0, max(8, int(zanb_ppf(0.99, np.array([piC_[g]]),
                                              np.array([muC_[g]]),
                                              np.array([thC_[g]]))[0]) + 2))
        F = zanb_cdf(ys, piC_[g], muC_[g], thC_[g])
        axes[1].step(ys, F, where="post", color=c, lw=1.5,
                     label=f"price x{fmult[g]:.2f}")
    axes[1].set_xlabel("y (weekly units)")
    axes[1].set_ylabel("F(y)")
    axes[1].set_title("Predictive CDFs ordered by price:\nhigher price = CDF uniformly above")
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIGDIR, "f3_stochastic_dominance.png"), bbox_inches="tight")
    plt.close(fig)

    # F4: parameter channels at the trap unit
    i4 = base.index.get_loc(picks[2][0])
    r4 = base.loc[picks[2][0]]
    pi_t, mu_t, th_t = true_params(r4.bunit, r4.bunit_c, r4.seasval, r4.gunit,
                                   r4.dunit, r4.t2j, LR_GRID)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    chans = [("gate pi = P(zero)", pi_t, 0), ("conditional mean mu/(1-p0)", None, None),
             ("dispersion theta", th_t, 2)]
    axes[0].plot(fmult, pi_t, "k--", lw=1.4, label="truth")
    axes[1].plot(fmult, ztnb_mean(mu_t, th_t), "k--", lw=1.4)
    axes[2].plot(fmult, th_t, "k--", lw=1.4)
    for kind in ("A", "B", "C"):
        pi_, mu_, th_ = (params_sw[kind][j][i4] for j in range(3))
        axes[0].plot(fmult, pi_, color=CLR[kind], lw=1.5, label=LBL[kind])
        axes[1].plot(fmult, ztnb_mean(mu_, th_), color=CLR[kind], lw=1.5)
        axes[2].plot(fmult, th_, color=CLR[kind], lw=1.5)
    axes[0].set_title("gate pi(price): truth RISES with price;\nB's shared sign forbids rising")
    axes[1].set_title("conditional mean E[Y|Y>0]: truth RISES\nwith price (theta-truncation channel)")
    axes[2].set_title("dispersion theta(price): C excludes\nprice by design (flat line)")
    for ax in axes:
        ax.set_xlabel("price multiplier")
    axes[0].legend(frameon=False, fontsize=7.2)
    fig.suptitle("The trap unit, channel by channel: two structural holes of the shared-sign constraint",
                 y=1.06, fontsize=10.5)
    fig.savefig(os.path.join(FIGDIR, "f4_parameter_channels.png"), bbox_inches="tight")
    plt.close(fig)

    # F5: skill / cost of constraints
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3))
    ks = ["A", "B", "C"]
    nlls = [metrics[k]["test_nll"] for k in ks]
    axes[0].bar([k for k in ks], nlls, color=[CLR[k] for k in ks], width=0.55)
    axes[0].axhline(metrics["oracle"]["test_nll"], color="k", ls="--", lw=1.1)
    axes[0].text(0.02, metrics["oracle"]["test_nll"], "oracle (true params)",
                 fontsize=7.5, va="bottom")
    axes[0].set_ylim(min(nlls + [metrics['oracle']['test_nll']]) * 0.995,
                     max(nlls) * 1.004)
    axes[0].set_title("test NLL (lower=better)")

    r_log = [metrics[k]["rmsle_vs_true_mean"] for k in ks]
    axes[1].bar(ks, r_log, color=[CLR[k] for k in ks], width=0.55)
    axes[1].set_title("RMSLE of E-hat vs TRUE mean\n(scale-fair oracle gap)")

    for k in ("truth", "A", "B", "C"):
        sty = dict(color=CLR.get(k, "#000"), lw=1.6)
        if k == "truth":
            sty.update(ls="--", color="k")
        axes[2].plot(fmult, aggcurve[k], label=k, **sty)
    pc = results["product_elasticity_corr_with_truth"]
    axes[2].set_xlabel("price multiplier")
    axes[2].set_ylabel("mean E(f) / E(1) across units")
    axes[2].set_title("aggregate demand curve recovery\n"
                      f"product-level elasticity corr: A={pc['A']:.2f}, "
                      f"B={pc['B']:.2f}, C={pc['C']:.2f}")
    axes[2].legend(frameon=False, fontsize=7.5)

    for kind in ("A", "B", "C"):
        z = models[kind].ztnb
        h = z.history
        axes[3].plot(np.arange(1, len(h) + 1), h, color=CLR[kind], lw=1.3,
                     label=f"{kind} (mu {z.best_iter}, th {z.best_iter_th})")
        axes[3].axvline(z.phase1_end, color=CLR[kind], lw=0.8, ls=":", alpha=0.7)
    axes[3].set_xlabel("outer iteration (dotted: Z0 -> Z1 hand-over)")
    axes[3].set_ylabel("val ZTNB NLL")
    axes[3].set_title("two-phase training: Z0 (profiled scalar theta)\nthen Z1 (cyclic theta booster)")
    axes[3].legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIGDIR, "f5_skill_and_cost.png"), bbox_inches="tight")
    plt.close(fig)

    # F6: calibration -- pooled PIT, gate calibration, PIT by tier, coverage
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.2))
    for ax, kind in zip(axes[0, :3], ("A", "B", "C")):
        ax.hist(pit[kind], bins=20, range=(0, 1), color=CLR[kind], alpha=0.85,
                density=True)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_title(f"PIT: {kind}")
        ax.set_xlabel("u")
        ax.set_ylim(0, 1.6)
    bins = np.quantile(te.lr, np.linspace(0, 1, 9))
    bid = np.clip(np.searchsorted(bins, te.lr, side="right") - 1, 0, 7)
    centers, obs = [], []
    for bq in range(8):
        m = bid == bq
        centers.append(np.exp(te.lr[m].mean()))
        obs.append((te.y[m] == 0).mean())
    axes[0, 3].plot(centers, obs, "ko", ms=4.5, label="observed zero share")
    for kind in ("A", "B", "C"):
        pihat = models[kind].gate.predict_pi(te)
        prd = [pihat[bid == bq].mean() for bq in range(8)]
        axes[0, 3].plot(centers, prd, color=CLR[kind], lw=1.5, marker=".",
                        label=kind)
    axes[0, 3].set_xlabel("realised price ratio bin (exp lr)")
    axes[0, 3].set_ylabel("P(zero week)")
    axes[0, 3].set_title("zero-rate vs price:\nB structurally cannot rise")
    axes[0, 3].legend(frameon=False, fontsize=7.2)

    # row 2: model C PIT by volume tier (G5, the theta-misfit detector)
    tier_edges = np.quantile(te.E_true.values, [0.0, 1 / 3, 2 / 3, 1.0])
    tier_id = np.clip(np.searchsorted(tier_edges, te.E_true.values,
                                      side="right") - 1, 0, 2)
    tier_names = ["low-volume tier", "mid-volume tier", "high-volume tier"]
    g5 = {"n_bins": 20, "band": [0.5, 1.5], "tiers": {}}
    for tnum, (ax, nm) in enumerate(zip(axes[1, :3], tier_names)):
        u = pit["C"][tier_id == tnum]
        dens, _, _ = ax.hist(u, bins=20, range=(0, 1), color=CLR["C"],
                             alpha=0.85, density=True)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.axhspan(0.5, 1.5, color="k", alpha=0.05)
        g5["tiers"][nm] = {"min_bin": float(dens.min()),
                           "max_bin": float(dens.max()),
                           "pass": bool(dens.min() >= 0.5 and dens.max() <= 1.5)}
        ax.set_title(f"PIT C, {nm} (n={int((tier_id == tnum).sum())})")
        ax.set_xlabel("u")
        ax.set_ylim(0, 1.8)
    g5["pass"] = bool(all(t["pass"] for t in g5["tiers"].values()))
    results["pit_tier_uniformity_G5_C"] = g5

    noms = np.array([0.5, 0.8, 0.9, 0.95])
    for kind in ("A", "B", "C"):
        cov = [float(np.mean((pit[kind] > (1 - nm) / 2)
                             & (pit[kind] <= 1 - (1 - nm) / 2))) for nm in noms]
        axes[1, 3].plot(noms, cov, marker="o", ms=3.5, color=CLR[kind],
                        lw=1.4, label=kind)
    axes[1, 3].plot([0.45, 1.0], [0.45, 1.0], "k--", lw=1)
    axes[1, 3].set_xlabel("nominal central coverage")
    axes[1, 3].set_ylabel("empirical coverage")
    axes[1, 3].set_title("central-interval coverage\n(randomised PIT)")
    axes[1, 3].legend(frameon=False, fontsize=7.2)
    fig.savefig(os.path.join(FIGDIR, "f6_calibration.png"), bbox_inches="tight")
    plt.close(fig)

    # F7: price coarse-graining diagnostic at production scale (3000 products)
    rng7 = np.random.default_rng(123)
    n_prod7, n_obs7 = 3000, 104
    p0_7 = np.exp(rng7.uniform(np.log(20), np.log(3000), n_prod7))
    ratio7 = np.clip(rng7.normal(0, 0.05, (n_prod7, n_obs7))
                     - rng7.uniform(0.12, 0.4, (n_prod7, 1))
                     * (rng7.uniform(size=(n_prod7, n_obs7)) < 0.10), -0.5, 0.35)
    prices7 = p0_7[:, None] * np.exp(ratio7)
    edges_raw = np.unique(np.quantile(prices7.ravel(), np.linspace(0, 1, 256)))
    edges_lr = np.unique(np.quantile(ratio7.ravel(), np.linspace(0, 1, 256)))
    pmed = np.median(prices7, axis=1)
    shift = 0.05
    crossings_raw = np.array([
        ((edges_raw > p) & (edges_raw <= p * (1 + shift))).sum() for p in pmed])
    cross_lr = int(((edges_lr > 0) & (edges_lr <= np.log(1 + shift))).sum())
    pct0 = float((crossings_raw == 0).mean() * 100)
    pct1 = float((crossings_raw <= 1).mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    kgrid = np.arange(0, 6)
    share_raw = [(crossings_raw <= k).mean() * 100 for k in kgrid]
    axes[0].bar(kgrid, share_raw, width=0.6, color="#777777",
                label="raw price feature")
    for k, v in zip(kgrid, share_raw):
        axes[0].text(k, v + 2.0, f"{v:.0f}%", ha="center", fontsize=7.5)
    axes[0].axhline(0.0, color=CLR["C"], lw=1.8)
    axes[0].text(0.0, 6.0, f"log price-ratio feature: {cross_lr} edges for every "
                 "product\n(share with <= 5 edges = 0%)", fontsize=7.5,
                 color=CLR["C"])
    axes[0].set_xticks(kgrid)
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("K = bin edges inside a +5% price move")
    axes[0].set_ylabel("% of products with <= K edges")
    axes[0].set_title("Resolution of a +5% price move\n"
                      "(255 global equal-frequency bins each)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[1].scatter(pmed, np.maximum(crossings_raw, 0.4), s=4, alpha=0.3,
                    color="#777777", label="raw price feature")
    axes[1].axhline(cross_lr, color=CLR["C"], lw=1.8,
                    label=f"price-ratio feature: {cross_lr} edges, every product")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("product price level")
    axes[1].set_ylabel("# bin edges inside +5% shift")
    axes[1].set_title("Resolution of relative price moves by price tier")
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIGDIR, "f7_binning_diagnostic.png"), bbox_inches="tight")
    plt.close(fig)

    results["binning_3000_products"] = {
        "pct_products_5pct_shift_zero_edges_raw": pct0,
        "pct_products_5pct_shift_at_most_one_edge_raw": pct1,
        "median_edges_in_5pct_shift_raw": float(np.median(crossings_raw)),
        "edges_in_5pct_shift_ratio_feature": cross_lr,
    }
    results["runtime_seconds_total"] = round(time.time() - t0, 1)

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[6/6] figures + results.json written")
    print(json.dumps({k: results[k] for k in
                      ("violations", "violations_extended_grid",
                       "quantile_violations_C", "test_metrics",
                       "unit_elasticity_corr_with_truth",
                       "product_elasticity_corr_with_truth",
                       "binning_3000_products")}, indent=2))


if __name__ == "__main__":
    main()
