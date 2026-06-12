"""Diagnose the theta (shape) misfit: corr(log th_hat, log th_true) = -0.91.

Hypotheses tested on the same panel/splits as run_experiment.py:
  V0  baseline cyclic trainer dynamics, with per-iteration snapshots of
      corr_log_mu / corr_log_theta on test + gradient norms -> mechanism
  V1  theta frozen at pooled start (Z0): does NLL/calibration survive?
  V2  oracle-mu: theta booster fit against TRUE mu -> is theta identifiable?
  V3  mu burn-in (theta frozen) until val plateau, then cyclic -> the fix?
  V4  unconstrained pair: does the theta booster ever split on lr?
"""

import json
import os
import sys

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zanb_lss import (ztnb_grad_hess_a, ztnb_grad_hess_b, ztnb_logpmf,
                      ztnb_score_a, ztnb_score_b, _clip_ab)
from run_experiment import simulate_panel, FEATS, TREAT


def ztnb_start_values(y):
    """LEGACY pooled unconditional ZTNB MLE -- kept here verbatim because this
    script exists to reproduce the failure it causes: pooling heterogeneous
    units drives (mu, theta) to the degenerate ridge corner (c_b hits the -5
    bound), all NB2 gradients die, and the theta booster anti-learns.
    The production initialiser is zanb_lss.ztnb_profile_theta (Z0 phase)."""
    y = np.asarray(y, dtype=float)

    def fun(v):
        a = np.full_like(y, v[0])
        b = np.full_like(y, v[1])
        aa, bb = _clip_ab(a, b)
        mu, th = np.exp(aa), np.exp(bb)
        nll = -np.sum(ztnb_logpmf(y, mu, th))
        ga = -np.sum(ztnb_score_a(y, mu, th)[0])
        gb = -np.sum(ztnb_score_b(y, mu, th)[0])
        return nll, np.array([ga, gb])

    a0 = np.log(max(y.mean(), 0.2))
    res = minimize(fun, x0=np.array([a0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(-10.0, 12.0), (-5.0, 5.0)])
    return float(res.x[0]), float(res.x[1])

OUT = {}


def panel():
    df = simulate_panel()
    feat_ok = df[FEATS].notna().all(axis=1)
    df = df.loc[feat_ok & (df.week >= 15)].reset_index(drop=True)
    tr = df[df.week <= 104]
    va = df[(df.week >= 105) & (df.week <= 117)]
    te = df[df.week >= 118]
    return tr, va, te


def cyclic_fit(tr_pos, va_pos, te, feat_mu, feat_th, mu_p, th_p,
               num_rounds=400, patience=150, theta_updates=2,
               burn_in=0, freeze_theta=False, oracle_mu=False,
               snap_iters=(1, 2, 3, 5, 8, 12, 18, 26, 40, 60, 90, 130, 200, 300, 400)):
    ytr = tr_pos.y.values.astype(float)
    yva = va_pos.y.values.astype(float)
    Xtr_mu = np.ascontiguousarray(tr_pos[feat_mu].to_numpy(np.float64))
    Xtr_th = np.ascontiguousarray(tr_pos[feat_th].to_numpy(np.float64))
    Xva_mu = np.ascontiguousarray(va_pos[feat_mu].to_numpy(np.float64))
    Xva_th = np.ascontiguousarray(va_pos[feat_th].to_numpy(np.float64))
    Xte_mu = np.ascontiguousarray(te[feat_mu].to_numpy(np.float64))
    Xte_th = np.ascontiguousarray(te[feat_th].to_numpy(np.float64))

    c_a, c_b = ztnb_start_values(ytr)
    a_true_tr = np.log(tr_pos.mu_true.values)
    a_true_va = np.log(va_pos.mu_true.values)

    bst_mu = lgb.Booster(params={**mu_p, "objective": "none", "verbosity": -1},
                         train_set=lgb.Dataset(Xtr_mu, label=ytr, free_raw_data=False))
    bst_th = lgb.Booster(params={**th_p, "objective": "none", "verbosity": -1},
                         train_set=lgb.Dataset(Xtr_th, label=ytr, free_raw_data=False))

    Fa = np.zeros(len(ytr)); Fb = np.zeros(len(ytr))
    Va = np.zeros(len(yva)); Vb = np.zeros(len(yva))
    Ta = np.zeros(len(te)); Tb = np.zeros(len(te))

    lth_true_te = np.log(te.th_true.values)
    lmu_true_te = np.log(te.mu_true.values)

    hist, snaps = [], []
    grad_norms = []
    best, best_it, since, n_th = np.inf, 0, 0, 0
    for it in range(1, num_rounds + 1):
        a_tr = a_true_tr if oracle_mu else Fa + c_a
        if not oracle_mu:
            ga, ha = ztnb_grad_hess_a(ytr, Fa + c_a, Fb + c_b)
            grad_norms.append([float(np.mean(np.abs(ga))), float(np.exp(np.mean(Fb)) * np.exp(c_b))])

            def fobj_mu(preds, data):
                return ztnb_grad_hess_a(ytr, preds + c_a, Fb + c_b)
            bst_mu.update(fobj=fobj_mu)
            Fa += bst_mu.predict(Xtr_mu, raw_score=True, start_iteration=it - 1, num_iteration=1)
            Va += bst_mu.predict(Xva_mu, raw_score=True, start_iteration=it - 1, num_iteration=1)
            Ta += bst_mu.predict(Xte_mu, raw_score=True, start_iteration=it - 1, num_iteration=1)

        theta_now = (not freeze_theta) and (it > burn_in)
        if theta_now:
            for _ in range(theta_updates):
                def fobj_th(preds, data):
                    return ztnb_grad_hess_b(ytr, a_tr, preds + c_b)
                bst_th.update(fobj=fobj_th)
                Fb += bst_th.predict(Xtr_th, raw_score=True, start_iteration=n_th, num_iteration=1)
                Vb += bst_th.predict(Xva_th, raw_score=True, start_iteration=n_th, num_iteration=1)
                Tb += bst_th.predict(Xte_th, raw_score=True, start_iteration=n_th, num_iteration=1)
                n_th += 1

        a_v = a_true_va if oracle_mu else Va + c_a
        av, bv = _clip_ab(a_v, Vb + c_b)
        nll = float(-np.mean(ztnb_logpmf(yva, np.exp(av), np.exp(bv))))
        hist.append(nll)
        if it in snap_iters:
            at, bt = _clip_ab(Ta + c_a, Tb + c_b)
            snaps.append({
                "iter": it,
                "corr_log_mu": float(np.corrcoef(at, lmu_true_te)[0, 1]),
                "corr_log_theta": float(np.corrcoef(bt, lth_true_te)[0, 1]) if np.std(bt) > 0 else None,
                "mean_log_theta_hat": float(np.mean(bt)),
                "std_log_theta_hat": float(np.std(bt)),
            })
        if nll < best - 1e-7:
            best, best_it, since = nll, it, 0
        else:
            since += 1
            if since >= patience:
                break

    at, bt = _clip_ab(Ta + c_a, Tb + c_b)
    res = {
        "c_a": c_a, "c_b": c_b, "stopped_at": it, "best_iter": best_it,
        "best_val_ztnb_nll": best, "final_val_ztnb_nll": hist[-1],
        "corr_log_mu_test": float(np.corrcoef(at, lmu_true_te)[0, 1]),
        "corr_log_theta_test": float(np.corrcoef(bt, lth_true_te)[0, 1]) if np.std(bt) > 0 else None,
        "mean_log_theta_hat": float(np.mean(bt)),
        "mean_log_theta_true": float(np.mean(lth_true_te)),
        "std_log_theta_hat": float(np.std(bt)),
        "std_log_theta_true": float(np.std(lth_true_te)),
        "snapshots": snaps,
    }
    if not oracle_mu:
        # theta residual vs mu residual coupling
        mu_res = at - lmu_true_te
        th_res = bt - lth_true_te
        res["corr(mu_resid, th_resid)"] = float(np.corrcoef(mu_res, th_res)[0, 1])
        res["grad_a_mean_abs_first_30"] = grad_norms[:30:3]
        imp = bst_th.feature_importance("gain")
        names = feat_th
        res["theta_gain_by_feature"] = {n: float(g) for n, g in
                                        sorted(zip(names, imp), key=lambda t: -t[1]) if g > 0}
    return res


def main():
    tr, va, te = panel()
    tr_pos = tr[tr.y > 0]
    va_pos = va[va.y > 0]
    mu_p = dict(num_leaves=31, learning_rate=0.04, min_data_in_leaf=30,
                feature_fraction=1.0, bagging_fraction=0.9, bagging_freq=1,
                lambda_l2=1.0, seed=43, deterministic=True,
                monotone_constraints=[(-1 if f == TREAT else 0) for f in FEATS],
                monotone_constraints_method="advanced")
    th_p = dict(num_leaves=31, learning_rate=0.08, min_data_in_leaf=60,
                feature_fraction=1.0, bagging_fraction=0.9, bagging_freq=1,
                lambda_l2=2.0, seed=44, deterministic=True)
    th_feats_C = [f for f in FEATS if f != TREAT]

    print("== V0 baseline cyclic (C config) ==")
    OUT["V0_baseline"] = cyclic_fit(tr_pos, va_pos, te, FEATS, th_feats_C, mu_p, th_p)
    print(json.dumps(OUT["V0_baseline"], indent=2, default=str))

    print("== V1 theta frozen at pooled start (Z0) ==")
    OUT["V1_frozen_theta"] = cyclic_fit(tr_pos, va_pos, te, FEATS, th_feats_C,
                                        mu_p, th_p, freeze_theta=True)
    print(json.dumps({k: v for k, v in OUT["V1_frozen_theta"].items()
                      if k != "snapshots"}, indent=2, default=str))

    print("== V2 oracle-mu, theta only ==")
    OUT["V2_oracle_mu"] = cyclic_fit(tr_pos, va_pos, te, FEATS, th_feats_C,
                                     mu_p, th_p, oracle_mu=True, theta_updates=1)
    print(json.dumps({k: v for k, v in OUT["V2_oracle_mu"].items()
                      if k != "snapshots"}, indent=2, default=str))

    print("== V3 mu burn-in 150 rounds, then cyclic ==")
    OUT["V3_burnin"] = cyclic_fit(tr_pos, va_pos, te, FEATS, th_feats_C,
                                  mu_p, th_p, burn_in=150, theta_updates=1,
                                  num_rounds=600, patience=200)
    print(json.dumps({k: v for k, v in OUT["V3_burnin"].items()
                      if k != "snapshots"}, indent=2, default=str))

    print("== V4 unconstrained pair, lr available to theta ==")
    mu_p_u = {k: v for k, v in mu_p.items()
              if k not in ("monotone_constraints", "monotone_constraints_method")}
    OUT["V4_uncon_lr_theta"] = cyclic_fit(tr_pos, va_pos, te, FEATS, FEATS,
                                          mu_p_u, th_p)
    print(json.dumps({k: v for k, v in OUT["V4_uncon_lr_theta"].items()
                      if k != "snapshots"}, indent=2, default=str))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "diagnose_theta.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=str)


if __name__ == "__main__":
    main()
