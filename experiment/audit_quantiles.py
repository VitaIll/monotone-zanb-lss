"""Audit: are the F3 quantiles correct?

1. Property test of zanb_ppf against a brute-force quantile computed from the
   pmf itself (ztnb_logpmf + zero atom; no nbinom.cdf/ppf anywhere) over a
   wide random parameter sweep.
2. Exact re-audit of the F3 unit: refit model C deterministically, recompute
   the fan, compare every (grid point, q) against brute force, and verify the
   left (fan) and right (CDF) panels agree with each other.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zanb_lss import zanb_ppf, zanb_cdf, zanb_mean, ztnb_logpmf
from run_experiment import simulate_panel, build_model, FEATS, TREAT, LR_GRID, SWEEP_WEEKS


def brute_quantile(q, pi, mu, th, kmax=4000):
    """Smallest y with F(y) >= q, F built by cumulating the exact pmf."""
    if q <= pi:
        return 0.0
    while True:
        y = np.arange(1, kmax + 1, dtype=float)
        pmf_pos = (1.0 - pi) * np.exp(ztnb_logpmf(y, mu, th))
        cdf = pi + np.cumsum(pmf_pos)
        if cdf[-1] >= q + 1e-12:        # support long enough for this q
            return float(y[np.searchsorted(cdf, q, side="left")])
        kmax *= 8
        assert kmax <= 5_000_000, "tail too heavy for brute force"


def main():
    rng = np.random.default_rng(7)
    # --- 1. property test --------------------------------------------------
    worst = 0
    n_checked = 0
    for _ in range(4000):
        pi = float(rng.uniform(0.01, 0.9))
        mu = float(np.exp(rng.uniform(-2.5, 4.0)))
        th = float(np.exp(rng.uniform(-2.0, 2.5)))
        for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
            a = float(zanb_ppf(q, np.array([pi]), np.array([mu]), np.array([th]))[0])
            b = brute_quantile(q, pi, mu, th)
            n_checked += 1
            if a != b:
                worst += 1
                if worst <= 10:
                    print(f"MISMATCH pi={pi:.4f} mu={mu:.4f} th={th:.4f} "
                          f"q={q}: ppf={a} brute={b}")
    print(f"[1] property test: {n_checked} (param, q) pairs, mismatches = {worst}")

    # --- 2. the exact F3 unit ----------------------------------------------
    df = simulate_panel()
    df = df[df[FEATS].notna().all(axis=1) & (df.week >= 15)].reset_index(drop=True)
    tr = df[df.week <= 104]
    va = df[(df.week >= 105) & (df.week <= 117)]
    te = df[df.week >= 118]
    model = build_model("C").fit(tr, tr.y.values, va, va.y.values)

    base = te[te.week.isin(SWEEP_WEEKS)].copy().reset_index(drop=True)
    bs = base.copy()
    bs["volume"] = np.expm1(bs.rate13)
    used = {bs[(bs.gunit < -0.8) & (bs.week == SWEEP_WEEKS[0])]["volume"].idxmax()}
    cand = bs[(bs.volume > 1) & (bs.week == SWEEP_WEEKS[0]) & ~bs.index.isin(used)]
    idx3 = cand["dunit"].idxmax()          # the F3 "gate-dominated unit" pick
    i3 = base.index.get_loc(idx3)

    rep = base.loc[[idx3]].loc[[idx3] * len(LR_GRID)].copy()
    rep[TREAT] = LR_GRID
    pi, mu, th = model.predict_params(rep)
    E = zanb_mean(pi, mu, th)

    print(f"[2] unit {int(base.loc[idx3].unit)}, week {int(base.loc[idx3].week)}: "
          f"pi range [{pi.min():.3f}, {pi.max():.3f}], "
          f"mu range [{mu.min():.3f}, {mu.max():.3f}], theta {th[0]:.3f}")

    bad = 0
    for g in range(len(LR_GRID)):
        for q in (0.25, 0.5, 0.75, 0.9, 0.95):
            a = float(zanb_ppf(q, pi[[g]], mu[[g]], th[[g]])[0])
            b = brute_quantile(q, float(pi[g]), float(mu[g]), float(th[g]))
            if a != b:
                bad += 1
                print(f"  MISMATCH at f={np.exp(LR_GRID[g]):.3f} q={q}: {a} vs {b}")
    print(f"[2] fan audit: {len(LR_GRID) * 5} cells, mismatches = {bad}")

    # cross-panel consistency at the three CDF prices
    fmult = np.exp(LR_GRID)
    for f in (0.75, 1.0, 1.30):
        g = int(np.argmin(np.abs(fmult - f)))
        ys = np.arange(0, 9, dtype=float)
        F = zanb_cdf(ys, float(pi[g]), np.full_like(ys, mu[g]), np.full_like(ys, th[g]))
        q50 = float(zanb_ppf(0.5, pi[[g]], mu[[g]], th[[g]])[0])
        q90 = float(zanb_ppf(0.9, pi[[g]], mu[[g]], th[[g]])[0])
        med_from_cdf = float(ys[np.searchsorted(F, 0.5, side="left")])
        q90_from_cdf = float(ys[np.searchsorted(F, 0.9, side="left")])
        print(f"  f={fmult[g]:.2f}: pi={pi[g]:.3f} E={E[g]:.3f} | "
              f"F(0)={F[0]:.3f} F(1)={F[1]:.3f} F(2)={F[2]:.3f} F(3)={F[3]:.3f} | "
              f"q50 ppf={q50} cdf={med_from_cdf} | q90 ppf={q90} cdf={q90_from_cdf}")
        assert q50 == med_from_cdf and q90 == q90_from_cdf

    # geometry facts the chart must render
    q25 = np.array([float(zanb_ppf(0.25, pi[[g]], mu[[g]], th[[g]])[0])
                    for g in range(len(LR_GRID))])
    q50 = np.array([float(zanb_ppf(0.50, pi[[g]], mu[[g]], th[[g]])[0])
                    for g in range(len(LR_GRID))])
    print(f"[3] P(Y=0) crosses 0.25 at f={fmult[np.argmax(pi > 0.25)]:.3f} "
          f"-> q25 hits 0 there (q25 path: {sorted(set(q25))})")
    print(f"    median path: {sorted(set(q50))}; mean at far right {E[-1]:.3f} "
          f"(below the integer median 1.0 is legitimate: zero atom "
          f"P(0)={pi[-1]:.3f} + thin right tail)")


if __name__ == "__main__":
    main()
