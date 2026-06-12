"""
ZANB-LSS: Zero-Adjusted (hurdle) Negative Binomial distributional gradient
boosting with per-parameter LightGBM boosters.

Distribution
------------
Y ~ ZANB(pi, mu, theta):
    P(Y = 0)         = pi
    P(Y = k), k >= 1 = (1 - pi) * f_NB2(k; mu, th) / (1 - f_NB2(0; mu, th))

NB2 (mean-dispersion) parameterisation:
    f_NB2(y; mu, th) = Gamma(y+th) / (Gamma(th) y!) * (th/(th+mu))^th * (mu/(th+mu))^y
    p0(mu, th)       = (th/(th+mu))^th
    E[Y]             = (1 - pi) * mu / (1 - p0)
    F(y)             = pi + (1 - pi) * (F_NB(y) - p0) / (1 - p0),   y >= 0

Raw (boosting) scores and links:
    g_pi: pi = sigmoid(g_pi)        a = g_mu: mu = exp(a)        b = g_th: th = exp(b)

The log-likelihood SEPARATES:
    l = [z log pi + (1-z) log(1-pi)] + (1-z) * l_ZTNB(y; mu, th),   z = 1{y=0}
so the gate is an ordinary LightGBM binary booster (trained on all rows), and
(mu, theta) form a 2-parameter LSS problem on the positive rows only, trained
with two coupled boosters updated cyclically (gamboostLSS RS / XGBoostLSS
Step-2 style), each with its OWN feature set and monotone-constraint vector.
That per-parameter structure is what stock LightGBMLSS cannot express: it
trains a single multiclass booster (num_class = n_dist_param), so a single
global `monotone_constraints` vector binds every parameter's trees with the
same sign (see `numclass_shared_constraint_demo`).

All ZTNB derivatives are analytic (scipy.special), validated against finite
differences in `check_gradients`.
"""

import json
import numpy as np
import lightgbm as lgb
from scipy.special import gammaln, digamma, polygamma, expit
from scipy.stats import nbinom

A_CLIP = (-15.0, 15.0)   # raw score clip for a = log mu
B_CLIP = (-8.0, 8.0)     # raw score clip for b = log theta
HESS_FLOOR = 1e-6


# ----------------------------------------------------------------------------
# ZTNB / ZANB math
# ----------------------------------------------------------------------------

def _clip_ab(a, b):
    return np.clip(a, *A_CLIP), np.clip(b, *B_CLIP)


def log_p0(mu, th):
    """log NB2 zero probability, (th/(th+mu))^th = exp(-th*log1p(mu/th)).

    The log1p form avoids the catastrophic cancellation of
    th*(log(th) - log(th+mu)) when mu << th.
    """
    return -th * np.log1p(mu / th)


def one_minus_p0(mu, th):
    return -np.expm1(log_p0(mu, th))


def ztnb_logpmf(y, mu, th):
    """Zero-truncated NB2 log pmf for y >= 1."""
    ll0 = (gammaln(y + th) - gammaln(th) - gammaln(y + 1.0)
           + th * np.log(th) + y * np.log(mu) - (th + y) * np.log(th + mu))
    return ll0 - np.log(one_minus_p0(mu, th))


def ztnb_score_a(y, mu, th):
    """d l / d a and d^2 l / d a^2 (a = log mu) of the ZTNB log-likelihood.

    d l0/d a = th (y - mu) / (th + mu)
    d (-log(1-p0))/d a = -k s,  k = th mu/(th+mu),  s = p0/(1-p0)
    """
    lp0 = log_p0(mu, th)
    p0 = np.exp(lp0)
    omp0 = -np.expm1(lp0)
    s = p0 / omp0
    k = th * mu / (th + mu)
    dl = th * (y - mu) / (th + mu) - k * s
    d2l = (-mu * th * (th + y) / (th + mu) ** 2
           - mu * th ** 2 * s / (th + mu) ** 2
           + k ** 2 * p0 / omp0 ** 2)
    return dl, d2l


def ztnb_score_b(y, mu, th):
    """d l / d b and d^2 l / d b^2 (b = log theta) of the ZTNB log-likelihood.

    With L' = d l/d th, L'' = d^2 l/d th^2:
        d l/d b   = th L'
        d^2l/d b^2 = th L' + th^2 L''
    L'  = psi(y+th) - psi(th) + log(th/(th+mu)) + (mu - y)/(th+mu) + s w
    L'' = psi1(y+th) - psi1(th) + mu/(th(th+mu)) - (mu - y)/(th+mu)^2
          + p0 w^2/(1-p0)^2 + s mu^2/(th (th+mu)^2)
    w   = log(th/(th+mu)) + mu/(th+mu)   (< 0)
    """
    lp0 = log_p0(mu, th)
    p0 = np.exp(lp0)
    omp0 = -np.expm1(lp0)
    s = p0 / omp0
    log_ratio = np.log(th) - np.log(th + mu)
    w = log_ratio + mu / (th + mu)
    Lp = (digamma(y + th) - digamma(th) + log_ratio
          + (mu - y) / (th + mu) + s * w)
    Lpp = (polygamma(1, y + th) - polygamma(1, th)
           + mu / (th * (th + mu)) - (mu - y) / (th + mu) ** 2
           + p0 * w ** 2 / omp0 ** 2 + s * mu ** 2 / (th * (th + mu) ** 2))
    dl = th * Lp
    d2l = th * Lp + th ** 2 * Lpp
    return dl, d2l


def ztnb_grad_hess_a(y, a, b):
    """NLL gradient/hessian wrt a, hessian floored positive."""
    a, b = _clip_ab(a, b)
    dl, d2l = ztnb_score_a(y, np.exp(a), np.exp(b))
    return -dl, np.maximum(-d2l, HESS_FLOOR)


def ztnb_grad_hess_b(y, a, b):
    a, b = _clip_ab(a, b)
    dl, d2l = ztnb_score_b(y, np.exp(a), np.exp(b))
    return -dl, np.maximum(-d2l, HESS_FLOOR)


def ztnb_mean(mu, th):
    """E[Y | Y >= 1] = mu / (1 - p0)."""
    return mu / one_minus_p0(mu, th)


def zanb_mean(pi, mu, th):
    return (1.0 - pi) * ztnb_mean(mu, th)


def zanb_nll(y, pi, mu, th):
    """Per-row ZANB negative log-likelihood."""
    z = (y == 0)
    yy = np.maximum(y, 1.0)
    nll_pos = -np.log1p(-pi) - ztnb_logpmf(yy, mu, th)
    return np.where(z, -np.log(pi), nll_pos)


def zanb_cdf(y, pi, mu, th):
    """ZANB cdf at integer y (vectorised); F(y<0)=0, F(0)=pi."""
    p = th / (th + mu)
    lp0 = log_p0(mu, th)
    p0 = np.exp(lp0)
    omp0 = -np.expm1(lp0)
    base = (nbinom.cdf(y, th, p) - p0) / omp0
    out = pi + (1.0 - pi) * np.clip(base, 0.0, 1.0)
    out = np.where(y < 0, 0.0, np.where(y == 0, pi, out))
    return out


def zanb_ppf(q, pi, mu, th):
    """ZANB quantile function (q scalar, parameters vectors)."""
    p = th / (th + mu)
    lp0 = log_p0(mu, th)
    p0 = np.exp(lp0)
    omp0 = -np.expm1(lp0)
    inner = p0 + np.clip(q - pi, 0.0, None) / (1.0 - pi) * omp0
    inner = np.minimum(inner, 1.0 - 1e-12)
    res = nbinom.ppf(inner, th, p)
    return np.where(q <= pi, 0.0, res)


def zanb_pit(rng, y, pi, mu, th):
    """Randomised PIT (randomised quantile residuals on [0,1]) for discrete Y."""
    Fy = zanb_cdf(y, pi, mu, th)
    Fy1 = zanb_cdf(y - 1, pi, mu, th)
    return Fy1 + rng.uniform(size=np.shape(y)) * (Fy - Fy1)


def check_quantiles(seed=0, n_params=1500,
                    qs=(0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    """Property test: closed-form zanb_ppf == brute-force inversion of the
    exact pmf (zero atom + cumulated ztnb_logpmf; independent of
    scipy.nbinom.cdf/ppf). Asserts exact agreement on every pair."""
    rng = np.random.default_rng(seed)
    mismatches, checked = 0, 0
    for _ in range(n_params):
        pi = float(rng.uniform(0.01, 0.9))
        mu = float(np.exp(rng.uniform(-2.5, 4.0)))
        th = float(np.exp(rng.uniform(-2.0, 2.5)))
        kmax, qmax = 4000, max(qs)
        while True:                       # support long enough for the top q
            ygrid = np.arange(1, kmax + 1, dtype=float)
            cdf = pi + np.cumsum((1.0 - pi) * np.exp(ztnb_logpmf(ygrid, mu, th)))
            if cdf[-1] >= qmax + 1e-12:
                break
            kmax *= 8
        for q in qs:
            a = float(zanb_ppf(q, np.array([pi]), np.array([mu]),
                               np.array([th]))[0])
            b = 0.0 if q <= pi else float(
                ygrid[np.searchsorted(cdf, q, side="left")])
            checked += 1
            mismatches += int(a != b)
    out = {"pairs_checked": checked, "mismatches": int(mismatches)}
    assert mismatches == 0, "zanb_ppf disagrees with brute-force pmf inversion"
    return out


def ztnb_sample(rng, mu, th, max_tries=200):
    """Exact ZTNB sampling: NB2 = Poisson(Gamma(th, mu/th)) with zeros rejected."""
    mu = np.asarray(mu, dtype=float)
    th = np.asarray(th, dtype=float)
    out = np.zeros(mu.shape, dtype=np.int64)
    todo = np.ones(mu.shape, dtype=bool)
    for _ in range(max_tries):
        n = int(todo.sum())
        if n == 0:
            break
        lam = rng.gamma(shape=th[todo], scale=mu[todo] / th[todo])
        draw = rng.poisson(lam)
        idx = np.flatnonzero(todo)
        ok = draw > 0
        out[idx[ok]] = draw[ok]
        todo[idx[ok]] = False
    out[todo] = 1  # pathological p0 ~ 1 rows
    return out


def ztnb_profile_theta(y, a, b0=0.0, max_steps=40, tol=1e-9):
    """Profile MLE of the GLOBAL log-theta given per-row location scores a.

    1-D damped Newton on b with the analytic ZTNB score/information. This is
    the Z0 dispersion: a single scalar consistent with the CONDITIONAL fit.
    The unconditional pooled (mu, theta) MLE must not be used as a start
    value: pooling rows whose mu spans orders of magnitude makes the marginal
    look infinitely overdispersed and its MLE degenerates to the
    (mu -> 0, theta -> 0) ridge corner, where all NB2 gradients (which scale
    with theta) die and boosting stalls.
    """
    y = np.asarray(y, dtype=float)
    mu = np.exp(np.clip(a, *A_CLIP))
    b = float(np.clip(b0, *B_CLIP))
    for _ in range(max_steps):
        dl, d2l = ztnb_score_b(y, mu, np.exp(b))
        g = -float(np.sum(dl))          # d NLL / d b
        h = -float(np.sum(d2l))         # d^2 NLL / d b^2 (observed info)
        step = -g / h if h > 0 else -0.2 * np.sign(g)
        step = float(np.clip(step, -0.5, 0.5))
        b = float(np.clip(b + step, *B_CLIP))
        if abs(step) < tol:
            break
    return b


# ----------------------------------------------------------------------------
# Verification utilities
# ----------------------------------------------------------------------------

def check_gradients(seed=0, n=4000, eps=1e-5):
    """Validate analytic ZTNB derivatives against central finite differences.

    Gradients are checked against FD of the log-pmf. Hessians are checked
    against FD of the *analytic gradient* (second-differencing the raw NLL is
    numerically hopeless on heavy-tailed rows where individual gammaln terms
    are O(1e4) and cancel to O(10)). Also checks E[score] ~ 0 at the true
    parameters (score identity of the truncated likelihood).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 5.5, n)
    b = rng.uniform(-2.0, 3.0, n)
    mu, th = np.exp(a), np.exp(b)
    y = ztnb_sample(rng, mu, th).astype(float)

    def nll(aa, bb):
        return -ztnb_logpmf(y, np.exp(aa), np.exp(bb))

    rel = lambda x, t: np.max(np.abs(x - t) / (1.0 + np.abs(t)))

    ga_fd = (nll(a + eps, b) - nll(a - eps, b)) / (2 * eps)
    gb_fd = (nll(a, b + eps) - nll(a, b - eps)) / (2 * eps)

    grad_a = lambda aa, bb: -ztnb_score_a(y, np.exp(aa), np.exp(bb))[0]
    grad_b = lambda aa, bb: -ztnb_score_b(y, np.exp(aa), np.exp(bb))[0]
    ha_fd = (grad_a(a + eps, b) - grad_a(a - eps, b)) / (2 * eps)
    hb_fd = (grad_b(a, b + eps) - grad_b(a, b - eps)) / (2 * eps)

    dla, d2la = ztnb_score_a(y, mu, th)
    dlb, d2lb = ztnb_score_b(y, mu, th)

    # score identity at the truth: |mean| must sit within Monte-Carlo noise
    se_a = float(np.std(dla) / np.sqrt(n))
    se_b = float(np.std(dlb) / np.sqrt(n))
    out = {
        "grad_a_max_rel_err": float(rel(-dla, ga_fd)),
        "hess_a_max_rel_err": float(rel(-d2la, ha_fd)),
        "grad_b_max_rel_err": float(rel(-dlb, gb_fd)),
        "hess_b_max_rel_err": float(rel(-d2lb, hb_fd)),
        "mean_score_a_at_truth": float(np.mean(dla)),
        "mean_score_b_at_truth": float(np.mean(dlb)),
        "se_score_a": se_a,
        "se_score_b": se_b,
        "score_identity_pass": bool(abs(np.mean(dla)) <= 4 * se_a
                                    and abs(np.mean(dlb)) <= 4 * se_b),
    }
    assert out["grad_a_max_rel_err"] <= 1e-6, "grad_a mismatch vs FD"
    assert out["hess_a_max_rel_err"] <= 1e-6, "hess_a mismatch vs FD"
    assert out["grad_b_max_rel_err"] <= 1e-6, "grad_b mismatch vs FD"
    assert out["hess_b_max_rel_err"] <= 1e-6, "hess_b mismatch vs FD"
    assert out["score_identity_pass"], "E[score] != 0 at true parameters"
    return out


def numclass_shared_constraint_demo(seed=0, n=6000, rounds=150):
    """Empirical proof that LightGBM's `monotone_constraints` binds EVERY
    output of a multiclass (num_class = K) booster with the same sign.

    Stock LightGBMLSS trains exactly such a booster (model.py sets
    num_class = n_dist_param), so per-parameter constraint signs -- gate +1,
    mu -1, theta excluded -- are inexpressible there.

    We fit a 2-output booster with an L2 custom objective whose targets have
    OPPOSITE slopes in x (+3x and -3x) under a single constraint vector [-1].
    Both outputs come out non-increasing; output 0 (which needs +slope)
    is forced flat and its loss stays high.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (n, 1))
    t0, t1 = 3.0 * x[:, 0], -3.0 * x[:, 0]
    targets = np.column_stack([t0, t1])  # lgb 4.x passes multiclass preds as (n, K)

    ds = lgb.Dataset(x, label=np.zeros(n), free_raw_data=False)
    params = dict(objective="none", num_class=2, num_leaves=31,
                  learning_rate=0.2, min_data_in_leaf=20, verbosity=-1,
                  monotone_constraints=[-1], monotone_constraints_method="advanced",
                  seed=7, deterministic=True)
    bst = lgb.Booster(params=params, train_set=ds)

    def fobj(preds, data):
        preds = preds.reshape(targets.shape) if preds.ndim == 1 else preds
        return preds - targets, np.ones_like(targets)

    for _ in range(rounds):
        bst.update(fobj=fobj)

    pred = bst.predict(x, raw_score=True)  # (n, 2)
    order = np.argsort(x[:, 0])
    inc0 = float(np.max(np.diff(pred[order, 0]), initial=-np.inf))
    inc1 = float(np.max(np.diff(pred[order, 1]), initial=-np.inf))
    rmse0 = float(np.sqrt(np.mean((pred[:, 0] - t0) ** 2)))
    rmse1 = float(np.sqrt(np.mean((pred[:, 1] - t1) ** 2)))
    return {
        "max_increase_output0": inc0,   # <= 0: forced non-increasing
        "max_increase_output1": inc1,   # <= 0: non-increasing (matches target)
        "rmse_output0_target_plus3x": rmse0,   # large: +slope inexpressible
        "rmse_output1_target_minus3x": rmse1,  # small
    }


# ----------------------------------------------------------------------------
# Boosters
# ----------------------------------------------------------------------------

class ZTNBCyclicBooster:
    """Two coupled LightGBM boosters for the zero-truncated NB2 part.

    B_mu  predicts a = log mu     (features `feat_mu`,  params `params_mu`)
    B_th  predicts b = log theta  (features `feat_th`,  params `params_th`)

    Two-phase fit (the D2 variant ladder, Z0 -> Z1):

    Phase 1 (Z0)  location boosting with theta held as a GLOBAL scalar,
        re-profiled by 1-D Newton after every tree (`ztnb_profile_theta`).
        Theta stays consistent with the evolving location fit, so the NB2
        gradients -- which scale with theta -- keep a healthy magnitude.
        Early stop on val ZTNB NLL; roll the booster back to the best
        iteration and freeze c_b at the profiled scalar of that iteration.
    Phase 2 (Z1)  cyclic per-parameter updates: one Newton tree for B_mu at
        current theta scores, one for B_th at refreshed mu scores
        (gamboostLSS RS / XGBoostLSS step-2), starting from the calibrated
        c_b. Early stop on the same metric. The theta booster is KEPT only
        if phase 2 beats the Z0 val NLL by `z1_min_gain` (promotion gate);
        otherwise the model ships as Z0 (best_iter_th = 0, theta constant).

    Never initialise from the pooled unconditional ZTNB MLE: with unit means
    spanning orders of magnitude the pooled marginal drives (mu, theta) to a
    degenerate ridge corner (theta ~ 0), where location gradients vanish,
    early stopping fires immediately, and the theta booster absorbs the
    location misfit with the opposite sign (anti-correlated dispersion).
    """

    def __init__(self, feat_mu, feat_th, params_mu, params_th,
                 num_rounds=400, early_stopping=60, theta_updates_per_iter=1,
                 z1_min_gain=1e-4):
        self.feat_mu, self.feat_th = list(feat_mu), list(feat_th)
        self.params_mu = dict(params_mu)
        self.params_th = dict(params_th)
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.theta_updates_per_iter = int(theta_updates_per_iter)
        self.z1_min_gain = float(z1_min_gain)
        self.best_iter = None       # mu-booster trees kept
        self.best_iter_th = None    # theta-booster trees kept (0 = Z0 scalar)
        self.c_a = self.c_b = None
        self.bst_mu = self.bst_th = None
        self.history = []           # val ZTNB NLL per outer iteration
        self.phase1_end = None      # index into history where phase 2 starts
        self.z0_val_nll = None
        self.z1_val_nll = None
        self.z1_accepted = None

    def _val_nll(self, yva, Va, Vb):
        a_v, b_v = _clip_ab(Va + self.c_a, Vb + self.c_b)
        return float(-np.mean(ztnb_logpmf(yva, np.exp(a_v), np.exp(b_v))))

    def fit(self, Xtr, ytr, Xva, yva):
        ytr = np.asarray(ytr, dtype=float)
        yva = np.asarray(yva, dtype=float)
        Xtr_mu = np.ascontiguousarray(Xtr[self.feat_mu].to_numpy(dtype=np.float64))
        Xtr_th = np.ascontiguousarray(Xtr[self.feat_th].to_numpy(dtype=np.float64))
        Xva_mu = np.ascontiguousarray(Xva[self.feat_mu].to_numpy(dtype=np.float64))
        Xva_th = np.ascontiguousarray(Xva[self.feat_th].to_numpy(dtype=np.float64))

        ds_mu = lgb.Dataset(Xtr_mu, label=ytr, free_raw_data=False)
        ds_th = lgb.Dataset(Xtr_th, label=ytr, free_raw_data=False)
        pm = {**self.params_mu, "objective": "none", "verbosity": -1}
        pt = {**self.params_th, "objective": "none", "verbosity": -1}
        self.bst_mu = lgb.Booster(params=pm, train_set=ds_mu)
        self.bst_th = lgb.Booster(params=pt, train_set=ds_th)

        # ---- phase 1 (Z0): location boosting + profiled scalar dispersion --
        self.c_a = float(np.log(max(ytr.mean(), 1.0)))   # scale-correct centre
        b_prof = ztnb_profile_theta(ytr, np.full_like(ytr, self.c_a), b0=0.0)
        b_path = [b_prof]

        Fa = np.zeros(len(ytr)); Va = np.zeros(len(yva))
        best_nll, best_it, since = np.inf, 0, 0
        n_mu = 0
        for it in range(1, self.num_rounds + 1):
            theta_tr = np.full_like(ytr, b_prof)

            def fobj_mu(preds, data):
                return ztnb_grad_hess_a(ytr, preds + self.c_a, theta_tr)
            self.bst_mu.update(fobj=fobj_mu)
            Fa += self.bst_mu.predict(Xtr_mu, raw_score=True,
                                      start_iteration=n_mu, num_iteration=1)
            Va += self.bst_mu.predict(Xva_mu, raw_score=True,
                                      start_iteration=n_mu, num_iteration=1)
            n_mu += 1

            b_prof = ztnb_profile_theta(ytr, Fa + self.c_a, b0=b_prof, max_steps=5)
            b_path.append(b_prof)

            self.c_b = b_prof
            val_nll = self._val_nll(yva, Va, np.zeros(len(yva)))
            self.history.append(val_nll)
            if val_nll < best_nll - 1e-7:
                best_nll, best_it, since = val_nll, it, 0
            else:
                since += 1
                if since >= self.early_stopping:
                    break

        # roll back to the best phase-1 state, freeze the scalar of that state
        while n_mu > best_it:
            self.bst_mu.rollback_one_iter()
            n_mu -= 1
        self.c_b = b_path[best_it]
        Fa = self.bst_mu.predict(Xtr_mu, raw_score=True)
        Va = self.bst_mu.predict(Xva_mu, raw_score=True)
        self.z0_val_nll = self._val_nll(yva, Va, np.zeros(len(yva)))
        self.phase1_end = len(self.history)

        # ---- phase 2 (Z1): cyclic per-parameter updates from the Z0 state --
        Fb = np.zeros(len(ytr)); Vb = np.zeros(len(yva))
        best_nll2 = self.z0_val_nll
        best_state = (n_mu, 0)
        n_th, since = 0, 0
        for it in range(1, self.num_rounds + 1):
            def fobj_mu2(preds, data):
                return ztnb_grad_hess_a(ytr, preds + self.c_a, Fb + self.c_b)
            self.bst_mu.update(fobj=fobj_mu2)
            Fa += self.bst_mu.predict(Xtr_mu, raw_score=True,
                                      start_iteration=n_mu, num_iteration=1)
            Va += self.bst_mu.predict(Xva_mu, raw_score=True,
                                      start_iteration=n_mu, num_iteration=1)
            n_mu += 1

            for _ in range(self.theta_updates_per_iter):
                def fobj_th(preds, data):
                    return ztnb_grad_hess_b(ytr, Fa + self.c_a, preds + self.c_b)
                self.bst_th.update(fobj=fobj_th)
                Fb += self.bst_th.predict(Xtr_th, raw_score=True,
                                          start_iteration=n_th, num_iteration=1)
                Vb += self.bst_th.predict(Xva_th, raw_score=True,
                                          start_iteration=n_th, num_iteration=1)
                n_th += 1

            val_nll = self._val_nll(yva, Va, Vb)
            self.history.append(val_nll)
            if val_nll < best_nll2 - 1e-7:
                best_nll2, best_state, since = val_nll, (n_mu, n_th), 0
            else:
                since += 1
                if since >= self.early_stopping:
                    break

        self.z1_val_nll = best_nll2
        # promotion gate Z0 -> Z1: keep the theta booster only on material gain
        self.z1_accepted = bool(self.z0_val_nll - best_nll2 >= self.z1_min_gain)
        if self.z1_accepted:
            self.best_iter, self.best_iter_th = best_state
        else:                       # ship Z0: scalar theta, phase-1 location
            self.best_iter, self.best_iter_th = best_it, 0

        # consistency: maintained scores == full booster predict
        full_a = self.bst_mu.predict(Xtr_mu, raw_score=True)
        assert np.allclose(full_a, Fa, atol=1e-6), "incremental score drift (mu)"
        if n_th > 0:
            full_b = self.bst_th.predict(Xtr_th, raw_score=True)
            assert np.allclose(full_b, Fb, atol=1e-6), "incremental score drift (theta)"
        return self

    def predict_ab(self, X):
        a = self.c_a + self.bst_mu.predict(
            np.ascontiguousarray(X[self.feat_mu].to_numpy(dtype=np.float64)),
            raw_score=True, num_iteration=self.best_iter)
        if self.best_iter_th and self.best_iter_th > 0:
            b = self.c_b + self.bst_th.predict(
                np.ascontiguousarray(X[self.feat_th].to_numpy(dtype=np.float64)),
                raw_score=True, num_iteration=self.best_iter_th)
        else:
            b = np.full(len(X), self.c_b)
        return _clip_ab(a, b)


class GateBooster:
    """P(Y = 0) gate: ordinary LightGBM binary booster (exact hurdle MLE)."""

    def __init__(self, feats, params, num_rounds=400, early_stopping=60):
        self.feats = list(feats)
        self.params = dict(params)
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.bst = None

    def fit(self, Xtr, ztr, Xva, zva):
        ds = lgb.Dataset(Xtr[self.feats].to_numpy(dtype=np.float64),
                         label=ztr.astype(float), free_raw_data=False)
        dv = lgb.Dataset(Xva[self.feats].to_numpy(dtype=np.float64),
                         label=zva.astype(float), reference=ds, free_raw_data=False)
        params = {**self.params, "objective": "binary",
                  "metric": "binary_logloss", "verbosity": -1}
        self.bst = lgb.train(params, ds, num_boost_round=self.num_rounds,
                             valid_sets=[dv],
                             callbacks=[lgb.early_stopping(self.early_stopping, verbose=False),
                                        lgb.log_evaluation(0)])
        return self

    def predict_pi(self, X):
        pi = self.bst.predict(X[self.feats].to_numpy(dtype=np.float64))
        return np.clip(pi, 1e-7, 1.0 - 1e-7)


class ZANBModel:
    """Gate + cyclic ZTNB pair = full ZANB(pi, mu, theta) distributional model."""

    def __init__(self, gate: GateBooster, ztnb: ZTNBCyclicBooster):
        self.gate = gate
        self.ztnb = ztnb

    def fit(self, Xtr, ytr, Xva, yva):
        ytr = np.asarray(ytr, dtype=float)
        yva = np.asarray(yva, dtype=float)
        self.gate.fit(Xtr, (ytr == 0), Xva, (yva == 0))
        pos_tr, pos_va = ytr > 0, yva > 0
        self.ztnb.fit(Xtr.loc[pos_tr], ytr[pos_tr], Xva.loc[pos_va], yva[pos_va])
        return self

    def predict_params(self, X):
        pi = self.gate.predict_pi(X)
        a, b = self.ztnb.predict_ab(X)
        return pi, np.exp(a), np.exp(b)

    def predict_mean(self, X):
        pi, mu, th = self.predict_params(X)
        return zanb_mean(pi, mu, th)

    def nll(self, X, y):
        pi, mu, th = self.predict_params(X)
        return float(np.mean(zanb_nll(np.asarray(y, dtype=float), pi, mu, th)))


if __name__ == "__main__":
    print(json.dumps(check_gradients(), indent=2))
    print(json.dumps(numclass_shared_constraint_demo(), indent=2))
