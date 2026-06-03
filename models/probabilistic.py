"""Probabilistic Modeling.

Fit distribusi Normal dan Poisson ke demand empiris, estimasi parameter,
dan lakukan goodness-of-fit test Kolmogorov-Smirnov.
"""

import numpy as np
from scipy import stats


def run(demand_series):
    """Fit Normal & Poisson lalu uji KS.

    Returns dict: normal/poisson params & p-value, best ('normal'|'poisson'),
    data histogram & PDF/PMF untuk plotting.
    """
    data = np.asarray(demand_series, dtype=float)
    data = data[data >= 0]

    # --- Normal -----------------------------------------------------------
    mu, sigma = stats.norm.fit(data)
    if sigma <= 0:
        sigma = 1e-6
    ks_norm = stats.kstest(data, "norm", args=(mu, sigma))

    # --- Poisson ----------------------------------------------------------
    lam = float(np.mean(data))
    # KS test untuk Poisson menggunakan CDF diskret.
    ks_pois = stats.kstest(data, "poisson", args=(lam,))

    # Distribusi terbaik = p-value tertinggi (paling tidak ditolak).
    if ks_norm.pvalue >= ks_pois.pvalue:
        best = "normal"
    else:
        best = "poisson"

    # --- Data untuk plotting ---------------------------------------------
    counts, bin_edges = np.histogram(data, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    x_cont = np.linspace(max(0, data.min()), data.max(), 200)
    normal_pdf = stats.norm.pdf(x_cont, mu, sigma)

    x_disc = np.arange(0, int(data.max()) + 1)
    poisson_pmf = stats.poisson.pmf(x_disc, lam)

    return {
        "normal": {"mean": float(mu), "std": float(sigma),
                   "ks_stat": float(ks_norm.statistic),
                   "p_value": float(ks_norm.pvalue)},
        "poisson": {"lambda": lam,
                    "ks_stat": float(ks_pois.statistic),
                    "p_value": float(ks_pois.pvalue)},
        "best": best,
        "hist": {"x": bin_centers, "y": counts, "edges": bin_edges},
        "normal_curve": {"x": x_cont, "y": normal_pdf},
        "poisson_curve": {"x": x_disc, "y": poisson_pmf},
        "data": data,
    }


def sample_best(prob_result, size, rng=None):
    """Sample `size` nilai demand dari distribusi terbaik."""
    if rng is None:
        rng = np.random.default_rng()
    if prob_result["best"] == "normal":
        mu = prob_result["normal"]["mean"]
        sigma = prob_result["normal"]["std"]
        samples = rng.normal(mu, sigma, size)
        return np.clip(samples, 0, None)
    else:
        lam = prob_result["poisson"]["lambda"]
        return rng.poisson(lam, size).astype(float)
