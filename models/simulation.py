"""Monte Carlo Simulation.

Jalankan 10.000 iterasi; tiap iterasi sample demand dari distribusi terbaik
(hasil probabilistic), hitung profit untuk tiap alternatif order quantity.
"""

import numpy as np
import pandas as pd

from .certainty import order_alternatives
from . import probabilistic


def _profit_vec(order, demand, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Profit ter-vektorisasi untuk satu order quantity atas array demand."""
    sold = np.minimum(order, demand)
    revenue = sold * unit_price
    over = np.maximum(order - demand, 0) * holding_cost
    short = np.maximum(demand - order, 0) * shortage_cost
    return revenue - ordering_cost - over - short


def run(demand_series, prob_result, unit_price, holding_cost,
        ordering_cost, shortage_cost, n_iter=10000, seed=42):
    """Jalankan simulasi Monte Carlo.

    Returns dict: sensitivity (DataFrame profit vs order), best, dan
    distribusi profit untuk alternatif terbaik.
    """
    rng = np.random.default_rng(seed)
    mean_demand = float(np.mean(demand_series))
    alts = order_alternatives(mean_demand)

    # Sample demand sekali, dipakai untuk semua alternatif (common random
    # numbers) agar perbandingan adil.
    demand_samples = probabilistic.sample_best(prob_result, n_iter, rng=rng)

    rows = []
    profit_dist = {}     # order -> array profit
    for q in alts:
        profits = _profit_vec(q, demand_samples, unit_price, holding_cost,
                              ordering_cost, shortage_cost)
        profit_dist[int(q)] = profits
        lo, hi = np.percentile(profits, [2.5, 97.5])
        rows.append({
            "Order Quantity": int(q),
            "Mean Profit": round(float(np.mean(profits)), 2),
            "Std Dev": round(float(np.std(profits)), 2),
            "CI 95% Lower": round(float(lo), 2),
            "CI 95% Upper": round(float(hi), 2),
        })

    sensitivity = pd.DataFrame(rows)
    best_idx = int(sensitivity["Mean Profit"].idxmax())
    best_order = int(sensitivity.loc[best_idx, "Order Quantity"])
    best = {
        "order": best_order,
        "mean": float(sensitivity.loc[best_idx, "Mean Profit"]),
        "std": float(sensitivity.loc[best_idx, "Std Dev"]),
        "ci": (float(sensitivity.loc[best_idx, "CI 95% Lower"]),
               float(sensitivity.loc[best_idx, "CI 95% Upper"])),
    }

    return {
        "sensitivity": sensitivity,
        "best": best,
        "profit_dist": profit_dist,
        "n_iter": n_iter,
        "dist_used": prob_result["best"],
        "demand_samples": demand_samples,
    }
