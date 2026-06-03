"""Decision Under Certainty.

Demand diasumsikan tetap = mean historis. Bangun array alternatif order
quantity dari 50%..150% mean (step 10%) dan hitung payoff tiap alternatif.
"""

import numpy as np
import pandas as pd


def order_alternatives(mean_demand):
    """Array alternatif order quantity 50%..150% mean, step 10%."""
    factors = np.arange(0.5, 1.5 + 1e-9, 0.1)
    alts = np.round(factors * mean_demand).astype(int)
    # Hindari nol/duplikat agar tabel informatif.
    alts = np.clip(alts, 1, None)
    return alts


def payoff(order, demand, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Payoff sebuah keputusan order pada realisasi demand tertentu."""
    revenue = min(order, demand) * unit_price
    over = max(order - demand, 0) * holding_cost
    short = max(demand - order, 0) * shortage_cost
    return revenue - ordering_cost - over - short


def run(demand_series, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Jalankan analisis certainty.

    Returns dict: table (DataFrame), best (dict), mean_demand (float).
    """
    mean_demand = float(np.mean(demand_series))
    alts = order_alternatives(mean_demand)

    rows = []
    for q in alts:
        p = payoff(q, mean_demand, unit_price, holding_cost,
                   ordering_cost, shortage_cost)
        rows.append({
            "Order Quantity": int(q),
            "% of Mean": f"{q / mean_demand * 100:.0f}%",
            "Payoff": round(p, 2),
        })

    table = pd.DataFrame(rows)
    best_idx = table["Payoff"].idxmax()
    best = {
        "order": int(table.loc[best_idx, "Order Quantity"]),
        "payoff": float(table.loc[best_idx, "Payoff"]),
    }

    return {
        "table": table,
        "best": best,
        "mean_demand": mean_demand,
    }
