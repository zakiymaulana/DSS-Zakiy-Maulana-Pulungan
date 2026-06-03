"""Decision Under Uncertainty.

Memakai payoff matrix yang sama dengan risk_ev tetapi mengabaikan
probabilitas. Empat kriteria: Maximax, Maximin, Minimax Regret, Laplace.
"""

import numpy as np
import pandas as pd

from . import risk_ev


def run(demand_series, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Jalankan analisis uncertainty.

    Returns dict: table (perbandingan per alternatif), summary (rekomendasi
    tiap kriteria), criteria detail.
    """
    rk = risk_ev.run(demand_series, unit_price, holding_cost,
                     ordering_cost, shortage_cost)
    alts = rk["alts"]
    M = rk["matrix"]  # [alternatif x state]

    maximax = M.max(axis=1)          # best-case tiap alternatif
    maximin = M.min(axis=1)          # worst-case tiap alternatif
    laplace = M.mean(axis=1)         # rata-rata (asumsi equal-likely)

    # Regret = max payoff per kolom (state) - payoff. Minimax regret = pilih
    # alternatif dengan max-regret terkecil.
    col_max = M.max(axis=0)
    regret = col_max[np.newaxis, :] - M
    max_regret = regret.max(axis=1)

    table = pd.DataFrame({
        "Order Quantity": alts.astype(int),
        "Maximax (best)": np.round(maximax, 2),
        "Maximin (worst)": np.round(maximin, 2),
        "Max Regret": np.round(max_regret, 2),
        "Laplace (avg)": np.round(laplace, 2),
    })

    criteria = {
        "Maximax": {
            "idx": int(np.argmax(maximax)),
            "rule": "Optimistik — pilih payoff terbaik dari skenario terbaik.",
        },
        "Maximin": {
            "idx": int(np.argmax(maximin)),
            "rule": "Pesimistik — maksimalkan payoff pada skenario terburuk.",
        },
        "Minimax Regret": {
            "idx": int(np.argmin(max_regret)),
            "rule": "Minimalkan penyesalan maksimum (opportunity loss).",
        },
        "Laplace": {
            "idx": int(np.argmax(laplace)),
            "rule": "Asumsikan semua state equally likely, maksimalkan rata-rata.",
        },
    }

    summary_rows = []
    for name, info in criteria.items():
        i = info["idx"]
        if name == "Maximax":
            val = maximax[i]
        elif name == "Maximin":
            val = maximin[i]
        elif name == "Minimax Regret":
            val = max_regret[i]
        else:
            val = laplace[i]
        summary_rows.append({
            "Kriteria": name,
            "Order Quantity": int(alts[i]),
            "Nilai Kriteria": round(float(val), 2),
            "Penjelasan": info["rule"],
        })
    summary = pd.DataFrame(summary_rows)

    return {
        "table": table,
        "summary": summary,
        "criteria": criteria,
        "alts": alts,
    }
