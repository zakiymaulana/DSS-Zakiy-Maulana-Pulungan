"""Decision Under Risk (Expected Value).

Demand historis dibagi menjadi 3 state (rendah/normal/tinggi) berdasarkan
tercile, dengan probabilitas empiris dari frekuensi data. Hitung Expected
Value tiap alternatif order quantity.
"""

import numpy as np
import pandas as pd

from .certainty import order_alternatives, payoff


def build_states(demand_series):
    """Bagi demand jadi 3 state berdasarkan tercile.

    Returns: (states, probs) di mana states adalah nilai demand representatif
    (mean tiap kelompok) dan probs adalah probabilitas empiris.
    """
    data = np.asarray(demand_series, dtype=float)
    q1, q2 = np.quantile(data, [1 / 3, 2 / 3])

    low = data[data <= q1]
    normal = data[(data > q1) & (data <= q2)]
    high = data[data > q2]

    groups = [("Rendah", low), ("Normal", normal), ("Tinggi", high)]
    n = len(data)

    states = []
    probs = []
    labels = []
    for label, grp in groups:
        if len(grp) == 0:
            continue
        labels.append(label)
        states.append(float(np.mean(grp)))
        probs.append(len(grp) / n)

    return labels, np.array(states), np.array(probs)


def payoff_matrix(alts, states, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Matrix payoff [alternatif x state]."""
    M = np.zeros((len(alts), len(states)))
    for i, q in enumerate(alts):
        for j, d in enumerate(states):
            M[i, j] = payoff(q, d, unit_price, holding_cost,
                             ordering_cost, shortage_cost)
    return M


def run(demand_series, unit_price, holding_cost, ordering_cost, shortage_cost):
    """Jalankan analisis risk/EV.

    Returns dict: table, matrix_table, best, states info.
    """
    mean_demand = float(np.mean(demand_series))
    alts = order_alternatives(mean_demand)
    labels, states, probs = build_states(demand_series)

    M = payoff_matrix(alts, states, unit_price, holding_cost,
                      ordering_cost, shortage_cost)
    ev = M @ probs

    # Tabel matrix payoff per state (untuk ditampilkan & dipakai uncertainty).
    matrix_rows = []
    for i, q in enumerate(alts):
        row = {"Order Quantity": int(q)}
        for j, label in enumerate(labels):
            row[f"{label} (d={states[j]:.0f})"] = round(M[i, j], 2)
        row["Expected Value"] = round(ev[i], 2)
        matrix_rows.append(row)
    matrix_table = pd.DataFrame(matrix_rows)

    ev_table = pd.DataFrame({
        "Order Quantity": alts.astype(int),
        "Expected Value": np.round(ev, 2),
    })

    best_idx = int(np.argmax(ev))
    best = {
        "order": int(alts[best_idx]),
        "ev": float(ev[best_idx]),
    }

    return {
        "table": ev_table,
        "matrix_table": matrix_table,
        "best": best,
        "labels": labels,
        "states": states,
        "probs": probs,
        "alts": alts,
        "matrix": M,
    }
