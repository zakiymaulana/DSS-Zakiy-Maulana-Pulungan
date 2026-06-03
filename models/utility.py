"""Utility & Risk Preference.

Fungsi utilitas eksponensial U(x) = 1 - exp(-x / R), R = risk tolerance.
Hitung Expected Utility tiap alternatif dan bandingkan dengan keputusan EV.
"""

import numpy as np
import pandas as pd

from . import risk_ev


def utility(x, R):
    """Fungsi utilitas eksponensial (constant absolute risk aversion)."""
    x = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-x / R)


def run(demand_series, unit_price, holding_cost, ordering_cost,
        shortage_cost, risk_tolerance):
    """Jalankan analisis utility.

    Returns dict: table (EV vs EU), best_ev, best_eu, curve, changed, R.
    """
    rk = risk_ev.run(demand_series, unit_price, holding_cost,
                     ordering_cost, shortage_cost)
    alts = rk["alts"]
    M = rk["matrix"]      # [alternatif x state]
    probs = rk["probs"]

    R = float(risk_tolerance)

    ev = M @ probs
    # Expected Utility: terapkan utilitas pada tiap payoff, lalu rata-rata
    # berbobot probabilitas.
    U = utility(M, R)
    eu = U @ probs

    # Certainty equivalent dari EU: CE = -R * ln(1 - EU)
    with np.errstate(divide="ignore"):
        ce = -R * np.log(np.clip(1.0 - eu, 1e-12, None))

    table = pd.DataFrame({
        "Order Quantity": alts.astype(int),
        "Expected Value": np.round(ev, 2),
        "Expected Utility": np.round(eu, 4),
        "Certainty Equivalent": np.round(ce, 2),
    })

    best_ev_idx = int(np.argmax(ev))
    best_eu_idx = int(np.argmax(eu))

    best_ev = {"order": int(alts[best_ev_idx]), "ev": float(ev[best_ev_idx])}
    best_eu = {"order": int(alts[best_eu_idx]), "eu": float(eu[best_eu_idx]),
               "ce": float(ce[best_eu_idx])}

    changed = best_ev_idx != best_eu_idx
    if changed:
        explanation = (
            f"Keputusan **berubah**: berdasarkan EV optimal order = "
            f"{best_ev['order']}, namun dengan utilitas (risk tolerance R={R:.0f}) "
            f"optimal bergeser ke order = {best_eu['order']}. Ini menunjukkan "
            f"preferensi risk-averse: pengambil keputusan rela mengorbankan "
            f"expected payoff demi mengurangi variabilitas/risiko."
        )
    else:
        explanation = (
            f"Keputusan **tidak berubah**: baik EV maupun EU memilih order = "
            f"{best_ev['order']}. Pada level risk tolerance R={R:.0f}, preferensi "
            f"risiko tidak cukup kuat untuk mengubah keputusan optimal."
        )

    # Kurva utilitas untuk visualisasi.
    x_max = float(np.max(M)) if np.max(M) > 0 else 1.0
    x_min = float(np.min(M))
    x_curve = np.linspace(x_min, x_max, 200)
    y_curve = utility(x_curve, R)

    return {
        "table": table,
        "best_ev": best_ev,
        "best_eu": best_eu,
        "changed": changed,
        "explanation": explanation,
        "curve": {"x": x_curve, "y": y_curve},
        "R": R,
    }
