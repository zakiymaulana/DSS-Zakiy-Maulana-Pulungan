"""DSS Inventory — Decision Support System untuk Manajemen Stok.

Mengintegrasikan 6 metode decision theory atas dataset UCI Online Retail II:
Certainty, Risk (EV), Uncertainty, Probabilistic, Utility, dan Monte Carlo.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.loader import load_data
from models import (
    certainty,
    risk_ev,
    uncertainty,
    probabilistic,
    utility,
    simulation,
)

st.set_page_config(page_title="DSS Inventory", layout="wide",
                   page_icon="📦")

st.title("📦 Decision Support System — Manajemen Stok")
st.caption("Dataset: UCI Online Retail II (id=502) · 6 metode decision theory")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    bundle = load_data(top_n=50)
except Exception as exc:  # noqa: BLE001
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()

products = bundle["products"]
descriptions = bundle["descriptions"]
demand_all = bundle["demand"]

if not products:
    st.error("Tidak ada produk dengan data cukup. Coba turunkan min_weeks.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar global
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Parameter Global")


def _fmt_product(code):
    desc = descriptions.get(code, "")
    desc = (desc[:30] + "…") if len(desc) > 30 else desc
    return f"{code} — {desc}" if desc else code


stock_code = st.sidebar.selectbox(
    "Pilih StockCode (50 produk terbanyak)",
    products,
    format_func=_fmt_product,
)

demand_series = demand_all[stock_code]
mean_demand = float(np.mean(demand_series))
default_price = max(round(bundle["avg_price"].get(stock_code, 5.0), 2), 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameter Biaya")

unit_price = st.sidebar.slider(
    "Unit Price (harga jual / unit)",
    min_value=0.5, max_value=max(50.0, default_price * 3),
    value=float(default_price), step=0.5,
)
holding_cost = st.sidebar.slider(
    "Holding Cost (per unit / minggu)",
    min_value=0.0, max_value=10.0, value=0.5, step=0.1,
)
ordering_cost = st.sidebar.slider(
    "Ordering Cost (biaya tetap / order)",
    min_value=0.0, max_value=200.0, value=20.0, step=5.0,
)
shortage_cost = st.sidebar.slider(
    "Shortage Cost (per unit kekurangan)",
    min_value=0.0, max_value=20.0, value=2.0, step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameter Utility")
risk_tolerance = st.sidebar.slider(
    "Risk Tolerance R (kecil = risk-averse)",
    min_value=10.0, max_value=2000.0, value=200.0, step=10.0,
)

st.sidebar.markdown("---")
summ = bundle["raw_summary"]
st.sidebar.caption(
    f"Baris mentah: {summ['rows_raw']:,}\n\n"
    f"Setelah filter valid: {summ['rows_clean']:,}\n\n"
    f"Produk tersedia: {summ['n_products']}"
)

# Info produk terpilih
st.sidebar.markdown("---")
st.sidebar.metric("Mean Demand / minggu", f"{mean_demand:.1f}")
st.sidebar.metric("Jumlah minggu data", f"{len(demand_series)}")

# ---------------------------------------------------------------------------
# Jalankan semua model (sekali, dipakai lintas tab + rekomendasi)
# ---------------------------------------------------------------------------
cost_args = (unit_price, holding_cost, ordering_cost, shortage_cost)

res_cert = certainty.run(demand_series, *cost_args)
res_risk = risk_ev.run(demand_series, *cost_args)
res_unc = uncertainty.run(demand_series, *cost_args)
res_prob = probabilistic.run(demand_series)
res_util = utility.run(demand_series, *cost_args, risk_tolerance)
res_sim = simulation.run(demand_series, res_prob, *cost_args)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📖 Panduan",
    "1️⃣ Certainty",
    "2️⃣ Risk (EV)",
    "3️⃣ Uncertainty",
    "4️⃣ Probabilistic",
    "5️⃣ Utility",
    "6️⃣ Monte Carlo",
    "✅ Rekomendasi",
])

# --- Tab 0: Panduan --------------------------------------------------------
with tabs[0]:
    st.header("📖 Selamat Datang — Apa itu DSS Manajemen Stok ini?")
    st.markdown("""
Aplikasi ini adalah **Decision Support System (DSS)** atau *Sistem Pendukung
Keputusan* untuk membantu pemilik usaha menjawab satu pertanyaan sederhana:

> ### 🛒 *"Minggu ini, sebaiknya saya pesan stok barang sebanyak berapa?"*

Pesan **terlalu sedikit** → barang cepat habis, pelanggan kecewa, kehilangan
penjualan. Pesan **terlalu banyak** → barang menumpuk di gudang, modal
tertahan, ada biaya simpan. DSS ini membantu mencari titik **paling
menguntungkan** menggunakan data penjualan nyata.
""")

    st.info(
        "📊 **Data yang dipakai:** Dataset transaksi ritel online nyata "
        "(*UCI Online Retail II*), berisi ratusan ribu transaksi penjualan. "
        "Aplikasi mengubahnya menjadi pola permintaan mingguan per produk."
    )

    st.markdown("### 🚀 Cara Memakai (3 Langkah)")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            "**1. Pilih Produk**\n\n"
            "Di panel kiri (*sidebar*), pilih kode produk yang ingin dianalisis "
            "dari daftar 50 produk terlaris."
        )
    with g2:
        st.markdown(
            "**2. Atur Biaya**\n\n"
            "Geser slider biaya di panel kiri sesuai kondisi usaha Anda "
            "(harga jual, biaya simpan, biaya pesan, biaya kehabisan)."
        )
    with g3:
        st.markdown(
            "**3. Baca Hasil**\n\n"
            "Buka tab-tab di atas. Tiap tab memakai satu metode. Tab "
            "**✅ Rekomendasi** merangkum semuanya jadi satu kesimpulan."
        )

    st.markdown("### 🎛️ Arti Parameter di Panel Kiri")
    st.markdown("""
| Parameter | Arti dalam bahasa sehari-hari | Contoh |
|---|---|---|
| **Unit Price** | Harga jual satu barang ke pelanggan | Sabun Rp 5.000/buah |
| **Holding Cost** | Biaya menyimpan 1 barang yang *belum laku* selama seminggu (listrik, sewa gudang, risiko rusak) | Rp 500/unit/minggu |
| **Ordering Cost** | Biaya tetap setiap kali memesan ke pemasok (ongkir, admin), berapapun jumlahnya | Rp 20.000/order |
| **Shortage Cost** | Kerugian tiap 1 barang yang diminta pelanggan tapi stok habis (penjualan hilang) | Rp 2.000/unit |
| **Risk Tolerance (R)** | Seberapa berani Anda ambil risiko. Kecil = hati-hati, besar = berani | 50 (penakut) … 2000 (pemberani) |
""")
    st.caption(
        "💡 Semakin tinggi **Holding Cost** → sistem menyarankan pesan lebih "
        "sedikit. Semakin tinggi **Shortage Cost** → sistem menyarankan pesan "
        "lebih banyak."
    )

    st.markdown("### 🧭 Penjelasan 6 Metode (tab di atas)")
    with st.expander("1️⃣ **Certainty** — Anggap permintaan selalu sama"):
        st.markdown(
            "Berasumsi permintaan minggu depan **pasti** sama dengan rata-rata "
            "historis. Cocok untuk usaha yang penjualannya sangat stabil. "
            "*Analogi: warung nasi yang tiap hari pasti habis 50 porsi.*"
        )
    with st.expander("2️⃣ **Risk (EV)** — Permintaan bisa sepi/normal/ramai, dengan peluangnya"):
        st.markdown(
            "Membagi permintaan jadi 3 kondisi (rendah/normal/tinggi) beserta "
            "seberapa sering masing-masing terjadi, lalu menghitung keuntungan "
            "rata-rata tertimbang. *Analogi: penjual payung memperhitungkan "
            "peluang hujan.*"
        )
    with st.expander("3️⃣ **Uncertainty** — Tidak tahu peluangnya sama sekali"):
        st.markdown(
            "Untuk kondisi tanpa data peluang. Memberi 4 sudut pandang: "
            "**Maximax** (optimis/berani), **Maximin** (hati-hati), "
            "**Minimax Regret** (anti-menyesal), **Laplace** (anggap semua "
            "kemungkinan sama). *Analogi: buka toko di kota baru tanpa data.*"
        )
    with st.expander("4️⃣ **Probabilistic** — Mengenali pola sebaran permintaan"):
        st.markdown(
            "Mencari tahu pola permintaan lebih mirip distribusi **Normal** "
            "(lonceng) atau **Poisson** (kejadian acak), lewat uji statistik. "
            "Hasilnya dipakai oleh metode Monte Carlo. *Analogi: menganalisis "
            "pola curah hujan.*"
        )
    with st.expander("5️⃣ **Utility** — Menyesuaikan dengan keberanian Anda"):
        st.markdown(
            "Dua orang dengan untung sama bisa memilih beda karena satu "
            "penakut, satu pemberani. Geser slider **Risk Tolerance** untuk "
            "melihat bagaimana keputusan berubah. *Analogi: pilih deposito "
            "aman vs saham berisiko.*"
        )
    with st.expander("6️⃣ **Monte Carlo** — Simulasi 10.000 kemungkinan minggu"):
        st.markdown(
            "Daripada menebak satu angka, komputer mensimulasikan 10.000 "
            "skenario minggu berbeda lalu merangkum hasilnya secara statistik. "
            "*Analogi: prakiraan cuaca '70% kemungkinan hujan'.*"
        )
    with st.expander("✅ **Rekomendasi** — Kesimpulan akhir dari semua metode"):
        st.markdown(
            "Menggabungkan keputusan optimal dari ke-6 metode menjadi satu "
            "tabel ringkas, lalu menghitung **angka pesanan final** yang "
            "disarankan (nilai tengah/median dari semua metode)."
        )

    st.success(
        "👉 **Mulai sekarang:** pilih produk & atur biaya di panel kiri, lalu "
        "buka tab **✅ Rekomendasi** untuk melihat kesimpulan akhirnya."
    )

    st.divider()
    st.caption(
        "Dibuat untuk mata kuliah **Teori Pengambilan Keputusan**. "
        "DSS mengintegrasikan 6 metode decision theory dalam satu aplikasi."
    )

# --- Tab 1: Certainty ------------------------------------------------------
with tabs[1]:
    st.header("Decision Under Certainty")
    st.markdown(
        "Demand diasumsikan **tetap = mean historis** "
        f"({mean_demand:.1f} unit/minggu). Payoff dihitung untuk tiap "
        "alternatif order quantity (50%–150% mean)."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(res_cert["table"], x="Order Quantity", y="Payoff",
                     title="Payoff per Alternatif Order Quantity",
                     color="Payoff", color_continuous_scale="Blues")
        fig.add_vline(x=res_cert["best"]["order"], line_dash="dash",
                      line_color="red", annotation_text="Optimal")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Order Optimal", res_cert["best"]["order"])
        st.metric("Payoff Optimal", f"{res_cert['best']['payoff']:.2f}")
    st.dataframe(res_cert["table"], hide_index=True, width='stretch')

# --- Tab 2: Risk (EV) ------------------------------------------------------
with tabs[2]:
    st.header("Decision Under Risk — Expected Value")
    st.markdown(
        "Demand dibagi menjadi 3 state (tercile) dengan probabilitas empiris."
    )
    state_df = pd.DataFrame({
        "State": res_risk["labels"],
        "Demand Representatif": np.round(res_risk["states"], 1),
        "Probabilitas": np.round(res_risk["probs"], 3),
    })
    st.subheader("State Demand")
    st.dataframe(state_df, hide_index=True, width='stretch')

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(res_risk["table"], x="Order Quantity",
                      y="Expected Value", markers=True,
                      title="Expected Value per Alternatif")
        fig.add_vline(x=res_risk["best"]["order"], line_dash="dash",
                      line_color="red", annotation_text="Optimal EV")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Order Optimal (EV)", res_risk["best"]["order"])
        st.metric("Expected Value", f"{res_risk['best']['ev']:.2f}")

    st.subheader("Payoff Matrix (alternatif × state)")
    st.dataframe(res_risk["matrix_table"], hide_index=True, width='stretch')

# --- Tab 3: Uncertainty ----------------------------------------------------
with tabs[3]:
    st.header("Decision Under Uncertainty")
    st.markdown(
        "Memakai payoff matrix yang sama, tetapi **mengabaikan probabilitas**. "
        "Empat kriteria klasik diterapkan."
    )
    st.subheader("Rekomendasi tiap Kriteria")
    st.dataframe(res_unc["summary"], hide_index=True, width='stretch')

    st.subheader("Perbandingan Nilai per Alternatif")
    tbl = res_unc["table"]
    fig = go.Figure()
    for col in ["Maximax (best)", "Maximin (worst)", "Laplace (avg)"]:
        fig.add_trace(go.Scatter(x=tbl["Order Quantity"], y=tbl[col],
                                 mode="lines+markers", name=col))
    fig.update_layout(title="Kriteria Uncertainty per Order Quantity",
                      xaxis_title="Order Quantity", yaxis_title="Payoff")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(tbl, x="Order Quantity", y="Max Regret",
                  title="Max Regret per Alternatif (Minimax Regret → minimum)",
                  color="Max Regret", color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(tbl, hide_index=True, width='stretch')

# --- Tab 4: Probabilistic --------------------------------------------------
with tabs[4]:
    st.header("Probabilistic Modeling")
    st.markdown(
        "Fit distribusi **Normal** dan **Poisson**, lalu uji "
        "**Kolmogorov-Smirnov** untuk goodness-of-fit."
    )
    c1, c2, c3 = st.columns(3)
    nrm = res_prob["normal"]
    poi = res_prob["poisson"]
    with c1:
        st.subheader("Normal")
        st.metric("Mean (μ)", f"{nrm['mean']:.2f}")
        st.metric("Std (σ)", f"{nrm['std']:.2f}")
        st.metric("KS p-value", f"{nrm['p_value']:.4f}")
    with c2:
        st.subheader("Poisson")
        st.metric("Lambda (λ)", f"{poi['lambda']:.2f}")
        st.metric("KS p-value", f"{poi['p_value']:.4f}")
    with c3:
        st.subheader("Best Fit")
        st.success(f"**{res_prob['best'].upper()}**\n\n(p-value tertinggi)")
        st.caption("p-value > 0.05 → distribusi tidak ditolak.")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=res_prob["hist"]["x"], y=res_prob["hist"]["y"],
                         name="Histogram (empiris)", opacity=0.6,
                         marker_color="lightblue"))
    fig.add_trace(go.Scatter(x=res_prob["normal_curve"]["x"],
                             y=res_prob["normal_curve"]["y"],
                             mode="lines", name="Normal PDF",
                             line=dict(color="red")))
    fig.add_trace(go.Scatter(x=res_prob["poisson_curve"]["x"],
                             y=res_prob["poisson_curve"]["y"],
                             mode="lines+markers", name="Poisson PMF",
                             line=dict(color="green")))
    fig.update_layout(title="Histogram Demand vs Fitted Distribution",
                      xaxis_title="Demand (unit/minggu)", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Utility --------------------------------------------------------
with tabs[5]:
    st.header("Utility & Risk Preference")
    st.markdown(
        r"Fungsi utilitas eksponensial $U(x) = 1 - e^{-x/R}$ dengan "
        f"risk tolerance **R = {res_util['R']:.0f}**."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(x=res_util["curve"]["x"], y=res_util["curve"]["y"],
                      title="Kurva Utilitas U(x)",
                      labels={"x": "Payoff (x)", "y": "Utility U(x)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Optimal menurut EV", res_util["best_ev"]["order"])
        st.metric("Optimal menurut EU", res_util["best_eu"]["order"])
        st.metric("Certainty Equivalent",
                  f"{res_util['best_eu']['ce']:.2f}")

    if res_util["changed"]:
        st.warning(res_util["explanation"])
    else:
        st.info(res_util["explanation"])

    st.subheader("Perbandingan EV vs EU per Alternatif")
    st.dataframe(res_util["table"], hide_index=True, width='stretch')

# --- Tab 6: Monte Carlo ----------------------------------------------------
with tabs[6]:
    st.header("Monte Carlo Simulation")
    st.markdown(
        f"**{res_sim['n_iter']:,} iterasi**, demand di-sample dari distribusi "
        f"terbaik (**{res_sim['dist_used'].upper()}**)."
    )
    best = res_sim["best"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order Optimal", best["order"])
    c2.metric("Mean Profit", f"{best['mean']:.2f}")
    c3.metric("Std Dev", f"{best['std']:.2f}")
    c4.metric("CI 95%", f"[{best['ci'][0]:.0f}, {best['ci'][1]:.0f}]")

    st.subheader("Distribusi Profit (alternatif optimal)")
    profits = res_sim["profit_dist"][best["order"]]
    fig = px.histogram(x=profits, nbins=60,
                       title=f"Distribusi Profit — Order {best['order']}",
                       labels={"x": "Profit"})
    fig.add_vline(x=best["mean"], line_dash="dash", line_color="red",
                  annotation_text="Mean")
    fig.add_vline(x=best["ci"][0], line_dash="dot", line_color="orange")
    fig.add_vline(x=best["ci"][1], line_dash="dot", line_color="orange")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensitivity: Profit vs Order Quantity")
    sens = res_sim["sensitivity"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sens["Order Quantity"], y=sens["Mean Profit"],
        mode="lines+markers", name="Mean Profit",
        error_y=dict(
            type="data",
            array=sens["CI 95% Upper"] - sens["Mean Profit"],
            arrayminus=sens["Mean Profit"] - sens["CI 95% Lower"],
        ),
    ))
    fig2.update_layout(title="Mean Profit ± CI 95% per Order Quantity",
                       xaxis_title="Order Quantity", yaxis_title="Profit")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(sens, hide_index=True, width='stretch')

# --- Tab 7: Rekomendasi ----------------------------------------------------
with tabs[7]:
    st.header("✅ Rekomendasi Konsolidasi")
    st.markdown(
        f"Ringkasan keputusan optimal dari **6 metode** untuk produk "
        f"**{stock_code}** — {descriptions.get(stock_code, '')}"
    )

    rows = [
        {
            "Metode": "1. Certainty",
            "Order Optimal": res_cert["best"]["order"],
            "Metrik": f"Payoff = {res_cert['best']['payoff']:.2f}",
            "Reasoning": "Demand tetap = mean; payoff maksimum.",
        },
        {
            "Metode": "2. Risk (EV)",
            "Order Optimal": res_risk["best"]["order"],
            "Metrik": f"EV = {res_risk['best']['ev']:.2f}",
            "Reasoning": "Expected value tertinggi atas 3 state probabilistik.",
        },
        {
            "Metode": "3. Uncertainty (Maximin)",
            "Order Optimal": res_unc["summary"].iloc[1]["Order Quantity"],
            "Metrik": f"Worst-case = {res_unc['summary'].iloc[1]['Nilai Kriteria']:.2f}",
            "Reasoning": "Kriteria pesimistik — aman pada skenario terburuk.",
        },
        {
            "Metode": "3. Uncertainty (Minimax Regret)",
            "Order Optimal": res_unc["summary"].iloc[2]["Order Quantity"],
            "Metrik": f"Max regret = {res_unc['summary'].iloc[2]['Nilai Kriteria']:.2f}",
            "Reasoning": "Minimalkan penyesalan maksimum.",
        },
        {
            "Metode": "4. Probabilistic",
            "Order Optimal": "—",
            "Metrik": f"Best fit: {res_prob['best'].upper()} "
                      f"(p={max(res_prob['normal']['p_value'], res_prob['poisson']['p_value']):.3f})",
            "Reasoning": "Menentukan distribusi demand untuk simulasi.",
        },
        {
            "Metode": "5. Utility (EU)",
            "Order Optimal": res_util["best_eu"]["order"],
            "Metrik": f"CE = {res_util['best_eu']['ce']:.2f}",
            "Reasoning": "Memperhitungkan preferensi risiko (risk-averse)."
                         + (" Keputusan berubah dari EV." if res_util["changed"]
                            else " Sama dengan EV."),
        },
        {
            "Metode": "6. Monte Carlo",
            "Order Optimal": res_sim["best"]["order"],
            "Metrik": f"Mean profit = {res_sim['best']['mean']:.2f} "
                      f"(CI [{res_sim['best']['ci'][0]:.0f}, {res_sim['best']['ci'][1]:.0f}])",
            "Reasoning": f"{res_sim['n_iter']:,} simulasi stokastik.",
        },
    ]
    rec_df = pd.DataFrame(rows)
    # Kolom "Order Optimal" berisi campuran int dan "—"; paksa ke string
    # agar PyArrow tidak gagal saat Streamlit konversi ke Arrow table.
    rec_df["Order Optimal"] = rec_df["Order Optimal"].astype(str)
    st.dataframe(rec_df, hide_index=True, width='stretch')

    # Konsensus order quantity dari metode yang menghasilkan angka.
    numeric_orders = [r["Order Optimal"] for r in rows
                      if isinstance(r["Order Optimal"], (int, np.integer))]
    if numeric_orders:
        consensus = int(round(np.median(numeric_orders)))
        st.success(
            f"### 🎯 Rekomendasi Akhir: Order ≈ **{consensus}** unit/minggu\n\n"
            f"Berdasarkan median keputusan optimal lintas metode "
            f"(range {min(numeric_orders)}–{max(numeric_orders)} unit). "
            f"Pertimbangkan profil risiko: gunakan nilai lebih rendah bila "
            f"risk-averse (lihat Maximin/Utility), lebih tinggi bila "
            f"mengejar potensi penjualan (lihat Maximax)."
        )
