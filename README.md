# 📦 DSS Inventory — Sistem Pendukung Keputusan Manajemen Stok

> Tugas Akhir Mata Kuliah **Teori Pengambilan Keputusan**
> Oleh: **Zakiy Maulana Pulungan**

Aplikasi web yang membantu menjawab satu pertanyaan sederhana:

### 🛒 *"Minggu ini, sebaiknya pesan stok barang sebanyak berapa?"*

Pesan terlalu sedikit → barang habis, pelanggan kecewa. Pesan terlalu banyak →
modal tertahan, ada biaya simpan. DSS ini mencari titik **paling menguntungkan**
menggunakan data transaksi ritel nyata (*UCI Online Retail II*).

---

## 🚀 Coba Langsung (Online)

**👉 [Buka Aplikasi](https://LINK-APLIKASI-ANDA.streamlit.app)**

Tidak perlu instalasi apa pun — cukup buka link di atas lewat browser.

---

## 🧠 6 Metode Decision Theory

| # | Metode | Inti |
|---|---|---|
| 1 | **Certainty** | Permintaan dianggap tetap (= rata-rata historis) |
| 2 | **Risk (EV)** | Permintaan sepi/normal/ramai beserta peluangnya |
| 3 | **Uncertainty** | 4 kriteria tanpa peluang: Maximax, Maximin, Minimax Regret, Laplace |
| 4 | **Probabilistic** | Fit distribusi Normal vs Poisson (uji Kolmogorov-Smirnov) |
| 5 | **Utility** | Menyesuaikan keputusan dengan tingkat keberanian terhadap risiko |
| 6 | **Monte Carlo** | Simulasi 10.000 skenario permintaan |
| ✅ | **Rekomendasi** | Kesimpulan akhir yang merangkum semua metode |

> Penjelasan lengkap untuk orang awam tersedia di dalam aplikasi pada
> tab **📖 Panduan** (tab paling kiri).

---

## 💻 Menjalankan Secara Lokal (Opsional)

```bash
# 1. Install pustaka yang dibutuhkan (sekali saja)
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py
