============================================================
  DSS INVENTORY - Sistem Pendukung Keputusan Manajemen Stok
  Tugas Akhir Mata Kuliah Teori Pengambilan Keputusan
============================================================

APA INI?
--------
Aplikasi web (Streamlit) yang membantu menjawab pertanyaan:
"Minggu ini, sebaiknya pesan stok barang sebanyak berapa?"

Aplikasi memakai data transaksi ritel nyata (UCI Online Retail II)
dan mengintegrasikan 6 metode Teori Pengambilan Keputusan:
  1. Certainty       - permintaan dianggap tetap (= rata-rata)
  2. Risk (EV)       - permintaan sepi/normal/ramai + peluangnya
  3. Uncertainty     - 4 kriteria tanpa peluang (Maximax, Maximin,
                       Minimax Regret, Laplace)
  4. Probabilistic   - fit distribusi Normal vs Poisson (uji KS)
  5. Utility         - menyesuaikan keberanian terhadap risiko
  6. Monte Carlo     - simulasi 10.000 skenario
  + Tab Rekomendasi  - kesimpulan akhir dari semua metode

Penjelasan lengkap tiap metode & parameter tersedia di dalam
aplikasi, pada tab "Panduan" (tab paling kiri).


CARA MENJALANKAN (Windows)
--------------------------
1. Pastikan Python 3.10+ terpasang. Cek dengan: python --version
2. Pastikan file dataset "online_retail_II.xlsx" ada di folder data/
3. Install pustaka yang dibutuhkan (sekali saja):
       pip install -r requirements.txt
4. Jalankan aplikasi:
       streamlit run app.py
   ATAU cukup klik dua kali file: run.bat
5. Browser akan terbuka otomatis di http://localhost:8501


STRUKTUR FOLDER
---------------
Zakiy Maulana Pulungan - DSS/
  app.py             -> aplikasi utama (tampilan & 7 tab)
  run.bat            -> jalan pintas menjalankan aplikasi
  requirements.txt   -> daftar pustaka Python (PENTING, jangan dihapus)
  README.txt         -> berkas ini
  data/
    online_retail_II.xlsx  -> dataset (letakkan di sini)
    loader.py        -> membaca & mengolah data jadi permintaan mingguan
  models/
    certainty.py     -> metode 1
    risk_ev.py       -> metode 2
    uncertainty.py   -> metode 3
    probabilistic.py -> metode 4
    utility.py       -> metode 5
    simulation.py    -> metode 6 (Monte Carlo)


CATATAN
-------
- Pemuatan pertama agak lama (~1-2 menit) karena membaca file Excel
  besar. Setelah itu dibuat cache otomatis (online_retail_II.parquet)
  sehingga membuka berikutnya jauh lebih cepat.
- Semua biaya pada panel kiri dapat diubah untuk simulasi
  skenario usaha yang berbeda.
