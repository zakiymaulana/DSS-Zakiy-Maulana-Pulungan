"""Data pipeline untuk DSS Inventory.

Membaca dataset UCI Online Retail II dari file lokal di folder data/.

Dua format didukung:
  - online_retail_II.parquet  (ringan ~7MB, cepat ~2 detik) -> diutamakan
  - online_retail_II.xlsx     (besar ~44MB, lambat ~100 detik) -> fallback

Untuk deploy ke cloud, cukup sertakan file .parquet agar aplikasi ringan
dan tidak timeout. Jika hanya ada .xlsx, aplikasi akan otomatis membuat
.parquet dari xlsx pada pemuatan pertama.
"""

import os

import pandas as pd
import streamlit as st


# Path dataset lokal.
_DATA_DIR     = os.path.dirname(__file__)
_XLSX_PATH    = os.path.join(_DATA_DIR, "online_retail_II.xlsx")
_PARQUET_PATH = os.path.join(_DATA_DIR, "online_retail_II.parquet")

# Nama kolom yang mungkin berbeda antar versi dataset.
_QUANTITY_CANDIDATES = ["Quantity"]
_PRICE_CANDIDATES    = ["Price", "UnitPrice"]
_STOCK_CANDIDATES    = ["StockCode"]
_DATE_CANDIDATES     = ["InvoiceDate"]
_DESC_CANDIDATES     = ["Description"]


def _load_raw_dataframe():
    """Baca DataFrame mentah, utamakan parquet (ringan), fallback ke xlsx.

    1. Jika ada parquet yang masih segar -> baca itu (cepat, ~2 detik).
    2. Jika tidak -> parse xlsx (lambat) lalu simpan ke parquet untuk
       pemuatan berikutnya.
    3. Jika keduanya tidak ada -> error yang jelas.
    """
    have_parquet = os.path.exists(_PARQUET_PATH)
    have_xlsx    = os.path.exists(_XLSX_PATH)

    # 1) Utamakan parquet bila ada & tidak lebih tua dari xlsx.
    if have_parquet:
        fresh = (not have_xlsx) or (
            os.path.getmtime(_PARQUET_PATH) >= os.path.getmtime(_XLSX_PATH)
        )
        if fresh:
            try:
                return pd.read_parquet(_PARQUET_PATH)
            except Exception:  # noqa: BLE001
                pass  # parquet rusak -> coba xlsx di bawah.

    # 2) Fallback: parse xlsx (lambat, ~100 detik untuk ~540K baris).
    if have_xlsx:
        raw = pd.read_excel(_XLSX_PATH, sheet_name=None, engine="openpyxl")
        df  = pd.concat(raw.values(), ignore_index=True)
        # Simpan ke parquet agar pemuatan berikutnya cepat.
        # Kolom tipe campuran (misal Invoice: int + 'C489449') harus jadi
        # StringDtype dulu agar PyArrow bisa menyimpannya.
        try:
            df_save = df.copy()
            for col in df_save.select_dtypes(include=["object", "str"]).columns:
                df_save[col] = df_save[col].astype("string")
            df_save.to_parquet(_PARQUET_PATH)
        except Exception:  # noqa: BLE001
            pass
        return df

    # 3) Tidak ada data sama sekali.
    raise FileNotFoundError(
        "Dataset tidak ditemukan di folder data/.\n\n"
        "Sertakan salah satu:\n"
        "  - online_retail_II.parquet  (disarankan, ringan)\n"
        "  - online_retail_II.xlsx     (akan dikonversi otomatis)"
    )


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Kolom tidak ditemukan. Dicari: {candidates}. "
        f"Tersedia: {list(df.columns)}"
    )


@st.cache_data(show_spinner="Membaca dataset lokal ...")
def load_data(top_n=50, min_weeks=8):
    """Baca & proses dataset dari file lokal.

    Returns
    -------
    dict:
        products    -> list[str] StockCode terurut berdasarkan jumlah data
        descriptions-> dict[StockCode -> str]
        demand      -> dict[StockCode -> pd.Series (index=minggu, value=Quantity)]
        avg_price   -> dict[StockCode -> float]
        raw_summary -> dict ringkasan jumlah baris
    """
    df = _load_raw_dataframe()

    rows_raw = len(df)

    qty_col   = _pick_col(df, _QUANTITY_CANDIDATES)
    price_col = _pick_col(df, _PRICE_CANDIDATES)
    stock_col = _pick_col(df, _STOCK_CANDIDATES)
    date_col  = _pick_col(df, _DATE_CANDIDATES)
    try:
        desc_col = _pick_col(df, _DESC_CANDIDATES)
    except KeyError:
        desc_col = None

    # --- Filter transaksi valid -------------------------------------------
    df = df.dropna(subset=[qty_col, price_col, stock_col, date_col])
    df[qty_col]   = pd.to_numeric(df[qty_col],   errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[qty_col, price_col])
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    rows_clean = len(df)

    # --- Agregasi mingguan per produk ------------------------------------
    df["_week"] = df[date_col].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = (
        df.groupby([stock_col, "_week"])[qty_col]
        .sum()
        .reset_index()
    )

    avg_price = df.groupby(stock_col)[price_col].mean()

    demand      = {}
    weeks_count = {}
    for code, grp in weekly.groupby(stock_col):
        series = grp.set_index("_week")[qty_col].sort_index()
        if len(series) >= min_weeks:
            demand[str(code)]      = series
            weeks_count[str(code)] = len(series)

    products = sorted(weeks_count, key=weeks_count.get, reverse=True)[:top_n]

    descriptions = {}
    if desc_col is not None:
        desc_map = (
            df.dropna(subset=[desc_col])
            .groupby(stock_col)[desc_col]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        )
        for code in products:
            descriptions[code] = str(desc_map.get(code, ""))
    else:
        descriptions = {code: "" for code in products}

    avg_price_map = {code: float(avg_price.get(code, 0.0)) for code in products}

    return {
        "products":     products,
        "descriptions": descriptions,
        "demand":       {code: demand[code] for code in products},
        "avg_price":    avg_price_map,
        "raw_summary": {
            "rows_raw":   rows_raw,
            "rows_clean": rows_clean,
            "n_products": len(products),
        },
    }
