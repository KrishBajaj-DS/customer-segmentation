"""
data_loader.py — Efficient loading & preprocessing of the Online Retail II dataset.

Strategy for 1M+ rows:
  - Read in chunks (chunksize=50_000) → concat after basic cleaning
  - Sample down to ~100k rows for exploratory work; use full data for model training
  - Cache processed RFM table to disk so Streamlit reruns skip heavy work
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
RAW_FILE_CSV  = os.path.join(DATA_DIR, "online_retail_II.csv")
RAW_FILE_XLSX = os.path.join(DATA_DIR, "online_retail_II.xlsx")
RFM_CACHE  = os.path.join(CACHE_DIR, "rfm.pkl")
TXN_CACHE  = os.path.join(CACHE_DIR, "transactions.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)


def _detect_raw_file() -> str:
    """Return the path to whichever raw file exists (CSV preferred over xlsx)."""
    if os.path.exists(RAW_FILE_CSV):
        return RAW_FILE_CSV
    if os.path.exists(RAW_FILE_XLSX):
        return RAW_FILE_XLSX
    raise FileNotFoundError(
        f"Dataset not found in {DATA_DIR}.\n"
        "Place 'online_retail_II.csv' (or .xlsx) in the /data folder.\n"
        "Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw load (chunked for memory safety)
# ─────────────────────────────────────────────────────────────────────────────
def load_raw(filepath: str = None, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load the Online Retail II dataset (CSV or xlsx).
    CSV is read in 50k-row chunks for memory safety.
    Pass sample_frac < 1.0 to work with a random subset.
    """
    if filepath is None:
        filepath = _detect_raw_file()

    print(f"[DataLoader] Reading {filepath} …")

    if filepath.endswith(".csv"):
        chunks = []
        for chunk in pd.read_csv(
            filepath,
            chunksize=50_000,
            dtype={"Customer ID": str},
            encoding="utf-8",
            on_bad_lines="skip",
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        # xlsx fallback
        chunks = []
        xl = pd.ExcelFile(filepath)
        for sheet in xl.sheet_names:
            df_chunk = pd.read_excel(xl, sheet_name=sheet, dtype={"Customer ID": str})
            chunks.append(df_chunk)
        df = pd.concat(chunks, ignore_index=True)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[DataLoader] Sampled to {len(df):,} rows ({sample_frac*100:.0f}%)")

    print(f"[DataLoader] Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Rename to consistent names
    rename = {
        "Invoice":     "InvoiceNo",
        "StockCode":   "StockCode",
        "Description": "Description",
        "Quantity":    "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price":       "UnitPrice",
        "Customer_ID": "CustomerID",
        "Country":     "Country",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Drop rows without CustomerID (guest checkouts) or negative quantity/price
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"]  > 0]
    df = df[df["UnitPrice"] > 0]

    # Remove cancelled invoices (start with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Coerce types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"]  = df["CustomerID"].astype(str).str.strip()
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    print(f"[DataLoader] After cleaning: {len(df):,} rows")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RFM Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency  = days since last purchase (lower = better)
    Frequency = number of distinct invoices
    Monetary  = total spend
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
            Frequency = ("InvoiceNo",    "nunique"),
            Monetary  = ("TotalPrice",   "sum"),
        )
        .reset_index()
    )
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Cached pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────
def get_rfm(force_rebuild: bool = False) -> pd.DataFrame:
    if not force_rebuild and os.path.exists(RFM_CACHE):
        print("[DataLoader] Loading RFM from cache …")
        return joblib.load(RFM_CACHE)

    df_raw   = load_raw()
    df_clean = clean(df_raw)

    rfm = build_rfm(df_clean)
    joblib.dump(rfm, RFM_CACHE)

    # Also cache transaction table for the recommender
    joblib.dump(df_clean, TXN_CACHE)

    print(f"[DataLoader] RFM built for {len(rfm):,} customers. Cached.")
    return rfm


def get_transactions(force_rebuild: bool = False) -> pd.DataFrame:
    if not force_rebuild and os.path.exists(TXN_CACHE):
        return joblib.load(TXN_CACHE)
    get_rfm(force_rebuild=True)          # rebuilds both
    return joblib.load(TXN_CACHE)