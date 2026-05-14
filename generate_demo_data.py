"""
generate_demo_data.py
─────────────────────
Generates a synthetic Online Retail II-compatible dataset (~50k rows)
so you can test the full pipeline without downloading the real 1M-row xlsx.

Usage:
    python generate_demo_data.py

Outputs:
    data/online_retail_II.xlsx
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_CUSTOMERS  = 2_000
N_PRODUCTS   = 400
N_ROWS       = 50_000
OUT_DIR      = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE     = os.path.join(OUT_DIR, "online_retail_II.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Product catalogue ─────────────────────────────────────────────────────────
ADJECTIVES = ["RED", "BLUE", "WHITE", "GREEN", "PINK", "VINTAGE", "FLORAL",
              "METAL", "WOODEN", "CERAMIC", "GLASS", "RETRO", "SPOTTED",
              "STRIPED", "LARGE", "SMALL", "MINI", "HEART", "STAR"]
NOUNS      = ["MUG", "CANDLE", "BAG", "HOLDER", "BOWL", "FRAME", "CLOCK",
              "LANTERN", "BASKET", "CUSHION", "VASE", "TRAY", "BOX",
              "NAPKIN", "PLATE", "CUP", "JAR", "BOTTLE", "HANGER", "SIGN"]

descriptions = [
    f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(ADJECTIVES)}"
    for _ in range(N_PRODUCTS)
]
stock_codes = [f"S{str(i).zfill(5)}" for i in range(N_PRODUCTS)]
unit_prices = np.round(np.random.lognormal(mean=1.5, sigma=0.7, size=N_PRODUCTS), 2)

product_df = pd.DataFrame({
    "StockCode":   stock_codes,
    "Description": descriptions,
    "UnitPrice":   unit_prices,
})

# ── Customer IDs ──────────────────────────────────────────────────────────────
customer_ids = [str(10000 + i) for i in range(N_CUSTOMERS)]

# ── Simulate purchase behaviour (3 customer tiers) ────────────────────────────
champion_ids    = customer_ids[:200]          # frequent, recent, high-value
at_risk_ids     = customer_ids[200:600]       # declining frequency
hibernating_ids = customer_ids[600:]          # rare purchasers

def rand_date(start, end):
    delta = end - start
    return start + pd.Timedelta(days=int(np.random.rand() * delta.days))

end_date   = pd.Timestamp("2011-12-09")
start_date = pd.Timestamp("2009-12-01")

rows = []
inv_counter = 500000

for _ in range(N_ROWS):
    tier = np.random.choice(["champion", "at_risk", "hibernating"],
                             p=[0.3, 0.4, 0.3])
    if tier == "champion":
        cid   = random.choice(champion_ids)
        date  = rand_date(pd.Timestamp("2011-06-01"), end_date)
        qty   = np.random.randint(3, 20)
    elif tier == "at_risk":
        cid   = random.choice(at_risk_ids)
        date  = rand_date(pd.Timestamp("2010-06-01"), pd.Timestamp("2011-03-01"))
        qty   = np.random.randint(1, 8)
    else:
        cid   = random.choice(hibernating_ids)
        date  = rand_date(start_date, pd.Timestamp("2010-06-01"))
        qty   = np.random.randint(1, 4)

    prod    = product_df.sample(1).iloc[0]
    inv_no  = str(inv_counter + np.random.randint(0, 50000))
    inv_counter += 1

    rows.append({
        "Invoice":     inv_no,
        "StockCode":   prod["StockCode"],
        "Description": prod["Description"],
        "Quantity":    qty,
        "InvoiceDate": date,
        "Price":       prod["UnitPrice"],
        "Customer ID": cid,
        "Country":     random.choice(["United Kingdom", "Germany", "France",
                                       "Netherlands", "Spain", "Australia"]),
    })

df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"Writing {len(df):,} rows to {OUT_FILE} …")
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    # Split into two "year" sheets like the real dataset
    df[df["InvoiceDate"].dt.year == 2009].to_excel(writer, sheet_name="Year 2009-2010", index=False)
    df[df["InvoiceDate"].dt.year >= 2010].to_excel(writer, sheet_name="Year 2010-2011", index=False)

print("Done! You can now run:  streamlit run app.py")
