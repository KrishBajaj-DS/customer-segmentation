# 🔮 Customer Segmentation + Recommendation Engine

> **BTech Final Year Project** — Production-grade ML system with RFM clustering,
> hybrid recommendations, and interactive Streamlit dashboard.

---

## 📁 Folder Structure

```
customer_segmentation/
├── app.py                      ← Streamlit main entry-point
├── generate_demo_data.py       ← Synthetic data generator (no download needed)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml             ← Streamlit theme + server settings
│   └── secrets.toml            ← (not committed) API keys etc.
├── data/
│   └── online_retail_II.xlsx   ← Real or generated dataset (gitignored)
├── models/                     ← Auto-created pickle cache (gitignored)
│   ├── rfm.pkl
│   ├── transactions.pkl
│   ├── segmentation.pkl
│   ├── recommender.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── umap.pkl
│   ├── svd.pkl
│   └── tfidf.pkl
├── utils/
│   ├── data_loader.py          ← Chunked loading, cleaning, RFM engineering
│   ├── segmentation.py         ← K-Means, DBSCAN, UMAP, labels, metrics
│   └── recommender.py          ← SVD CF + TF-IDF CB + hybrid blend + eval
└── pages/                      ← (reserved for future multi-page expansion)
```

---

## ⚡ Quick Start (Local)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd customer_segmentation

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2a. Use the real dataset (recommended for resume)

Download **Online Retail II.xlsx** from:
- UCI: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Place it in `data/online_retail_II.xlsx`.

### 2b. Generate synthetic demo data (instant, no download)

```bash
python generate_demo_data.py
```

Generates ~50k rows in ~10 seconds — full pipeline works identically.

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 — the first run builds all caches (~2-5 min on real data,
~30s on demo data). Subsequent runs load from cache instantly.

---

## 🏗️ Architecture

### Part 1 — Customer Segmentation

```
Raw XLSX (1M+ rows)
    │
    ▼  [chunked read + clean]
Transactions (CustomerID, InvoiceNo, StockCode, Quantity, UnitPrice, Date)
    │
    ▼  [RFM engineering]
RFM Table (Recency, Frequency, Monetary per customer)
    │
    ├──▶ RobustScaler → K-Means (elbow + silhouette → optimal K)
    ├──▶ RobustScaler → DBSCAN (eps, min_samples tuned)
    └──▶ UMAP 2D projection → interactive Plotly scatter
             │
             ▼  [rule-based label engine]
    Segment Labels: Champions / Potential Loyalists / New Customers /
                    At-Risk / Hibernating / Lost
```

**Why RobustScaler?** Monetary values have extreme outliers (bulk buyers).
RobustScaler uses IQR and is outlier-resistant — better than StandardScaler here.

**Why UMAP over PCA?** UMAP preserves local *and* global structure; PCA only
captures linear variance. The clusters look more meaningful visually in UMAP.

### Part 2 — Recommendation Engine

```
Transactions
    │
    ├──▶ User × Item matrix (CustomerID × StockCode, rating = log(Quantity))
    │        │
    │        └──▶ Surprise SVD (n_factors=50, 20 epochs)
    │                  → CF scores for unseen items
    │
    └──▶ Product descriptions
             │
             └──▶ TF-IDF (5000 features, bigrams) → Cosine Similarity
                      → CB scores per purchased product → aggregated
    │
    ▼
Hybrid Score = α × CF_score + (1-α) × CB_score   (α tunable in UI)
    │
    ▼
Evaluation: Leave-One-Out → Precision@10, Recall@10
```

### Part 3 — Streamlit UI

| Page | Features |
|------|----------|
| Overview | KPI cards, UMAP, Elbow/Silhouette charts, Segment pie, Revenue table |
| Customer Lookup | Segment badge, RFM metrics, top-10 hybrid recs with score bars |
| Cluster Explorer | Multi-select filter, DBSCAN comparison, sample table |
| Product Similarity | Keyword search → TF-IDF top-K similar products |

---

## 📊 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Silhouette Score | Cluster separation quality (higher = better, range −1 to 1) |
| Davies-Bouldin Index | Avg similarity ratio of clusters (lower = better) |
| Precision@10 | Of 10 recommendations, how many are relevant |
| Recall@10 | Of all relevant items, how many are in top 10 |

---

## 🌐 Deploy on Streamlit Cloud

### Step 1 — Prepare the repo

The models/ and data/ folders are gitignored (too large).
For cloud deployment you have two options:

**Option A — Use demo data (easiest)**
Add a `startup.py` or call `generate_demo_data.py` via `@st.cache_resource` on first run.

**Option B — Use GitHub LFS for the real dataset**
```bash
git lfs install
git lfs track "data/*.xlsx"
git add .gitattributes data/online_retail_II.xlsx
git commit -m "add dataset via LFS"
```

### Step 2 — Deploy

1. Push your repo to GitHub
2. Go to https://share.streamlit.io → **New app**
3. Select repo, branch, and set **Main file path** = `app.py`
4. Under **Advanced settings** → increase memory if needed (1GB+ recommended for real data)
5. Click **Deploy**

### Step 3 — Environment

Streamlit Cloud uses `requirements.txt` automatically. No Dockerfile needed.

If umap-learn fails to install on Cloud, add this to requirements.txt:
```
numba==0.59.1
llvmlite==0.42.0
umap-learn==0.5.6
```

---

## 🚀 Handling Large Datasets (1M+ rows)

The codebase uses several strategies:

| Strategy | Where | Effect |
|----------|-------|--------|
| Chunked Excel read | `data_loader.py:load_raw()` | Reads sheet-by-sheet, avoids OOM |
| `sample_frac` param | `load_raw(sample_frac=0.3)` | Quick iteration on 30% sample |
| `joblib.dump/load` | All pipeline files | Skip re-computation on rerun |
| `@st.cache_resource` | `app.py:load_all()` | Streamlit-level caching per session |
| RobustScaler | `segmentation.py` | Handles monetary outliers |
| `log1p` capping | `recommender.py:build_user_item()` | Prevents bulk-order dominance |
| `min_purchases` filter | `recommender.py` | Removes cold-start users |

For very large data (>500k after cleaning), consider:
- Using `pandas.read_csv` with `chunksize=50000` if you convert to CSV first
- Running PCA (50 components) before UMAP to speed up embedding
- Training SVD on a stratified sample, then predicting for all

---

## 🎓 For Your Resume/Interview

**Key talking points:**
1. "I used RobustScaler instead of StandardScaler because RFM Monetary has extreme outliers from bulk buyers — IQR-based scaling is more appropriate."
2. "I chose UMAP over t-SNE for the final visualization because UMAP preserves global cluster structure and scales better to 5000+ points."
3. "The hybrid recommender uses a tunable α parameter so business teams can shift from pure collaborative to content-based filtering depending on data sparsity."
4. "I implemented leave-one-out evaluation rather than random splits because in recommendation systems, temporal holdout is more realistic."
5. "All models are cached to disk with joblib so the Streamlit app loads in under 2 seconds after the first build."

---

## 📦 Tech Stack

- **Data**: pandas, numpy, openpyxl
- **ML**: scikit-learn (KMeans, DBSCAN, TF-IDF, PCA, RobustScaler)
- **Dim. Reduction**: umap-learn
- **Recommender**: scikit-surprise (SVD)
- **Visualization**: plotly
- **UI**: streamlit
- **Serialization**: joblib
