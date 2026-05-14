"""
recommender.py — Hybrid recommendation engine.

Architecture
────────────
1. Collaborative Filtering  — Surprise SVD on user × item purchase matrix
2. Content-Based            — TF-IDF on product Description → cosine similarity
3. Hybrid                   — weighted blend: α·CF + (1-α)·CB

Evaluation
──────────
Leave-one-out split → Precision@10, Recall@10
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
REC_CACHE   = os.path.join(CACHE_DIR, "recommender.pkl")
TFIDF_CACHE = os.path.join(CACHE_DIR, "tfidf.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build user-item matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_user_item(df: pd.DataFrame, min_purchases: int = 5) -> pd.DataFrame:
    """
    Aggregate purchase counts per (CustomerID, StockCode).
    Filter out customers with fewer than min_purchases to reduce noise.
    """
    # Count how many times a customer bought each product
    ui = (
        df.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "PurchaseCount"})
    )
    # Keep only active customers
    cust_counts = ui.groupby("CustomerID")["StockCode"].count()
    active      = cust_counts[cust_counts >= min_purchases].index
    ui          = ui[ui["CustomerID"].isin(active)]

    # Cap extreme purchase counts (log scale + clip)
    ui["Rating"] = np.log1p(ui["PurchaseCount"]).clip(upper=5.0)
    return ui


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Collaborative Filtering — Surprise SVD
# ─────────────────────────────────────────────────────────────────────────────
def train_svd(ui: pd.DataFrame, n_factors: int = 50) -> object:
    try:
        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import train_test_split as sur_split

        reader  = Reader(rating_scale=(0, 5))
        data    = Dataset.load_from_df(ui[["CustomerID", "StockCode", "Rating"]], reader)
        trainset, _ = sur_split(data, test_size=0.1, random_state=42)

        algo = SVD(n_factors=n_factors, n_epochs=20, lr_all=0.005,
                   reg_all=0.02, random_state=42)
        algo.fit(trainset)
        joblib.dump(algo, os.path.join(CACHE_DIR, "svd.pkl"))
        print(f"[SVD] Trained on {trainset.n_ratings:,} ratings.")
        return algo
    except ImportError:
        print("[SVD] scikit-surprise not installed — CF disabled.")
        return None


def cf_predict(algo, customer_id: str, all_items: list, purchased: set,
               top_n: int = 20) -> pd.DataFrame:
    """Return top-N unseen items for a customer from the SVD model."""
    if algo is None:
        return pd.DataFrame(columns=["StockCode", "CF_Score"])
    predictions = [
        (item, algo.predict(customer_id, item).est)
        for item in all_items
        if item not in purchased
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:top_n]
    return pd.DataFrame(top, columns=["StockCode", "CF_Score"])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Content-Based — TF-IDF on product descriptions
# ─────────────────────────────────────────────────────────────────────────────
def build_tfidf(df: pd.DataFrame) -> tuple:
    """
    Returns (item_df, tfidf_matrix, vectorizer).
    item_df has [StockCode, Description] — one row per product.
    """
    items = (
        df[["StockCode", "Description"]]
        .dropna()
        .drop_duplicates("StockCode")
        .reset_index(drop=True)
    )
    items["Description"] = items["Description"].str.lower().str.strip()

    vec    = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(items["Description"])

    joblib.dump((items, matrix, vec), TFIDF_CACHE)
    print(f"[TF-IDF] Built matrix {matrix.shape} for {len(items):,} products.")
    return items, matrix, vec


def cb_recommend_by_product(stock_code: str, items: pd.DataFrame,
                             tfidf_matrix, top_n: int = 10) -> pd.DataFrame:
    """Given a StockCode, find the most similar products."""
    if stock_code not in items["StockCode"].values:
        return pd.DataFrame(columns=["StockCode", "Description", "CB_Score"])

    idx  = items[items["StockCode"] == stock_code].index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = 0                          # exclude self

    top_idx = np.argsort(sims)[::-1][:top_n]
    result  = items.iloc[top_idx].copy()
    result["CB_Score"] = sims[top_idx]
    return result.reset_index(drop=True)


def cb_scores_for_customer(purchased_codes: list, items: pd.DataFrame,
                            tfidf_matrix, top_n: int = 30) -> pd.DataFrame:
    """Aggregate CB scores across all purchased items."""
    score_dict: dict = {}
    purchased_set = set(purchased_codes)

    for code in purchased_codes:
        if code not in items["StockCode"].values:
            continue
        idx  = items[items["StockCode"] == code].index[0]
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        for i, sim in enumerate(sims):
            sc = items.iloc[i]["StockCode"]
            if sc not in purchased_set:
                score_dict[sc] = score_dict.get(sc, 0.0) + sim

    if not score_dict:
        return pd.DataFrame(columns=["StockCode", "CB_Score"])

    cb_df = pd.DataFrame(list(score_dict.items()), columns=["StockCode", "CB_Score"])
    cb_df = cb_df.sort_values("CB_Score", ascending=False).head(top_n)
    return cb_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Hybrid blend
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_recommend(customer_id: str,
                     algo,
                     items: pd.DataFrame,
                     tfidf_matrix,
                     ui: pd.DataFrame,
                     alpha: float = 0.6,
                     top_n: int = 10) -> pd.DataFrame:
    """
    alpha  = weight for CF score (1-alpha for CB).
    Returns top_n recommendations with scores.
    """
    purchased = set(ui[ui["CustomerID"] == customer_id]["StockCode"].tolist())
    all_items = items["StockCode"].tolist()

    # CF scores
    cf_df = cf_predict(algo, customer_id, all_items, purchased, top_n=50)

    # CB scores
    purchased_list = list(purchased)
    cb_df = cb_scores_for_customer(purchased_list, items, tfidf_matrix, top_n=50)

    # Merge
    merged = pd.merge(cf_df, cb_df, on="StockCode", how="outer").fillna(0)

    # Normalise each score to [0, 1]
    scaler = MinMaxScaler()
    if merged["CF_Score"].max() > 0:
        merged["CF_Score"] = scaler.fit_transform(merged[["CF_Score"]])
    if merged["CB_Score"].max() > 0:
        merged["CB_Score"] = scaler.fit_transform(merged[["CB_Score"]])

    merged["Hybrid_Score"] = alpha * merged["CF_Score"] + (1 - alpha) * merged["CB_Score"]
    merged = merged.sort_values("Hybrid_Score", ascending=False).head(top_n)

    # Add descriptions
    merged = merged.merge(items[["StockCode", "Description"]], on="StockCode", how="left")
    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation — Leave-One-Out Precision@K / Recall@K
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_recommender(ui: pd.DataFrame, items: pd.DataFrame,
                          tfidf_matrix, algo,
                          k: int = 10, sample_users: int = 200) -> dict:
    """
    For each sampled user:
      - Hold out their most recently purchased item
      - Generate top-K recommendations
      - Precision@K = hits / K,  Recall@K = hits / |relevant|
    """
    # Only evaluate users with enough history
    user_counts = ui.groupby("CustomerID")["StockCode"].count()
    eligible    = user_counts[user_counts >= 5].index.tolist()
    sampled     = pd.Series(eligible).sample(
        min(sample_users, len(eligible)), random_state=42
    ).tolist()

    precisions, recalls = [], []

    for cid in sampled:
        user_items = ui[ui["CustomerID"] == cid].sort_values("Rating", ascending=False)
        if len(user_items) < 2:
            continue

        # Hold out top-rated item as ground truth
        held_out = user_items.iloc[0]["StockCode"]
        train_ui = ui[(ui["CustomerID"] != cid) |
                      (ui["StockCode"]   != held_out)]

        purchased  = set(train_ui[train_ui["CustomerID"] == cid]["StockCode"])
        all_items  = items["StockCode"].tolist()

        cf_df = cf_predict(algo, cid, all_items, purchased, top_n=k)
        recs  = set(cf_df["StockCode"].tolist())

        hit        = 1 if held_out in recs else 0
        n_relevant = 1   # leave-one-out: one held-out item

        precisions.append(hit / k)
        recalls.append(hit / n_relevant)

    return {
        f"Precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"Recall@{k}":    round(np.mean(recalls),    4) if recalls    else 0.0,
        "Users_Evaluated": len(precisions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Full pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────
def run_recommender_pipeline(df: pd.DataFrame,
                              force_rebuild: bool = False,
                              evaluate: bool = True) -> dict:
    if not force_rebuild and os.path.exists(REC_CACHE):
        print("[Recommender] Loading from cache …")
        return joblib.load(REC_CACHE)

    print("[Recommender] Building user-item matrix …")
    ui = build_user_item(df, min_purchases=3)

    print("[Recommender] Training SVD …")
    algo = train_svd(ui)

    print("[Recommender] Building TF-IDF index …")
    items, tfidf_matrix, vectorizer = build_tfidf(df)

    eval_metrics = {}
    if evaluate and algo is not None:
        print("[Recommender] Evaluating (Leave-One-Out) …")
        eval_metrics = evaluate_recommender(ui, items, tfidf_matrix, algo)
        print(f"[Recommender] Metrics: {eval_metrics}")

    result = {
        "ui":           ui,
        "algo":         algo,
        "items":        items,
        "tfidf_matrix": tfidf_matrix,
        "vectorizer":   vectorizer,
        "eval_metrics": eval_metrics,
    }
    joblib.dump(result, REC_CACHE)
    print("[Recommender] Done. Cached.")
    return result
