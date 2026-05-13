"""
08_churn_labeling.py
--------------------
STEP 8: Churn Label Creation — Time-Window Method

Methodology:
  Observation window : dataset_start  → 2011-06-30  (purchase history used for features)
  Prediction window  : 2011-07-01     → dataset_end  (used ONLY to define the label)

  Eligible customers : those with ≥1 purchase in the observation window
  churned = 1        : purchased in observation window, NOT in prediction window
  churned = 0        : purchased in BOTH windows

Leakage prevention:
  All features in churn_model_base are computed EXCLUSIVELY from observation_data
  (invoice_date < cutoff_date). No prediction-window information is used in any
  feature column. Full-period RFM segment is attached only to churn_by_rfm_segment.csv
  for descriptive reporting and is NOT included in churn_model_base.

Outputs:
  data/processed/churn_labels.csv / .parquet
  data/processed/churn_model_base.csv / .parquet
  reports/churn_label_summary.txt
  reports/churn_label_distribution.csv
  reports/churn_by_rfm_segment.csv
  reports/figures/churn_label_distribution.html
  reports/figures/churn_by_rfm_segment.html
  reports/figures/churn_by_recency_bucket.html
  reports/figures/churn_by_frequency_bucket.html
  reports/figures/churn_by_monetary_bucket.html

Run from project root:
    python src/08_churn_labeling.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PATH   = Path("data/processed/clean_transactions.parquet")
RFM_PATH     = Path("data/processed/rfm_segments.parquet")
REPORTS_DIR  = Path("reports")
FIGURES_DIR  = Path("reports/figures")
OUT_DIR      = Path("data/processed")

for p in [REPORTS_DIR, FIGURES_DIR, OUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ── Window definition ─────────────────────────────────────────────────────────
CUTOFF_DATE = pd.Timestamp("2011-07-01")


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_chart(fig, filename: str):
    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    print(f"  Chart saved -> {path}")


def safe_rank_score(series: pd.Series, n: int = 5,
                    higher_is_better: bool = True) -> pd.Series:
    """Rank-percentile scoring into 1..n buckets (same logic as Step 7)."""
    labels_asc = list(range(1, n + 1))
    rank_pct   = series.rank(method="first", ascending=True, pct=True)
    result     = pd.cut(rank_pct, bins=n, labels=labels_asc,
                        include_lowest=True).astype(int)
    if not higher_is_better:
        result = (n + 1) - result
    return result


def recency_bucket(days: pd.Series) -> pd.Series:
    bins   = [-1, 30, 60, 90, 180, 365, float("inf")]
    labels = ["0-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365+d"]
    return pd.cut(days, bins=bins, labels=labels)


def frequency_bucket(inv: pd.Series) -> pd.Series:
    bins   = [0, 1, 3, 5, 10, float("inf")]
    labels = ["1 invoice", "2-3", "4-5", "6-10", "11+"]
    return pd.cut(inv, bins=bins, labels=labels)


def monetary_bucket(rev: pd.Series) -> pd.Series:
    """Quantile-based 5-bucket split labelled Low → High."""
    labels = ["Low", "Low-Mid", "Mid", "Mid-High", "High"]
    try:
        return pd.qcut(rev, q=5, labels=labels, duplicates="drop")
    except Exception:
        rank_pct = rev.rank(method="first", pct=True)
        return pd.cut(rank_pct, bins=5, labels=labels, include_lowest=True)


def avg_gap_days(dates: pd.Series) -> float:
    """Mean gap between successive unique purchase dates; 0 for single-visit customers."""
    unique = sorted(dates.dt.normalize().unique())
    if len(unique) < 2:
        return 0.0
    gaps = [(unique[i+1] - unique[i]).days for i in range(len(unique) - 1)]
    return float(np.mean(gaps))


# ═════════════════════════════════════════════════════════════════════════════
# STEP A — Load & split
# ═════════════════════════════════════════════════════════════════════════════
def load_and_split(path: Path):
    print("\n── Loading clean transactions ───────────────────")
    df = pd.read_parquet(path, engine="pyarrow")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    required = ["customer_id", "invoice_id", "stock_code",
                "quantity", "invoice_date", "unit_price", "revenue", "country"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    dataset_start = df["invoice_date"].min()
    dataset_end   = df["invoice_date"].max()
    print(f"  Dataset start        : {dataset_start.date()}")
    print(f"  Dataset end          : {dataset_end.date()}")
    print(f"  Cutoff date          : {CUTOFF_DATE.date()}")
    print(f"  Total rows           : {len(df):,}")

    obs  = df[df["invoice_date"] <  CUTOFF_DATE].copy()
    pred = df[df["invoice_date"] >= CUTOFF_DATE].copy()
    print(f"\n  Observation rows     : {len(obs):,}")
    print(f"  Prediction rows      : {len(pred):,}")

    return df, obs, pred, dataset_start, dataset_end


# ═════════════════════════════════════════════════════════════════════════════
# STEP B — Churn labels
# ═════════════════════════════════════════════════════════════════════════════
def build_churn_labels(obs: pd.DataFrame, pred: pd.DataFrame,
                       dataset_start, dataset_end) -> pd.DataFrame:
    print("\n── Building churn labels ────────────────────────")
    obs_custs  = set(obs["customer_id"].unique())
    pred_custs = set(pred["customer_id"].unique())

    churned_n  = len(obs_custs - pred_custs)
    retained_n = len(obs_custs & pred_custs)
    total      = len(obs_custs)
    print(f"  Observation customers: {total:,}")
    print(f"  Prediction customers : {len(pred_custs):,}")
    print(f"  Churned              : {churned_n:,} ({churned_n/total*100:.1f}%)")
    print(f"  Retained             : {retained_n:,} ({retained_n/total*100:.1f}%)")

    churn_df = pd.DataFrame({
        "customer_id"              : sorted(obs_custs),
        "observation_start_date"   : dataset_start.date(),
        "observation_end_date"     : (CUTOFF_DATE - pd.Timedelta(days=1)).date(),
        "prediction_start_date"    : CUTOFF_DATE.date(),
        "prediction_end_date"      : dataset_end.date(),
        "purchased_in_observation" : 1,
        "purchased_in_prediction"  : [1 if cid in pred_custs else 0
                                       for cid in sorted(obs_custs)],
    })
    churn_df["churned"] = 1 - churn_df["purchased_in_prediction"]

    # Class balance check
    churn_rate = churn_df["churned"].mean() * 100
    if churn_rate < 20 or churn_rate > 80:
        print(f"  ⚠ CLASS IMBALANCE WARNING: churn rate = {churn_rate:.1f}%")
    else:
        print(f"  Class balance OK: churn={churn_rate:.1f}% retained={100-churn_rate:.1f}%")

    return churn_df


# ═════════════════════════════════════════════════════════════════════════════
# STEP C — Observation-only customer features (NO LEAKAGE)
# ═════════════════════════════════════════════════════════════════════════════
def build_obs_features(obs: pd.DataFrame) -> pd.DataFrame:
    """
    All features computed exclusively from observation_data (invoice_date < cutoff).
    obs_recency_days = CUTOFF_DATE − obs_last_purchase_date.
    No prediction-window data is touched here.
    """
    print("\n── Building observation-only features ───────────")
    snapshot = CUTOFF_DATE   # recency anchor = start of prediction window

    # ── Invoice-level aggregation ─────────────────────────────────────────
    inv = (
        obs.groupby(["customer_id", "invoice_id"], sort=False)
        .agg(
            inv_date    = ("invoice_date", "min"),
            inv_revenue = ("revenue",      "sum"),
            inv_qty     = ("quantity",     "sum"),
            inv_lines   = ("revenue",      "count"),
        )
        .reset_index()
    )

    # ── Date features ─────────────────────────────────────────────────────
    date_feats = (
        inv.groupby("customer_id")
        .agg(
            obs_first_purchase_date = ("inv_date", "min"),
            obs_last_purchase_date  = ("inv_date", "max"),
        )
        .reset_index()
    )
    date_feats["obs_tenure_days"]  = (
        date_feats["obs_last_purchase_date"] - date_feats["obs_first_purchase_date"]
    ).dt.days
    date_feats["obs_recency_days"] = (
        snapshot - date_feats["obs_last_purchase_date"]
    ).dt.days

    # ── Invoice frequency features ────────────────────────────────────────
    inv_feats = (
        inv.groupby("customer_id")
        .agg(
            obs_total_invoices       = ("invoice_id", "count"),
            obs_unique_purchase_days = ("inv_date",
                                        lambda x: x.dt.normalize().nunique()),
        )
        .reset_index()
    )

    # ── Transaction-level features ────────────────────────────────────────
    tx_feats = (
        obs.groupby("customer_id")
        .agg(
            obs_total_line_items = ("invoice_id",  "count"),
            obs_total_quantity   = ("quantity",    "sum"),
            obs_unique_products  = ("stock_code",  "nunique"),
        )
        .reset_index()
    )

    # ── Revenue features ──────────────────────────────────────────────────
    rev_feats = (
        inv.groupby("customer_id")
        .agg(
            obs_total_revenue     = ("inv_revenue", "sum"),
            obs_max_invoice_value = ("inv_revenue", "max"),
            obs_min_invoice_value = ("inv_revenue", "min"),
        )
        .reset_index()
    )

    # ── Country (most frequent in observation period) ─────────────────────
    country_feats = (
        obs.groupby("customer_id")["country"]
        .agg(lambda x: x.value_counts().sort_index().idxmax())
        .reset_index()
        .rename(columns={"country": "obs_country_mode"})
    )

    # ── avg days between purchases (observation only) ─────────────────────
    print("  Calculating obs avg_days_between_purchases ...")
    avg_gap = (
        obs.groupby("customer_id")["invoice_date"]
        .apply(avg_gap_days)
        .reset_index()
        .rename(columns={"invoice_date": "obs_avg_days_between_purchases"})
    )

    # ── Merge all feature groups ──────────────────────────────────────────
    feats = (
        date_feats
        .merge(inv_feats,     on="customer_id")
        .merge(tx_feats,      on="customer_id")
        .merge(rev_feats,     on="customer_id")
        .merge(country_feats, on="customer_id")
        .merge(avg_gap,       on="customer_id")
    )

    # ── Derived features ──────────────────────────────────────────────────
    months_active = (feats["obs_tenure_days"] / 30).clip(lower=1)
    feats["obs_purchase_frequency_per_month"] = feats["obs_total_invoices"] / months_active
    feats["obs_average_invoice_value"]        = (feats["obs_total_revenue"]
                                                  / feats["obs_total_invoices"])
    feats["obs_average_line_revenue"]         = (feats["obs_total_revenue"]
                                                  / feats["obs_total_line_items"])
    feats["obs_average_unit_price"]           = (feats["obs_total_revenue"]
                                                  / feats["obs_total_quantity"])
    feats["obs_products_per_invoice"]         = (feats["obs_unique_products"]
                                                  / feats["obs_total_invoices"])
    feats["obs_quantity_per_invoice"]         = (feats["obs_total_quantity"]
                                                  / feats["obs_total_invoices"])

    # ── Observation-only RFM scores ───────────────────────────────────────
    feats["obs_r_score"] = safe_rank_score(feats["obs_recency_days"],   higher_is_better=False)
    feats["obs_f_score"] = safe_rank_score(feats["obs_total_invoices"], higher_is_better=True)
    feats["obs_m_score"] = safe_rank_score(feats["obs_total_revenue"],  higher_is_better=True)
    feats["obs_rfm_total_score"] = (feats["obs_r_score"]
                                    + feats["obs_f_score"]
                                    + feats["obs_m_score"])

    print(f"  Obs features shape   : {feats.shape}")
    return feats


# ═════════════════════════════════════════════════════════════════════════════
# STEP D — Bucket analysis helpers
# ═════════════════════════════════════════════════════════════════════════════
def churn_by_bucket(base: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    return (
        base.groupby(bucket_col, observed=True)
        .agg(
            customers         = ("customer_id",       "count"),
            churned_customers = ("churned",           "sum"),
        )
        .reset_index()
        .assign(churn_rate=lambda d: (d["churned_customers"] / d["customers"] * 100).round(2))
        .sort_values(bucket_col)
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run():
    summary_lines = []

    def log(s=""):
        summary_lines.append(s)
        print(s)

    # ── Load & split ──────────────────────────────────────────────────────
    df, obs, pred, dataset_start, dataset_end = load_and_split(CLEAN_PATH)

    # ── Churn labels ──────────────────────────────────────────────────────
    churn_df = build_churn_labels(obs, pred, dataset_start, dataset_end)

    # ── Observation-only features ─────────────────────────────────────────
    obs_feats = build_obs_features(obs)

    # ── Build churn_model_base ─────────────────────────────────────────────
    print("\n── Merging label + obs features ─────────────────")
    base = churn_df[["customer_id", "churned", "purchased_in_prediction"]].merge(
        obs_feats, on="customer_id", how="left"
    )
    print(f"  churn_model_base shape: {base.shape}")

    # ── Ordered column list (per spec) ────────────────────────────────────
    ordered_cols = [
        "customer_id", "churned", "purchased_in_prediction",
        "obs_first_purchase_date", "obs_last_purchase_date",
        "obs_tenure_days", "obs_recency_days",
        "obs_total_invoices", "obs_total_line_items",
        "obs_total_quantity", "obs_unique_products",
        "obs_unique_purchase_days", "obs_total_revenue",
        "obs_average_invoice_value", "obs_average_line_revenue",
        "obs_average_unit_price", "obs_purchase_frequency_per_month",
        "obs_products_per_invoice", "obs_quantity_per_invoice",
        "obs_country_mode",
        # safe optional extras
        "obs_avg_days_between_purchases",
        "obs_r_score", "obs_f_score", "obs_m_score", "obs_rfm_total_score",
    ]
    base = base[ordered_cols]

    # ── Validation ────────────────────────────────────────────────────────
    print("\n── Running assertions ───────────────────────────")
    assert churn_df["customer_id"].is_unique,                   "FAIL: label customer_id not unique"
    assert base["customer_id"].is_unique,                       "FAIL: base customer_id not unique"
    assert len(base) == churn_df["customer_id"].nunique(),      "FAIL: row count mismatch"
    assert set(base["churned"].unique()).issubset({0, 1}),      "FAIL: invalid churned values"
    assert set(base["purchased_in_prediction"].unique()).issubset({0, 1}), \
                                                                "FAIL: invalid prediction flag"
    assert (base["churned"] == 1 - base["purchased_in_prediction"]).all(), \
                                                                "FAIL: churned != 1-prediction"
    assert (base["obs_recency_days"] >= 0).all(),               "FAIL: negative obs_recency_days"
    assert (base["obs_total_invoices"] >= 1).all(),             "FAIL: zero obs_total_invoices"
    assert (base["obs_total_revenue"]  > 0).all(),              "FAIL: zero obs_total_revenue"

    # Explicit leakage-column check
    leakage_cols = {"total_revenue", "total_invoices", "recency_days",
                    "rfm_segment", "customer_health_score"}
    found = leakage_cols & set(base.columns)
    assert not found, f"FAIL: leakage columns in model base: {found}"

    # Ensure no prediction-window dates crept into obs features
    max_obs_date = pd.to_datetime(base["obs_last_purchase_date"]).max()
    assert max_obs_date < CUTOFF_DATE, \
        f"FAIL: obs_last_purchase_date {max_obs_date} >= cutoff"

    print("  All assertions passed ✓")

    # ── Bucket analysis ───────────────────────────────────────────────────
    print("\n── Building bucket analysis ─────────────────────")
    base["obs_recency_bucket"]   = recency_bucket(base["obs_recency_days"])
    base["obs_frequency_bucket"] = frequency_bucket(base["obs_total_invoices"])
    base["obs_monetary_bucket"]  = monetary_bucket(base["obs_total_revenue"])

    rec_tbl  = churn_by_bucket(base, "obs_recency_bucket")
    freq_tbl = churn_by_bucket(base, "obs_frequency_bucket")
    mon_tbl  = churn_by_bucket(base, "obs_monetary_bucket")

    # ── RFM segment churn analysis (descriptive only) ─────────────────────
    print("\n── Merging full-period RFM segments (reporting only) ─")
    rfm = pd.read_parquet(RFM_PATH, engine="pyarrow")[["customer_id", "rfm_segment"]]
    seg_analysis = (
        churn_df[["customer_id", "churned"]]
        .merge(rfm, on="customer_id", how="left")
        .fillna({"rfm_segment": "No Segment (obs-window only)"})
        .groupby("rfm_segment")
        .agg(
            customers         = ("customer_id", "count"),
            churned_customers = ("churned",     "sum"),
        )
        .reset_index()
        .assign(
            churn_rate     = lambda d: (d["churned_customers"] / d["customers"] * 100).round(2),
            retained_customers = lambda d: d["customers"] - d["churned_customers"],
            retention_rate = lambda d: ((d["customers"] - d["churned_customers"])
                                         / d["customers"] * 100).round(2),
        )
        .sort_values("churn_rate", ascending=False)
    )

    # ── Distribution table ────────────────────────────────────────────────
    total = len(base)
    dist_tbl = (
        base["churned"]
        .value_counts()
        .reset_index()
        .rename(columns={"churned": "churned", "count": "customer_count"})
    )
    dist_tbl["customer_percent"] = (dist_tbl["customer_count"] / total * 100).round(2)
    dist_tbl = dist_tbl.sort_values("churned")

    # ── Charts ────────────────────────────────────────────────────────────
    print("\n── Saving charts ────────────────────────────────")

    # Churn distribution
    dist_plot = dist_tbl.copy()
    dist_plot["label"] = dist_plot["churned"].map({0: "Retained (0)", 1: "Churned (1)"})
    fig1 = px.bar(
        dist_plot, x="label", y="customer_count",
        color="label", text="customer_percent",
        title="Churn Label Distribution",
        labels={"label": "Churn Status", "customer_count": "Customers"},
        color_discrete_map={"Retained (0)": "#2ecc71", "Churned (1)": "#e74c3c"},
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    save_chart(fig1, "churn_label_distribution.html")

    # Churn by RFM segment
    fig2 = px.bar(
        seg_analysis.sort_values("churn_rate"),
        x="churn_rate", y="rfm_segment",
        orientation="h", text="churn_rate",
        title="Churn Rate by RFM Segment (full-period — descriptive only)",
        labels={"churn_rate": "Churn Rate (%)", "rfm_segment": "Segment"},
        color="churn_rate", color_continuous_scale="RdYlGn_r",
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    save_chart(fig2, "churn_by_rfm_segment.html")

    # Churn by recency bucket
    fig3 = px.bar(
        rec_tbl, x="obs_recency_bucket", y="churn_rate",
        text="churn_rate", title="Churn Rate by Observation Recency Bucket",
        labels={"obs_recency_bucket": "Recency Bucket", "churn_rate": "Churn Rate (%)"},
        color="churn_rate", color_continuous_scale="RdYlGn_r",
    )
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    save_chart(fig3, "churn_by_recency_bucket.html")

    # Churn by frequency bucket
    fig4 = px.bar(
        freq_tbl, x="obs_frequency_bucket", y="churn_rate",
        text="churn_rate", title="Churn Rate by Observation Frequency Bucket",
        labels={"obs_frequency_bucket": "Frequency Bucket", "churn_rate": "Churn Rate (%)"},
        color="churn_rate", color_continuous_scale="RdYlGn_r",
    )
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    save_chart(fig4, "churn_by_frequency_bucket.html")

    # Churn by monetary bucket
    fig5 = px.bar(
        mon_tbl, x="obs_monetary_bucket", y="churn_rate",
        text="churn_rate", title="Churn Rate by Observation Monetary Bucket (Quantile)",
        labels={"obs_monetary_bucket": "Revenue Bucket", "churn_rate": "Churn Rate (%)"},
        color="churn_rate", color_continuous_scale="RdYlGn_r",
    )
    fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    save_chart(fig5, "churn_by_monetary_bucket.html")

    # ── Save data files ───────────────────────────────────────────────────
    print("\n── Saving data outputs ──────────────────────────")

    # Drop bucket cols from base before saving (they are for reporting only)
    base_save = base.drop(columns=["obs_recency_bucket",
                                    "obs_frequency_bucket",
                                    "obs_monetary_bucket"])

    churn_df.to_csv(OUT_DIR / "churn_labels.csv", index=False)
    churn_df.to_parquet(OUT_DIR / "churn_labels.parquet", index=False, engine="pyarrow")
    base_save.to_csv(OUT_DIR / "churn_model_base.csv", index=False)
    base_save.to_parquet(OUT_DIR / "churn_model_base.parquet", index=False, engine="pyarrow")
    dist_tbl.to_csv(REPORTS_DIR / "churn_label_distribution.csv", index=False)
    seg_analysis.to_csv(REPORTS_DIR / "churn_by_rfm_segment.csv", index=False)

    for p in ["churn_labels.csv", "churn_labels.parquet",
              "churn_model_base.csv", "churn_model_base.parquet"]:
        print(f"  Saved: data/processed/{p}")
    for p in ["churn_label_distribution.csv", "churn_by_rfm_segment.csv"]:
        print(f"  Saved: reports/{p}")

    # ── Summary report ────────────────────────────────────────────────────
    print("\n── Saving summary report ────────────────────────")
    churn_n    = int(base_save["churned"].sum())
    retained_n = int((base_save["churned"] == 0).sum())
    churn_rate = churn_n / total * 100

    log("=" * 62)
    log("  STEP 8 — CHURN LABEL SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 62)
    log(f"\n  Dataset start date           : {dataset_start.date()}")
    log(f"  Dataset end date             : {dataset_end.date()}")
    log(f"  Cutoff date                  : {CUTOFF_DATE.date()}")
    log(f"  Observation window           : {dataset_start.date()} → 2011-06-30")
    log(f"  Prediction window            : 2011-07-01 → {dataset_end.date()}")

    log(f"\n  Total clean transaction rows : {len(df):,}")
    log(f"  Observation transaction rows : {len(obs):,}")
    log(f"  Prediction transaction rows  : {len(pred):,}")

    obs_custs = churn_df["customer_id"].nunique()
    pred_custs = pred["customer_id"].nunique()
    pred_only  = len(set(pred["customer_id"]) - set(obs["customer_id"]))
    log(f"\n  Observation customers        : {obs_custs:,}")
    log(f"  Prediction customers         : {pred_custs:,}")
    log(f"  Pred-only (excluded)         : {pred_only:,}  ← first appeared after cutoff")
    log(f"  Labeled customers            : {total:,}")
    log(f"  Churned  (label=1)           : {churn_n:,} ({churn_rate:.1f}%)")
    log(f"  Retained (label=0)           : {retained_n:,} ({100-churn_rate:.1f}%)")
    log(f"\n  Churn rate                   : {churn_rate:.1f}%")
    log(f"  Retention rate               : {100-churn_rate:.1f}%")

    if churn_rate < 20 or churn_rate > 80:
        log(f"\n  ⚠ CLASS IMBALANCE WARNING: Consider SMOTE or class weighting.")
    else:
        log(f"\n  Class balance               : GOOD — near 50/50 split.")
        log(f"  SMOTE / class weighting     : NOT required for this dataset.")

    log(f"\n  ── Leakage prevention note ─────────────────────────────")
    log(f"  churn_model_base uses ONLY observation-window features.")
    log(f"  obs_recency_days = CUTOFF_DATE − obs_last_purchase_date.")
    log(f"  No invoice_date >= {CUTOFF_DATE.date()} was used in any feature column.")
    log(f"  Full-period rfm_segment NOT included in churn_model_base.")
    log(f"  Confirmed: model base is leakage-free. ✓")

    log(f"\n  ── Model base columns ({len(base_save.columns)}) ───────────────────────────")
    for c in base_save.columns:
        log(f"    {c}")

    log(f"\n  ── Churn rate by obs_recency_bucket ────────────────────")
    for _, r in rec_tbl.iterrows():
        log(f"    {str(r['obs_recency_bucket']):<12} : {r['churn_rate']:>5.1f}% churn"
            f"  ({int(r['churned_customers'])}/{int(r['customers'])} customers)")

    log(f"\n  ── Churn rate by obs_frequency_bucket ──────────────────")
    for _, r in freq_tbl.iterrows():
        log(f"    {str(r['obs_frequency_bucket']):<12} : {r['churn_rate']:>5.1f}% churn"
            f"  ({int(r['churned_customers'])}/{int(r['customers'])} customers)")

    log(f"\n  ── Churn rate by obs_monetary_bucket ───────────────────")
    for _, r in mon_tbl.iterrows():
        log(f"    {str(r['obs_monetary_bucket']):<10} : {r['churn_rate']:>5.1f}% churn"
            f"  ({int(r['churned_customers'])}/{int(r['customers'])} customers)")

    with open(REPORTS_DIR / "churn_label_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary saved -> reports/churn_label_summary.txt")

    # ── Final printed terminal output ─────────────────────────────────────
    print("\n── Final results ────────────────────────────────")
    print(f"  Dataset range        : {dataset_start.date()} → {dataset_end.date()}")
    print(f"  Cutoff date          : {CUTOFF_DATE.date()}")
    print(f"  Observation rows     : {len(obs):,}")
    print(f"  Prediction rows      : {len(pred):,}")
    print(f"  Observation custs    : {obs_custs:,}")
    print(f"  Prediction custs     : {pred_custs:,}")
    print(f"  Churned              : {churn_n:,} ({churn_rate:.1f}%)")
    print(f"  Retained             : {retained_n:,} ({100-churn_rate:.1f}%)")
    print(f"  Class balance        : {'GOOD (near 50/50)' if 20 < churn_rate < 80 else 'IMBALANCED'}")
    print(f"  Model base           : observation-only features, no leakage ✓")
    print(f"\n  Output files:")
    for p in ["data/processed/churn_labels.csv",
              "data/processed/churn_labels.parquet",
              "data/processed/churn_model_base.csv",
              "data/processed/churn_model_base.parquet",
              "reports/churn_label_summary.txt",
              "reports/churn_label_distribution.csv",
              "reports/churn_by_rfm_segment.csv"]:
        print(f"    {p}")

    print("\n" + "=" * 62)
    print("  STEP 8 churn label creation completed successfully.")
    print("=" * 62)


if __name__ == "__main__":
    run()
