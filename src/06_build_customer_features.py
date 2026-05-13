"""
06_build_customer_features.py
-----------------------------
STEP 6: Customer Feature Engineering

Reads clean_transactions.parquet and builds one row per customer containing
identity, date, frequency, revenue, behaviour, and return features.

This table feeds into:
  - Step 7  : RFM segmentation
  - Step 8  : Churn label creation
  - Step 9  : Churn model training
  - Step 11 : Customer health score
  - Step 12 : Streamlit customer lookup

Outputs:
  data/processed/customer_features.csv
  data/processed/customer_features.parquet
  reports/customer_features_summary.txt
  reports/customer_features_sample.csv   (top-10 rows for quick inspection)

Run from project root:
    python src/06_build_customer_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PATH   = Path("data/processed/clean_transactions.parquet")
RETURNS_PATH = Path("data/processed/returns.csv")
OUT_CSV      = Path("data/processed/customer_features.csv")
OUT_PARQUET  = Path("data/processed/customer_features.parquet")
SUMMARY_TXT  = Path("reports/customer_features_summary.txt")
SAMPLE_CSV   = Path("reports/customer_features_sample.csv")

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════
def load_transactions(path: Path) -> pd.DataFrame:
    print(f"\n── Loading clean transactions ───────────────────")
    df = pd.read_parquet(path, engine="pyarrow")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Confirm required columns
    required = ["customer_id", "invoice_id", "stock_code",
                "quantity", "invoice_date", "unit_price", "revenue", "country"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    print(f"  Rows loaded          : {len(df):,}")
    print(f"  Unique customers     : {df['customer_id'].nunique():,}")
    print(f"  Columns              : {list(df.columns)}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# INVOICE-LEVEL AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════
def build_invoice_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse line-items to one row per (customer_id, invoice_id).
    invoice_date = earliest timestamp on that invoice.
    """
    print("\n── Building invoice-level table ─────────────────")
    inv = (
        df.groupby(["customer_id", "invoice_id"], sort=False)
        .agg(
            invoice_date    = ("invoice_date", "min"),
            invoice_revenue = ("revenue",      "sum"),
            invoice_qty     = ("quantity",     "sum"),
            invoice_lines   = ("revenue",      "count"),
        )
        .reset_index()
    )
    print(f"  Invoice rows         : {len(inv):,}")
    return inv


# ═════════════════════════════════════════════════════════════════════════════
# DATE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def build_date_features(inv: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    print("\n── Building date features ───────────────────────")
    date_feats = (
        inv.groupby("customer_id")
        .agg(
            first_purchase_date = ("invoice_date", "min"),
            last_purchase_date  = ("invoice_date", "max"),
        )
        .reset_index()
    )
    date_feats["customer_tenure_days"]    = (
        date_feats["last_purchase_date"] - date_feats["first_purchase_date"]
    ).dt.days

    date_feats["recency_days"]            = (
        snapshot_date - date_feats["last_purchase_date"]
    ).dt.days

    # active_purchase_span_days == customer_tenure_days (same definition)
    date_feats["active_purchase_span_days"] = date_feats["customer_tenure_days"]

    print(f"  Date features built  : {date_feats.shape}")
    return date_feats


# ═════════════════════════════════════════════════════════════════════════════
# FREQUENCY FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def avg_days_between_purchases(dates: pd.Series) -> float:
    """
    Returns the average number of days between successive unique purchase dates.
    Returns 0.0 if the customer has only one unique purchase day.
    """
    unique_dates = sorted(dates.dt.normalize().unique())
    if len(unique_dates) < 2:
        return 0.0
    gaps = [(unique_dates[i+1] - unique_dates[i]).days
            for i in range(len(unique_dates) - 1)]
    return float(np.mean(gaps))


def build_frequency_features(df: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    print("\n── Building frequency features ──────────────────")

    # From invoice-level table
    inv_feats = (
        inv.groupby("customer_id")
        .agg(
            total_invoices       = ("invoice_id",      "count"),
            unique_purchase_days = ("invoice_date",
                                    lambda x: x.dt.normalize().nunique()),
        )
        .reset_index()
    )

    # From transaction-level table
    tx_feats = (
        df.groupby("customer_id")
        .agg(
            total_line_items = ("invoice_id",  "count"),
            total_quantity   = ("quantity",    "sum"),
            unique_products  = ("stock_code",  "nunique"),
        )
        .reset_index()
    )

    # avg_days_between_purchases — requires per-customer date series
    print("  Calculating avg days between purchases (per customer) ...")
    avg_gap = (
        df.groupby("customer_id")["invoice_date"]
        .apply(avg_days_between_purchases)
        .reset_index()
        .rename(columns={"invoice_date": "avg_days_between_purchases"})
    )

    freq = inv_feats.merge(tx_feats, on="customer_id").merge(avg_gap, on="customer_id")
    print(f"  Frequency features   : {freq.shape}")
    return freq


# ═════════════════════════════════════════════════════════════════════════════
# REVENUE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def build_revenue_features(inv: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Building revenue features ─────────────────────")

    # Invoice-level revenue stats
    inv_rev = (
        inv.groupby("customer_id")
        .agg(
            total_revenue     = ("invoice_revenue", "sum"),
            max_invoice_value = ("invoice_revenue", "max"),
            min_invoice_value = ("invoice_revenue", "min"),
        )
        .reset_index()
    )

    rev = inv_rev.copy()
    print(f"  Revenue features     : {rev.shape}")
    return rev


# ═════════════════════════════════════════════════════════════════════════════
# COUNTRY
# ═════════════════════════════════════════════════════════════════════════════
def build_country_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Most-frequent country per customer. Ties broken alphabetically."""
    print("\n── Building country feature ──────────────────────")
    country_feats = (
        df.groupby("customer_id")["country"]
        .agg(lambda x: x.value_counts().sort_index().idxmax())
        .reset_index()
        .rename(columns={"country": "country_mode"})
    )
    print(f"  Country feature      : {country_feats.shape}")
    return country_feats


# ═════════════════════════════════════════════════════════════════════════════
# RETURN FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def build_return_features(returns_path: Path, valid_customer_ids: set) -> pd.DataFrame | None:
    print("\n── Building return features ─────────────────────")
    if not returns_path.exists():
        print("  WARNING: returns.csv not found. Skipping return features.")
        return None

    ret = pd.read_csv(returns_path)

    if "customer_id" not in ret.columns:
        print("  WARNING: customer_id not in returns.csv. Skipping return features.")
        return None

    # 714 rows have null customer_id — drop those
    before = len(ret)
    ret = ret.dropna(subset=["customer_id"])
    ret["customer_id"] = ret["customer_id"].astype(int)
    print(f"  Returns loaded       : {before:,} rows, {before - len(ret):,} dropped (null customer_id)")
    print(f"  Returns usable       : {len(ret):,} rows")

    # Only include return rows for customers that exist in sales
    ret = ret[ret["customer_id"].isin(valid_customer_ids)]
    print(f"  Returns matched      : {len(ret):,} rows for {ret['customer_id'].nunique():,} customers")

    # quantity in returns is negative; use abs for magnitude
    ret_feats = (
        ret.groupby("customer_id")
        .agg(
            total_returns      = ("invoice_id",  "count"),
            return_quantity_abs = ("quantity",   lambda x: abs(x).sum()),
        )
        .reset_index()
    )
    ret_feats["has_returned"] = 1
    print(f"  Return features      : {ret_feats.shape}")
    return ret_feats


# ═════════════════════════════════════════════════════════════════════════════
# DERIVED BEHAVIOUR FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def add_derived_features(cf: pd.DataFrame) -> pd.DataFrame:
    print("\n── Adding derived behaviour features ─────────────")

    cf["is_repeat_buyer"]            = (cf["total_invoices"] > 1).astype(int)

    # purchase_frequency_per_month: orders per month, using max(tenure/30, 1)
    months_active = (cf["customer_tenure_days"] / 30).clip(lower=1)
    cf["purchase_frequency_per_month"] = cf["total_invoices"] / months_active

    cf["average_invoice_value"]      = cf["total_revenue"]    / cf["total_invoices"]
    cf["average_line_revenue"]       = cf["total_revenue"]    / cf["total_line_items"]
    cf["average_unit_price"]         = cf["total_revenue"]    / cf["total_quantity"]
    cf["quantity_per_invoice"]       = cf["total_quantity"]   / cf["total_invoices"]
    cf["products_per_invoice"]       = cf["unique_products"]  / cf["total_invoices"]

    print(f"  Derived features added. Total columns: {cf.shape[1]}")
    return cf


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ═════════════════════════════════════════════════════════════════════════════
def validate(cf: pd.DataFrame, df_clean: pd.DataFrame):
    print("\n── Running assertions ───────────────────────────")

    assert "customer_id" in cf.columns,                    "FAIL: no customer_id"
    assert cf["customer_id"].is_unique,                    "FAIL: customer_id not unique"
    assert len(cf) == df_clean["customer_id"].nunique(),   "FAIL: row count mismatch"
    assert cf["customer_id"].isnull().sum() == 0,          "FAIL: null customer_ids"
    assert (cf["total_revenue"]    > 0).all(),             "FAIL: zero/negative revenue"
    assert (cf["total_invoices"]   >= 1).all(),            "FAIL: zero invoices"
    assert (cf["total_quantity"]   > 0).all(),             "FAIL: zero quantity"
    assert (cf["recency_days"]     >= 0).all(),            "FAIL: negative recency"
    assert (cf["customer_tenure_days"] >= 0).all(),        "FAIL: negative tenure"
    assert (cf["average_invoice_value"] > 0).all(),        "FAIL: zero avg invoice value"
    assert (cf["purchase_frequency_per_month"] > 0).all(), "FAIL: zero purchase frequency"

    # Revenue reconciliation (allow £0.01 rounding tolerance)
    rev_cf    = round(cf["total_revenue"].sum(), 2)
    rev_clean = round(df_clean["revenue"].sum(), 2)
    assert abs(rev_cf - rev_clean) < 0.02, \
        f"FAIL: revenue mismatch  features={rev_cf}  transactions={rev_clean}"

    print("  All assertions passed ✓")
    return rev_cf, rev_clean


# ═════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def save_summary(cf: pd.DataFrame, df_clean: pd.DataFrame,
                 snapshot_date: pd.Timestamp,
                 rev_cf: float, rev_clean: float,
                 return_features_included: bool):
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 58)
    log("  STEP 6 — CUSTOMER FEATURES SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 58)

    log(f"\n  Input transaction rows       : {len(df_clean):,}")
    log(f"  Unique customers (source)    : {df_clean['customer_id'].nunique():,}")
    log(f"  Snapshot date used           : {snapshot_date.date()}")
    log(f"  Final customer feature rows  : {len(cf):,}")
    log(f"  Total feature columns        : {cf.shape[1]}")

    log(f"\n  ── Revenue validation ──────────────────────────")
    log(f"  Total revenue (features)     : £{rev_cf:,.2f}")
    log(f"  Total revenue (transactions) : £{rev_clean:,.2f}")
    log(f"  Difference                   : £{abs(rev_cf - rev_clean):.2f}  ✓")

    repeat_n   = cf["is_repeat_buyer"].sum()
    repeat_pct = repeat_n / len(cf) * 100
    log(f"\n  ── Buyer behaviour ─────────────────────────────")
    log(f"  Repeat buyers                : {repeat_n:,} ({repeat_pct:.1f}%)")
    log(f"  One-time buyers              : {len(cf) - repeat_n:,} ({100 - repeat_pct:.1f}%)")

    log(f"\n  ── Revenue distribution ────────────────────────")
    log(f"  Mean revenue per customer    : £{cf['total_revenue'].mean():,.2f}")
    log(f"  Median revenue per customer  : £{cf['total_revenue'].median():,.2f}")
    log(f"  Min revenue per customer     : £{cf['total_revenue'].min():,.2f}")
    log(f"  Max revenue per customer     : £{cf['total_revenue'].max():,.2f}")

    log(f"\n  ── Invoice frequency ───────────────────────────")
    log(f"  Mean invoices per customer   : {cf['total_invoices'].mean():.2f}")
    log(f"  Median invoices per customer : {cf['total_invoices'].median():.1f}")
    log(f"  Max invoices (single cust.)  : {cf['total_invoices'].max()}")

    log(f"\n  ── Recency ─────────────────────────────────────")
    log(f"  Mean recency_days            : {cf['recency_days'].mean():.1f}")
    log(f"  Median recency_days          : {cf['recency_days'].median():.1f}")
    log(f"  Min recency_days             : {cf['recency_days'].min()}")
    log(f"  Max recency_days             : {cf['recency_days'].max()}")

    log(f"\n  ── Tenure ──────────────────────────────────────")
    log(f"  Mean tenure_days             : {cf['customer_tenure_days'].mean():.1f}")
    log(f"  Median tenure_days           : {cf['customer_tenure_days'].median():.1f}")

    log(f"\n  ── Top 10 customers by revenue ─────────────────")
    top10 = cf.nlargest(10, "total_revenue")[["customer_id", "total_revenue",
                                               "total_invoices", "recency_days"]]
    for _, row in top10.iterrows():
        log(f"    {int(row['customer_id']):<8}  £{row['total_revenue']:>12,.2f}"
            f"  {int(row['total_invoices']):>4} invoices"
            f"  recency {int(row['recency_days'])} days")

    log(f"\n  ── Return features ─────────────────────────────")
    if return_features_included:
        ret_n = cf["has_returned"].sum()
        log(f"  Customers with returns       : {ret_n:,} ({ret_n/len(cf)*100:.1f}%)")
        log(f"  Return features included     : total_returns, return_quantity_abs,"
            f" has_returned, return_rate")
    else:
        log("  Return features              : SKIPPED (no usable return data)")

    log(f"\n  ── Output files ────────────────────────────────")
    log(f"  {OUT_CSV}")
    log(f"  {OUT_PARQUET}")
    log(f"  {SAMPLE_CSV}")

    with open(SUMMARY_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved -> {SUMMARY_TXT}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run():
    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_transactions(CLEAN_PATH)

    # Snapshot date = day after last transaction in the dataset
    snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    print(f"\n  Snapshot date        : {snapshot_date.date()}")

    # ── Build component tables ─────────────────────────────────────────────
    inv         = build_invoice_level(df)
    date_feats  = build_date_features(inv, snapshot_date)
    freq_feats  = build_frequency_features(df, inv)
    rev_feats   = build_revenue_features(inv, df)
    country_ft  = build_country_feature(df)

    # ── Merge all components ──────────────────────────────────────────────
    print("\n── Merging all feature tables ────────────────────")
    cf = (
        date_feats
        .merge(freq_feats,  on="customer_id", how="left")
        .merge(rev_feats,   on="customer_id", how="left")
        .merge(country_ft,  on="customer_id", how="left")
    )
    print(f"  Merged shape         : {cf.shape}")

    # ── Return features ────────────────────────────────────────────────────
    valid_ids         = set(cf["customer_id"].unique())
    ret_feats         = build_return_features(RETURNS_PATH, valid_ids)
    return_included   = ret_feats is not None

    if return_included:
        cf = cf.merge(ret_feats, on="customer_id", how="left")
        cf["total_returns"]       = cf["total_returns"].fillna(0).astype(int)
        cf["return_quantity_abs"] = cf["return_quantity_abs"].fillna(0).astype(int)
        cf["has_returned"]        = cf["has_returned"].fillna(0).astype(int)
        # return_rate: return rows / sales invoices
        cf["return_rate"]         = cf["total_returns"] / cf["total_invoices"]

    # ── Derived behaviour features ─────────────────────────────────────────
    cf = add_derived_features(cf)

    # ── Column ordering ────────────────────────────────────────────────────
    identity_cols  = ["customer_id", "country_mode"]
    date_cols      = ["first_purchase_date", "last_purchase_date",
                      "customer_tenure_days", "recency_days", "active_purchase_span_days"]
    freq_cols      = ["total_invoices", "total_line_items", "total_quantity",
                      "unique_products", "unique_purchase_days", "avg_days_between_purchases"]
    rev_cols       = ["total_revenue", "average_invoice_value", "average_line_revenue",
                      "average_unit_price", "max_invoice_value", "min_invoice_value"]
    behav_cols     = ["is_repeat_buyer", "purchase_frequency_per_month",
                      "quantity_per_invoice", "products_per_invoice"]
    return_cols    = (["total_returns", "return_quantity_abs", "has_returned", "return_rate"]
                      if return_included else [])

    ordered_cols   = (identity_cols + date_cols + freq_cols +
                      rev_cols + behav_cols + return_cols)
    cf = cf[ordered_cols]

    # ── Validate ───────────────────────────────────────────────────────────
    rev_cf, rev_clean = validate(cf, df)

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n── Saving outputs ────────────────────────────────")
    cf.to_csv(OUT_CSV, index=False)
    cf.to_parquet(OUT_PARQUET, index=False, engine="pyarrow")
    cf.head(10).to_csv(SAMPLE_CSV, index=False)
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Saved: {SAMPLE_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────
    save_summary(cf, df, snapshot_date, rev_cf, rev_clean, return_included)

    # ── Final printed metrics ──────────────────────────────────────────────
    print("\n── Key metrics ───────────────────────────────────")
    print(f"  Customer feature rows  : {len(cf):,}")
    print(f"  Feature columns        : {cf.shape[1]}")
    print(f"  Total revenue check    : £{rev_cf:,.2f}  (matches source ✓)")
    repeat_n = int(cf["is_repeat_buyer"].sum())
    print(f"  Repeat buyers          : {repeat_n:,} ({repeat_n/len(cf)*100:.1f}%)")
    print("\n  Top 5 customers by revenue:")
    for _, row in cf.nlargest(5, "total_revenue").iterrows():
        print(f"    Customer {int(row['customer_id'])}  "
              f"£{row['total_revenue']:>12,.2f}  "
              f"{int(row['total_invoices'])} invoices  "
              f"recency {int(row['recency_days'])}d")

    print("\n" + "=" * 58)
    print("  STEP 6 customer feature engineering completed successfully.")
    print("=" * 58)


if __name__ == "__main__":
    run()
