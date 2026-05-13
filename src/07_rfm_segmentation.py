"""
07_rfm_segmentation.py
----------------------
STEP 7: RFM Segmentation

Reads customer_features.parquet and adds:
  - r_score, f_score, m_score  (1–5 each)
  - rfm_score                  (string e.g. "543")
  - rfm_total_score            (sum of R+F+M, 3–15)
  - rfm_average_score          (mean of R+F+M)
  - rfm_segment                (named business segment)
  - recommended_action         (business action string)

Scoring notes:
  Recency  : LOWER recency_days is BETTER → r_score=5 for most recent
  Frequency: HIGHER total_invoices is BETTER → f_score=5 for most frequent
  Monetary : HIGHER total_revenue is BETTER → m_score=5 for highest revenue

  pd.qcut is attempted first (duplicates="drop").
  If fewer than 5 unique bins result, a rank-percentile fallback is used
  to guarantee all 5 score levels appear.

Outputs:
  data/processed/rfm_segments.csv
  data/processed/rfm_segments.parquet
  reports/rfm_segmentation_summary.txt
  reports/rfm_segment_counts.csv
  reports/rfm_segment_revenue.csv
  reports/figures/rfm_segment_counts.html
  reports/figures/rfm_segment_revenue.html
  reports/figures/rfm_recency_distribution.html
  reports/figures/rfm_frequency_distribution.html
  reports/figures/rfm_monetary_distribution.html

Run from project root:
    python src/07_rfm_segmentation.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CF_PATH      = Path("data/processed/customer_features.parquet")
OUT_CSV      = Path("data/processed/rfm_segments.csv")
OUT_PARQUET  = Path("data/processed/rfm_segments.parquet")
REPORTS_DIR  = Path("reports")
FIGURES_DIR  = Path("reports/figures")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Scoring helpers ────────────────────────────────────────────────────────────
def safe_qcut_score(series: pd.Series, n: int = 5,
                    higher_is_better: bool = True) -> pd.Series:
    """
    Score a pandas Series into 1..n integer buckets.

    Strategy:
      1. Try pd.qcut(series, q=n, duplicates='drop').
      2. If fewer than n unique bins result, fall back to rank-percentile
         cutting so we always get n distinct score levels.

    Parameters
    ----------
    higher_is_better : bool
        True  → highest value receives score n   (frequency, monetary)
        False → lowest  value receives score n   (recency)
    """
    labels_asc = list(range(1, n + 1))           # [1,2,3,4,5] low→high value
    labels_inv = list(range(n, 0, -1))           # [5,4,3,2,1] low→high value

    assign_labels = labels_asc if higher_is_better else labels_inv

    # ── Attempt 1: standard qcut ──────────────────────────────────────────
    try:
        result = pd.qcut(series, q=n, labels=assign_labels, duplicates="drop")
        if result.nunique() == n:
            return result.astype(int)
    except Exception:
        pass

    # ── Fallback: rank-percentile cut ─────────────────────────────────────
    # Rank by value (higher value → higher rank).
    # Then cut the rank into n equal-sized buckets.
    rank_pct = series.rank(method="first", ascending=True, pct=True)
    result   = pd.cut(
        rank_pct,
        bins=n,
        labels=labels_asc,
        include_lowest=True,
    ).astype(int)

    # If higher_is_better=False, invert so smallest value → highest score
    if not higher_is_better:
        result = (n + 1) - result

    return result


# ── Segment assignment (priority order) ───────────────────────────────────────
SEGMENT_RULES = [
    # (segment_name, condition_fn)
    # Priority 1 — highest value, high engagement
    ("Champions",
     lambda d: (d["r_score"] >= 4) & (d["f_score"] >= 4) & (d["m_score"] >= 4)),

    # Priority 2 — used to be best, now gone quiet
    ("Cannot Lose",
     lambda d: (d["r_score"] <= 2) & (d["f_score"] >= 4) & (d["m_score"] >= 4)),

    # Priority 3 — loyal and valuable
    ("Loyal Customers",
     lambda d: (d["r_score"] >= 3) & (d["f_score"] >= 4) & (d["m_score"] >= 3)),

    # Priority 4 — high spend but low visit frequency
    ("Big Spenders",
     lambda d: (d["m_score"] >= 5) & (d["f_score"] <= 3)),

    # Priority 5 — previously active, now at risk
    ("At Risk",
     lambda d: (d["r_score"] <= 2) & (d["f_score"] >= 3) & (d["m_score"] >= 3)),

    # Priority 6 — recent but not yet frequent
    ("Potential Loyalists",
     lambda d: (d["r_score"] >= 4) & (d["f_score"].between(2, 3))),

    # Priority 7 — just arrived
    ("New Customers",
     lambda d: (d["r_score"] >= 4) & (d["f_score"] == 1)),

    # Priority 8 — inactive and infrequent
    ("Hibernating",
     lambda d: (d["r_score"] <= 2) & (d["f_score"] <= 2)),

    # Priority 9 — overall low combined score
    ("Low Value",
     lambda d: d["rfm_total_score"] <= 6),

    # Priority 10 — catch-all
    ("Needs Attention",
     lambda d: pd.Series(True, index=d.index)),
]

SEGMENT_ACTIONS = {
    "Champions":          "Reward with VIP offers and early access.",
    "Cannot Lose":        "Prioritize personalized retention outreach.",
    "Loyal Customers":    "Offer loyalty benefits and cross-sell bundles.",
    "Big Spenders":       "Provide premium service and personalized offers.",
    "At Risk":            "Trigger win-back campaign and discount offer.",
    "Potential Loyalists":"Encourage repeat purchase with targeted promotions.",
    "New Customers":      "Send onboarding offers and product recommendations.",
    "Hibernating":        "Send reactivation campaign.",
    "Low Value":          "Use low-cost automated campaigns.",
    "Needs Attention":    "Monitor behaviour and test targeted messaging.",
}


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply segment rules in priority order.
    Each customer receives only the FIRST matching segment.
    """
    df["rfm_segment"]        = ""
    df["recommended_action"] = ""

    unassigned = pd.Series(True, index=df.index)

    for seg_name, condition_fn in SEGMENT_RULES:
        mask = condition_fn(df) & unassigned
        df.loc[mask, "rfm_segment"]        = seg_name
        df.loc[mask, "recommended_action"] = SEGMENT_ACTIONS[seg_name]
        unassigned = unassigned & ~mask
        print(f"  {seg_name:<22} : {mask.sum():>5,} customers")

    remaining = unassigned.sum()
    if remaining > 0:
        print(f"  WARNING: {remaining} customers not assigned — check rules.")

    return df


# ── Chart helpers ──────────────────────────────────────────────────────────────
def save_chart(fig, filename: str):
    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    print(f"  Chart saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n── Loading customer features ────────────────────")
    cf = pd.read_parquet(CF_PATH, engine="pyarrow")
    print(f"  Rows loaded          : {len(cf):,}")
    print(f"  Unique customers     : {cf['customer_id'].nunique():,}")
    print(f"  Columns available    : {list(cf.columns)}")

    # Work on a copy so source is unchanged
    df = cf.copy()

    # ── RFM base variables ────────────────────────────────────────────────────
    # R = recency_days   (lower  is better)
    # F = total_invoices (higher is better)
    # M = total_revenue  (higher is better)

    print("\n── Calculating RFM scores (1–5) ─────────────────")

    df["r_score"] = safe_qcut_score(df["recency_days"],   higher_is_better=False)
    df["f_score"] = safe_qcut_score(df["total_invoices"], higher_is_better=True)
    df["m_score"] = safe_qcut_score(df["total_revenue"],  higher_is_better=True)

    print(f"  r_score range  : {df['r_score'].min()} – {df['r_score'].max()}")
    print(f"  r_score dist   : {df['r_score'].value_counts().sort_index().to_dict()}")
    print(f"  f_score range  : {df['f_score'].min()} – {df['f_score'].max()}")
    print(f"  f_score dist   : {df['f_score'].value_counts().sort_index().to_dict()}")
    print(f"  m_score range  : {df['m_score'].min()} – {df['m_score'].max()}")
    print(f"  m_score dist   : {df['m_score'].value_counts().sort_index().to_dict()}")

    # ── Composite scores ──────────────────────────────────────────────────────
    df["rfm_score"]         = (df["r_score"].astype(str)
                                + df["f_score"].astype(str)
                                + df["m_score"].astype(str))
    df["rfm_total_score"]   = df["r_score"] + df["f_score"] + df["m_score"]
    df["rfm_average_score"] = df["rfm_total_score"] / 3

    print(f"\n  rfm_total_score range : {df['rfm_total_score'].min()} – {df['rfm_total_score'].max()}")

    # ── Segment assignment ────────────────────────────────────────────────────
    print("\n── Assigning business segments ──────────────────")
    df = assign_segments(df)

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n── Running assertions ───────────────────────────")
    assert df["customer_id"].is_unique,                     "FAIL: customer_id not unique"
    assert len(df) == len(cf),                              "FAIL: row count mismatch"
    assert df["r_score"].isnull().sum() == 0,               "FAIL: null r_score"
    assert df["f_score"].isnull().sum() == 0,               "FAIL: null f_score"
    assert df["m_score"].isnull().sum() == 0,               "FAIL: null m_score"
    assert df["r_score"].between(1, 5).all(),               "FAIL: r_score out of range"
    assert df["f_score"].between(1, 5).all(),               "FAIL: f_score out of range"
    assert df["m_score"].between(1, 5).all(),               "FAIL: m_score out of range"
    assert (df["rfm_segment"] != "").all(),                 "FAIL: empty rfm_segment"
    assert df["rfm_segment"].notna().all(),                 "FAIL: null rfm_segment"
    assert df["recommended_action"].notna().all(),          "FAIL: null recommended_action"

    rev_rfm = round(df["total_revenue"].sum(), 2)
    rev_cf  = round(cf["total_revenue"].sum(), 2)
    assert abs(rev_rfm - rev_cf) < 0.02, \
        f"FAIL: revenue mismatch  rfm={rev_rfm}  features={rev_cf}"

    print("  All assertions passed ✓")

    # ── Segment summary tables ────────────────────────────────────────────────
    print("\n── Building segment summary tables ──────────────")
    total_customers = len(df)
    total_revenue   = df["total_revenue"].sum()

    seg_counts = (
        df.groupby("rfm_segment", sort=False)
        .agg(customer_count=("customer_id", "count"))
        .reset_index()
        .sort_values("customer_count", ascending=False)
    )
    seg_counts["customer_percent"] = (
        seg_counts["customer_count"] / total_customers * 100
    ).round(2)

    seg_revenue = (
        df.groupby("rfm_segment", sort=False)
        .agg(
            total_revenue        = ("total_revenue",   "sum"),
            avg_revenue_per_cust = ("total_revenue",   "mean"),
            avg_recency_days     = ("recency_days",    "mean"),
            avg_total_invoices   = ("total_invoices",  "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    seg_revenue["revenue_percent"] = (
        seg_revenue["total_revenue"] / total_revenue * 100
    ).round(2)
    seg_revenue["total_revenue"]        = seg_revenue["total_revenue"].round(2)
    seg_revenue["avg_revenue_per_cust"] = seg_revenue["avg_revenue_per_cust"].round(2)
    seg_revenue["avg_recency_days"]     = seg_revenue["avg_recency_days"].round(1)
    seg_revenue["avg_total_invoices"]   = seg_revenue["avg_total_invoices"].round(2)

    # Rename for output spec
    seg_revenue = seg_revenue.rename(columns={
        "avg_revenue_per_cust": "avg_revenue_per_customer"
    })

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n── Saving charts ────────────────────────────────")

    # Segment counts bar chart
    fig_counts = px.bar(
        seg_counts.sort_values("customer_count"),
        x="customer_count", y="rfm_segment",
        orientation="h",
        title="Customer Count by RFM Segment",
        labels={"customer_count": "Number of Customers", "rfm_segment": "Segment"},
        color="customer_count",
        color_continuous_scale="Blues",
        text="customer_count",
    )
    fig_counts.update_traces(textposition="outside")
    fig_counts.update_layout(yaxis={"categoryorder": "total ascending"})
    save_chart(fig_counts, "rfm_segment_counts.html")

    # Segment revenue bar chart
    fig_rev = px.bar(
        seg_revenue.sort_values("total_revenue"),
        x="total_revenue", y="rfm_segment",
        orientation="h",
        title="Total Revenue by RFM Segment",
        labels={"total_revenue": "Revenue (£)", "rfm_segment": "Segment"},
        color="total_revenue",
        color_continuous_scale="Greens",
        text="revenue_percent",
    )
    fig_rev.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_rev.update_layout(yaxis={"categoryorder": "total ascending"})
    save_chart(fig_rev, "rfm_segment_revenue.html")

    # Recency distribution
    fig_rec = px.histogram(
        df, x="recency_days", color="rfm_segment",
        nbins=60, barmode="overlay", opacity=0.6,
        title="Recency Distribution by Segment",
        labels={"recency_days": "Recency (days)"},
    )
    save_chart(fig_rec, "rfm_recency_distribution.html")

    # Frequency distribution (cap at 95th pct for readability)
    freq_cap = int(df["total_invoices"].quantile(0.95))
    fig_freq = px.histogram(
        df[df["total_invoices"] <= freq_cap],
        x="total_invoices", color="rfm_segment",
        nbins=50, barmode="overlay", opacity=0.6,
        title=f"Frequency Distribution by Segment (capped at {freq_cap} invoices)",
        labels={"total_invoices": "Number of Invoices"},
    )
    save_chart(fig_freq, "rfm_frequency_distribution.html")

    # Monetary distribution (cap at 95th pct)
    mon_cap = df["total_revenue"].quantile(0.95)
    fig_mon = px.histogram(
        df[df["total_revenue"] <= mon_cap],
        x="total_revenue", color="rfm_segment",
        nbins=60, barmode="overlay", opacity=0.6,
        title=f"Monetary Distribution by Segment (capped at £{mon_cap:,.0f})",
        labels={"total_revenue": "Total Revenue (£)"},
    )
    save_chart(fig_mon, "rfm_monetary_distribution.html")

    # ── Save CSV / Parquet outputs ────────────────────────────────────────────
    print("\n── Saving outputs ────────────────────────────────")
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False, engine="pyarrow")
    seg_counts.to_csv(REPORTS_DIR / "rfm_segment_counts.csv", index=False)
    seg_revenue.to_csv(REPORTS_DIR / "rfm_segment_revenue.csv", index=False)
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Saved: reports/rfm_segment_counts.csv")
    print(f"  Saved: reports/rfm_segment_revenue.csv")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n── Saving summary report ────────────────────────")
    lines = []

    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 60)
    log("  STEP 7 — RFM SEGMENTATION SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 60)

    log(f"\n  Input customer rows          : {len(cf):,}")
    log(f"  Output RFM rows              : {len(df):,}")
    log(f"  RFM columns added            : r_score, f_score, m_score,")
    log(f"                                 rfm_score, rfm_total_score,")
    log(f"                                 rfm_average_score, rfm_segment,")
    log(f"                                 recommended_action  (8 new columns)")
    log(f"\n  Revenue validation:")
    log(f"    RFM total revenue          : £{rev_rfm:,.2f}")
    log(f"    Features total revenue     : £{rev_cf:,.2f}")
    log(f"    Difference                 : £{abs(rev_rfm - rev_cf):.2f}  ✓")

    log(f"\n{'─'*60}")
    log(f"  SEGMENT COUNT TABLE")
    log(f"{'─'*60}")
    log(f"  {'Segment':<22} {'Customers':>10} {'%':>8}")
    log(f"  {'-'*42}")
    for _, row in seg_counts.sort_values("customer_count", ascending=False).iterrows():
        log(f"  {row['rfm_segment']:<22} {int(row['customer_count']):>10,} "
            f"{row['customer_percent']:>7.1f}%")

    log(f"\n{'─'*60}")
    log(f"  SEGMENT REVENUE TABLE")
    log(f"{'─'*60}")
    log(f"  {'Segment':<22} {'Revenue':>14} {'Rev%':>7} {'Avg Rev':>10} "
        f"{'AvgRec':>8} {'AvgInv':>8}")
    log(f"  {'-'*74}")
    for _, row in seg_revenue.sort_values("total_revenue", ascending=False).iterrows():
        log(f"  {row['rfm_segment']:<22} "
            f"£{row['total_revenue']:>12,.2f} "
            f"{row['revenue_percent']:>6.1f}% "
            f"£{row['avg_revenue_per_customer']:>8,.2f} "
            f"{row['avg_recency_days']:>7.0f}d "
            f"{row['avg_total_invoices']:>7.1f}")

    # Champions stats
    champ      = seg_counts[seg_counts["rfm_segment"] == "Champions"]
    champ_n    = int(champ["customer_count"].values[0]) if len(champ) else 0
    champ_pct  = champ["customer_percent"].values[0] if len(champ) else 0.0
    champ_rev  = seg_revenue[seg_revenue["rfm_segment"] == "Champions"]["total_revenue"]
    champ_rev_val = float(champ_rev.values[0]) if len(champ_rev) else 0.0
    champ_rev_pct = champ_rev_val / total_revenue * 100

    # At Risk + Cannot Lose
    at_risk_rows = seg_counts[seg_counts["rfm_segment"].isin(["At Risk", "Cannot Lose"])]
    at_risk_n    = int(at_risk_rows["customer_count"].sum())
    at_risk_pct  = at_risk_rows["customer_percent"].sum()

    log(f"\n{'─'*60}")
    log(f"  KEY BUSINESS INSIGHTS")
    log(f"{'─'*60}")
    top_seg_count  = seg_counts.iloc[0]
    top_seg_rev    = seg_revenue.iloc[0]
    log(f"  Top segment by count   : {top_seg_count['rfm_segment']}"
        f" ({int(top_seg_count['customer_count']):,} customers,"
        f" {top_seg_count['customer_percent']:.1f}%)")
    log(f"  Top segment by revenue : {top_seg_rev['rfm_segment']}"
        f" (£{top_seg_rev['total_revenue']:,.2f},"
        f" {top_seg_rev['revenue_percent']:.1f}%)")
    log(f"\n  Champions              : {champ_n:,} customers ({champ_pct:.1f}%)")
    log(f"  Champions revenue      : £{champ_rev_val:,.2f} ({champ_rev_pct:.1f}% of total)")
    log(f"\n  At Risk + Cannot Lose  : {at_risk_n:,} customers ({at_risk_pct:.1f}%)")

    log(f"\n{'─'*60}")
    log(f"  BUSINESS INTERPRETATION — TOP 3 SEGMENTS")
    log(f"{'─'*60}")
    top3 = seg_revenue.sort_values("total_revenue", ascending=False).head(3)
    interp = {
        "Champions":       "Best customers — high recency, frequency and spend. Protect and reward.",
        "Cannot Lose":     "Previously top customers now going quiet. Immediate retention priority.",
        "Loyal Customers": "Consistently engaged and valuable. Upsell and cross-sell opportunity.",
        "Big Spenders":    "High revenue per visit but infrequent. Increase visit frequency.",
        "At Risk":         "Active but drifting. Win-back campaigns needed urgently.",
        "Needs Attention": "Mixed profile. Requires deeper behavioural analysis.",
        "Hibernating":     "Long inactive. Low-cost reactivation; low expected conversion.",
        "Low Value":       "Minimal engagement and revenue. Automate low-cost outreach.",
        "Potential Loyalists": "Recent and growing. Nurture with personalised engagement.",
        "New Customers":   "Just arrived. Onboarding experience is critical now.",
    }
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        seg = row["rfm_segment"]
        log(f"  #{rank} {seg}")
        log(f"     Revenue : £{row['total_revenue']:,.2f} ({row['revenue_percent']:.1f}%)")
        log(f"     Avg Rev : £{row['avg_revenue_per_customer']:,.2f} per customer")
        log(f"     Insight : {interp.get(seg, 'Review segment characteristics.')}")

    with open(REPORTS_DIR / "rfm_segmentation_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved -> reports/rfm_segmentation_summary.txt")

    # ── Printed terminal summary ──────────────────────────────────────────────
    print("\n── Terminal summary ─────────────────────────────")
    print(f"  Input rows           : {len(cf):,}")
    print(f"  Unique customers     : {df['customer_id'].nunique():,}")
    print(f"  RFM rows created     : {len(df):,}")
    print(f"  Total revenue check  : £{rev_rfm:,.2f}  (matches source ✓)")
    print(f"  Segments created     : {df['rfm_segment'].nunique()}")
    print(f"\n  Segment count summary:")
    for _, row in seg_counts.sort_values("customer_count", ascending=False).iterrows():
        print(f"    {row['rfm_segment']:<22} {int(row['customer_count']):>5,}  "
              f"({row['customer_percent']:.1f}%)")
    print(f"\n  Output files:")
    for p in [OUT_CSV, OUT_PARQUET,
              REPORTS_DIR/"rfm_segment_counts.csv",
              REPORTS_DIR/"rfm_segment_revenue.csv",
              REPORTS_DIR/"rfm_segmentation_summary.txt"]:
        print(f"    {p}")

    print("\n" + "=" * 60)
    print("  STEP 7 RFM segmentation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run()
