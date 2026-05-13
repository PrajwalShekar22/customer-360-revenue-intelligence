"""
Step 11 — Customer Health Score
Customer 360 Revenue Intelligence Platform

Combines customer_features + rfm_segments + churn_predictions into
one customer-level Customer 360 table that powers the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED   = Path("data/processed")
REPORTS     = Path("reports")
FIGURES     = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

CF_PATH     = PROCESSED / "customer_features.parquet"
RFM_PATH    = PROCESSED / "rfm_segments.parquet"
CP_PATH     = PROCESSED / "churn_predictions.parquet"

OUT_PARQUET = PROCESSED / "customer_360.parquet"
OUT_CSV     = PROCESSED / "customer_360.csv"
SUMMARY_TXT = REPORTS   / "customer_health_score_summary.txt"
TIER_COUNTS = REPORTS   / "customer_health_tier_counts.csv"
TIER_REV    = REPORTS   / "customer_health_tier_revenue.csv"
ACTION_PLAN = REPORTS   / "customer_action_plan.csv"

HEALTH_TIER_ORDER  = ["Excellent", "Healthy", "Watchlist", "At Risk", "Critical"]
ACTION_PRIO_ORDER  = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5"]

# ── Load inputs ───────────────────────────────────────────────────────────────
print("── Loading inputs ────────────────────────────────────────────────────────")

cf  = pd.read_parquet(CF_PATH)
rfm = pd.read_parquet(RFM_PATH)
cp  = pd.read_parquet(CP_PATH)

print(f"  customer_features  : {cf.shape[0]:,} rows × {cf.shape[1]} cols")
print(f"  rfm_segments       : {rfm.shape[0]:,} rows × {rfm.shape[1]} cols")
print(f"  churn_predictions  : {cp.shape[0]:,} rows × {cp.shape[1]} cols")

# ── Validate uniqueness ───────────────────────────────────────────────────────
assert cf["customer_id"].is_unique,  "customer_features has duplicate customer_ids"
assert rfm["customer_id"].is_unique, "rfm_segments has duplicate customer_ids"
assert cp["customer_id"].is_unique,  "churn_predictions has duplicate customer_ids"
print("  ID uniqueness      : OK (all three inputs)")

# ── Select required columns ───────────────────────────────────────────────────
CF_COLS = [
    "customer_id", "country_mode", "first_purchase_date", "last_purchase_date",
    "customer_tenure_days", "recency_days", "total_invoices", "total_line_items",
    "total_quantity", "unique_products", "unique_purchase_days",
    "avg_days_between_purchases", "total_revenue", "average_invoice_value",
    "average_line_revenue", "average_unit_price", "max_invoice_value",
    "min_invoice_value", "is_repeat_buyer", "purchase_frequency_per_month",
    "quantity_per_invoice", "products_per_invoice", "total_returns",
    "return_quantity_abs", "has_returned", "return_rate",
]

RFM_COLS = [
    "customer_id", "r_score", "f_score", "m_score",
    "rfm_score", "rfm_total_score", "rfm_average_score",
    "rfm_segment", "recommended_action",
]

CP_COLS = [
    "customer_id", "actual_churned", "predicted_churned",
    "churn_probability", "risk_tier",
]

cf_sel  = cf[CF_COLS].copy()
rfm_sel = rfm[RFM_COLS].copy()
cp_sel  = cp[CP_COLS].copy()

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\n── Merging data ──────────────────────────────────────────────────────────")

# RFM is a superset of customer_features so merge is 1:1
df = cf_sel.merge(rfm_sel, on="customer_id", how="left")
assert len(df) == len(cf_sel), "RFM merge changed row count"
assert df["rfm_segment"].notna().all(), "rfm_segment has nulls after merge"
print(f"  After CF + RFM merge : {len(df):,} rows (expected {len(cf_sel):,})")

# Left join churn predictions
df = df.merge(cp_sel, on="customer_id", how="left")
assert len(df) == len(cf_sel), "Churn predictions merge changed row count"
print(f"  After churn merge    : {len(df):,} rows")

# Rename risk_tier → churn_risk_tier
df.rename(columns={"risk_tier": "churn_risk_tier"}, inplace=True)

# ── Fill unscored customers ────────────────────────────────────────────────────
df["model_scored"] = df["churn_probability"].notna().astype(int)
df.loc[df["model_scored"] == 0, "churn_risk_tier"] = "Not Scored"

scored    = (df["model_scored"] == 1).sum()
not_scored = (df["model_scored"] == 0).sum()
print(f"  Model scored         : {scored:,}")
print(f"  Not Scored           : {not_scored:,}")

# ── Compute health score components ───────────────────────────────────────────
print("\n── Computing customer health score ───────────────────────────────────────")

# 1. RFM component — 40 pts  (rfm_total_score max = 15)
df["rfm_component"] = (df["rfm_total_score"] / 15) * 40

# 2. Churn component — 30 pts
df["churn_component"] = np.where(
    df["model_scored"] == 1,
    (1 - df["churn_probability"]) * 30,
    15.0,   # neutral midpoint for unscored customers
)

# 3. Revenue component — 20 pts  (percentile rank of total_revenue)
df["revenue_pct"] = df["total_revenue"].rank(pct=True, method="average")
df["revenue_component"] = df["revenue_pct"] * 20

# 4. Engagement component — 10 pts
df["freq_pct"] = df["purchase_frequency_per_month"].rank(pct=True, method="average")
df["engagement_component"] = (
    (df["is_repeat_buyer"] * 0.5) + (df["freq_pct"] * 0.5)
) * 10

# Final health score
df["customer_health_score"] = (
    df["rfm_component"] +
    df["churn_component"] +
    df["revenue_component"] +
    df["engagement_component"]
).round(2).clip(0, 100)

print(f"  Health score min     : {df['customer_health_score'].min():.2f}")
print(f"  Health score max     : {df['customer_health_score'].max():.2f}")
print(f"  Health score mean    : {df['customer_health_score'].mean():.2f}")
print(f"  Health score median  : {df['customer_health_score'].median():.2f}")

# Drop intermediate columns
df.drop(columns=["revenue_pct", "freq_pct",
                  "rfm_component", "churn_component",
                  "revenue_component", "engagement_component"], inplace=True)

# ── Health tier ────────────────────────────────────────────────────────────────
def assign_health_tier(score):
    if score >= 80:  return "Excellent"
    if score >= 60:  return "Healthy"
    if score >= 40:  return "Watchlist"
    if score >= 20:  return "At Risk"
    return "Critical"

df["health_tier"] = df["customer_health_score"].apply(assign_health_tier)
print("\n  Health tier distribution:")
tier_counts_raw = df["health_tier"].value_counts()
for tier in HEALTH_TIER_ORDER:
    n = tier_counts_raw.get(tier, 0)
    print(f"    {tier:<12} : {n:,} ({n/len(df)*100:.1f}%)")

# ── Customer value tier ────────────────────────────────────────────────────────
rev_pct_rank = df["total_revenue"].rank(pct=True, method="average")

def assign_value_tier(pct):
    if pct >= 0.90: return "Top 10% Value"
    if pct >= 0.70: return "High Value"
    if pct >= 0.30: return "Mid Value"
    return "Low Value"

df["customer_value_tier"] = rev_pct_rank.apply(assign_value_tier)

# ── Action priority ────────────────────────────────────────────────────────────
rev_top20 = df["total_revenue"] >= df["total_revenue"].quantile(0.80)

p1_mask = (
    df["health_tier"].isin(["Critical", "At Risk"]) |
    (df["churn_risk_tier"] == "Critical Risk")
)
p2_mask = (
    ~p1_mask & (
        df["rfm_segment"].isin(["Cannot Lose", "At Risk"]) |
        (rev_top20 & df["churn_risk_tier"].isin(["High Risk", "Critical Risk"]))
    )
)
p3_mask = (
    ~p1_mask & ~p2_mask & (
        df["rfm_segment"].isin(["Potential Loyalists", "New Customers"]) |
        (df["health_tier"] == "Watchlist")
    )
)
p4_mask = (
    ~p1_mask & ~p2_mask & ~p3_mask & (
        df["rfm_segment"].isin(["Champions", "Loyal Customers", "Big Spenders"]) &
        df["health_tier"].isin(["Excellent", "Healthy"])
    )
)

df["action_priority"] = "Priority 5"
df.loc[p4_mask, "action_priority"] = "Priority 4"
df.loc[p3_mask, "action_priority"] = "Priority 3"
df.loc[p2_mask, "action_priority"] = "Priority 2"
df.loc[p1_mask, "action_priority"] = "Priority 1"

print("\n  Action priority distribution:")
prio_counts_raw = df["action_priority"].value_counts()
for prio in ACTION_PRIO_ORDER:
    n = prio_counts_raw.get(prio, 0)
    print(f"    {prio} : {n:,} ({n/len(df)*100:.1f}%)")

# ── Final recommended action ───────────────────────────────────────────────────
ACTION_MAP = {
    "Priority 1": "Immediate win-back outreach with personalized offer.",
    "Priority 2": "Assign retention priority and protect high-value relationship.",
    "Priority 3": "Encourage repeat purchase with targeted product recommendations.",
    "Priority 4": "Reward loyalty and offer premium cross-sell opportunities.",
    "Priority 5": "Use automated email nurture and monitor future activity.",
}
df["final_recommended_action"] = df["action_priority"].map(ACTION_MAP)
df.loc[df["model_scored"] == 0, "final_recommended_action"] = (
    df.loc[df["model_scored"] == 0, "final_recommended_action"]
    .fillna("Monitor customer until enough history exists for churn scoring.")
)

# ── Retention target & VIP flags ──────────────────────────────────────────────
df["retention_target_flag"] = (
    df["churn_risk_tier"].isin(["High Risk", "Critical Risk"]) |
    df["health_tier"].isin(["At Risk", "Critical"]) |
    df["rfm_segment"].isin(["Cannot Lose", "At Risk"])
).astype(int)

df["vip_flag"] = (
    df["rfm_segment"].isin(["Champions", "Loyal Customers", "Big Spenders"]) &
    df["customer_value_tier"].isin(["Top 10% Value", "High Value"])
).astype(int)

retention_count = df["retention_target_flag"].sum()
vip_count       = df["vip_flag"].sum()
print(f"\n  Retention targets    : {retention_count:,}")
print(f"  VIP customers        : {vip_count:,}")

# ── Revenue validation ────────────────────────────────────────────────────────
rev_c360 = round(df["total_revenue"].sum(), 2)
rev_cf   = round(cf["total_revenue"].sum(), 2)
print(f"\n  Revenue customer_360 : £{rev_c360:,.2f}")
print(f"  Revenue customer_feat: £{rev_cf:,.2f}")
print(f"  Difference           : £{abs(rev_c360 - rev_cf):,.2f}")
assert abs(rev_c360 - rev_cf) < 1.0, "Revenue mismatch > £1.00"

# ── Save Customer 360 ─────────────────────────────────────────────────────────
print("\n── Saving Customer 360 table ─────────────────────────────────────────────")
df.to_parquet(OUT_PARQUET, index=False, engine="pyarrow")
df.to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1024:.1f} KB)")
print(f"  Saved: {OUT_CSV}  ({OUT_CSV.stat().st_size/1024:.1f} KB)")

# ── Report: health tier counts ────────────────────────────────────────────────
tier_df = (
    df.groupby("health_tier", observed=False)
    .size()
    .reindex(HEALTH_TIER_ORDER, fill_value=0)
    .reset_index()
)
tier_df.columns = ["health_tier", "customer_count"]
tier_df["customer_percent"] = (tier_df["customer_count"] / len(df) * 100).round(2)
tier_df.to_csv(TIER_COUNTS, index=False)
print(f"  Saved: {TIER_COUNTS}")

# ── Report: health tier revenue ───────────────────────────────────────────────
tier_rev_df = (
    df.groupby("health_tier", observed=False)
    .agg(
        total_revenue         = ("total_revenue", "sum"),
        customer_count        = ("customer_id",   "count"),
        avg_churn_probability = ("churn_probability", "mean"),
        retention_target_count= ("retention_target_flag", "sum"),
        vip_count             = ("vip_flag", "sum"),
    )
    .reindex(HEALTH_TIER_ORDER)
    .reset_index()
)
tier_rev_df["revenue_percent"]         = (tier_rev_df["total_revenue"] / rev_c360 * 100).round(2)
tier_rev_df["avg_revenue_per_customer"] = (tier_rev_df["total_revenue"] / tier_rev_df["customer_count"]).round(2)
tier_rev_df["avg_churn_probability"]    = tier_rev_df["avg_churn_probability"].round(4)
tier_rev_df["total_revenue"]            = tier_rev_df["total_revenue"].round(2)
tier_rev_df = tier_rev_df[[
    "health_tier", "total_revenue", "revenue_percent",
    "avg_revenue_per_customer", "avg_churn_probability",
    "retention_target_count", "vip_count",
]]
tier_rev_df.to_csv(TIER_REV, index=False)
print(f"  Saved: {TIER_REV}")

# ── Report: action plan ───────────────────────────────────────────────────────
STRATEGY_MAP = {
    "Priority 1": "Immediate win-back outreach. Personal calls + discount codes for Critical/At-Risk customers.",
    "Priority 2": "Retention priority. Protect high-value / Cannot Lose relationships with dedicated CSM touch.",
    "Priority 3": "Growth nurture. Drive repeat purchase via targeted recommendations and onboarding flows.",
    "Priority 4": "Loyalty & upsell. Reward Champions with early access, premium bundles, and referral programs.",
    "Priority 5": "Automated nurture. Low-cost email drip; monitor for re-engagement signals.",
}

action_df = (
    df.groupby("action_priority", observed=False)
    .agg(
        customer_count        = ("customer_id",        "count"),
        total_revenue         = ("total_revenue",      "sum"),
        avg_churn_probability = ("churn_probability",  "mean"),
    )
    .reindex(ACTION_PRIO_ORDER)
    .reset_index()
)
action_df["avg_churn_probability"] = action_df["avg_churn_probability"].round(4)
action_df["total_revenue"]         = action_df["total_revenue"].round(2)
action_df["recommended_strategy"]  = action_df["action_priority"].map(STRATEGY_MAP)
action_df.to_csv(ACTION_PLAN, index=False)
print(f"  Saved: {ACTION_PLAN}")

# ── Charts ────────────────────────────────────────────────────────────────────
print("\n── Generating charts ─────────────────────────────────────────────────────")

PLOTLY_TEMPLATE = "plotly_white"
TIER_COLORS = {
    "Excellent": "#2ecc71", "Healthy": "#27ae60",
    "Watchlist": "#f39c12", "At Risk": "#e67e22", "Critical": "#e74c3c",
}

# Chart 1 — Health tier customer counts
fig1 = px.bar(
    tier_df, x="health_tier", y="customer_count",
    color="health_tier",
    color_discrete_map=TIER_COLORS,
    text="customer_count",
    category_orders={"health_tier": HEALTH_TIER_ORDER},
    title="Customer Health Tier Distribution",
    labels={"health_tier": "Health Tier", "customer_count": "Customers"},
    template=PLOTLY_TEMPLATE,
)
fig1.update_traces(textposition="outside")
fig1.update_layout(showlegend=False, xaxis_title="Health Tier", yaxis_title="Number of Customers")
path1 = FIGURES / "customer_health_tier_counts.html"
fig1.write_html(str(path1))
print(f"  Chart saved -> {path1}")

# Chart 2 — Health tier revenue
fig2 = px.bar(
    tier_rev_df, x="health_tier", y="total_revenue",
    color="health_tier",
    color_discrete_map=TIER_COLORS,
    text=tier_rev_df["revenue_percent"].apply(lambda x: f"{x:.1f}%"),
    category_orders={"health_tier": HEALTH_TIER_ORDER},
    title="Total Revenue by Health Tier",
    labels={"health_tier": "Health Tier", "total_revenue": "Total Revenue (£)"},
    template=PLOTLY_TEMPLATE,
)
fig2.update_traces(textposition="outside")
fig2.update_layout(showlegend=False)
path2 = FIGURES / "customer_health_tier_revenue.html"
fig2.write_html(str(path2))
print(f"  Chart saved -> {path2}")

# Chart 3 — Health score vs churn probability (scatter)
scatter_df = df[df["model_scored"] == 1].copy()
fig3 = px.scatter(
    scatter_df,
    x="churn_probability",
    y="customer_health_score",
    color="health_tier",
    color_discrete_map=TIER_COLORS,
    hover_data=["customer_id", "rfm_segment", "total_revenue"],
    title="Customer Health Score vs Churn Probability",
    labels={
        "churn_probability":    "Churn Probability",
        "customer_health_score": "Customer Health Score",
    },
    opacity=0.55,
    template=PLOTLY_TEMPLATE,
    category_orders={"health_tier": HEALTH_TIER_ORDER},
)
fig3.update_layout(legend_title_text="Health Tier")
path3 = FIGURES / "customer_health_vs_churn_probability.html"
fig3.write_html(str(path3))
print(f"  Chart saved -> {path3}")

# Chart 4 — Avg health score by RFM segment (box plot)
seg_order = [
    "Champions", "Loyal Customers", "Big Spenders",
    "Recent Customers", "Potential Loyalists", "New Customers",
    "Promising", "Need Attention", "At Risk", "Cannot Lose",
    "About To Sleep", "Hibernating", "Lost",
]
valid_segs = [s for s in seg_order if s in df["rfm_segment"].unique()]
fig4 = px.box(
    df[df["rfm_segment"].isin(valid_segs)],
    x="rfm_segment", y="customer_health_score",
    color="rfm_segment",
    category_orders={"rfm_segment": valid_segs},
    title="Customer Health Score by RFM Segment",
    labels={"rfm_segment": "RFM Segment", "customer_health_score": "Health Score"},
    template=PLOTLY_TEMPLATE,
)
fig4.update_layout(showlegend=False, xaxis_tickangle=-45)
path4 = FIGURES / "customer_health_by_rfm_segment.html"
fig4.write_html(str(path4))
print(f"  Chart saved -> {path4}")

# Chart 5 — Action priority matrix (bubble chart: revenue vs customer count)
fig5 = px.scatter(
    action_df,
    x="action_priority",
    y="customer_count",
    size="total_revenue",
    color="action_priority",
    text="action_priority",
    title="Customer Action Priority Matrix (bubble size = revenue at stake)",
    labels={"action_priority": "Action Priority", "customer_count": "Number of Customers"},
    template=PLOTLY_TEMPLATE,
    size_max=80,
)
fig5.update_traces(textposition="top center")
fig5.update_layout(showlegend=False)
path5 = FIGURES / "customer_action_priority_matrix.html"
fig5.write_html(str(path5))
print(f"  Chart saved -> {path5}")

# ── Build summary text ────────────────────────────────────────────────────────
print("\n── Writing summary report ────────────────────────────────────────────────")

top10_rev  = df.nlargest(10, "total_revenue")[
    ["customer_id", "rfm_segment", "total_revenue", "customer_health_score", "health_tier"]
]
top10_risk = df[df["model_scored"] == 1].nlargest(10, "churn_probability")[
    ["customer_id", "rfm_segment", "churn_probability", "churn_risk_tier", "health_tier"]
]

lines = []
lines.append("=" * 72)
lines.append("  STEP 11 — CUSTOMER HEALTH SCORE SUMMARY")
lines.append("  Customer 360 Revenue Intelligence Platform")
lines.append("=" * 72)

lines.append("\n  Input files:")
lines.append(f"    customer_features  : {len(cf):,} customers")
lines.append(f"    rfm_segments       : {len(rfm):,} customers")
lines.append(f"    churn_predictions  : {len(cp):,} customers")

lines.append("\n  Final customer_360 table:")
lines.append(f"    Total rows         : {len(df):,}")
lines.append(f"    Total columns      : {df.shape[1]}")
lines.append(f"    Model scored       : {scored:,}")
lines.append(f"    Not Scored         : {not_scored:,}")

lines.append("\n  Health score statistics:")
lines.append(f"    Min                : {df['customer_health_score'].min():.2f}")
lines.append(f"    Max                : {df['customer_health_score'].max():.2f}")
lines.append(f"    Mean               : {df['customer_health_score'].mean():.2f}")
lines.append(f"    Median             : {df['customer_health_score'].median():.2f}")

lines.append("\n" + "-" * 72)
lines.append("  HEALTH TIER DISTRIBUTION")
lines.append("-" * 72)
lines.append(f"  {'Tier':<12}  {'Customers':>10}  {'%':>6}")
lines.append(f"  {'-'*12}  {'-'*10}  {'-'*6}")
for _, row in tier_df.iterrows():
    lines.append(f"  {row['health_tier']:<12}  {int(row['customer_count']):>10,}  {row['customer_percent']:>5.1f}%")

lines.append("\n" + "-" * 72)
lines.append("  REVENUE BY HEALTH TIER")
lines.append("-" * 72)
lines.append(f"  {'Tier':<12}  {'Revenue':>14}  {'Rev %':>6}  {'Avg Rev':>10}")
lines.append(f"  {'-'*12}  {'-'*14}  {'-'*6}  {'-'*10}")
for _, row in tier_rev_df.iterrows():
    lines.append(
        f"  {row['health_tier']:<12}  £{row['total_revenue']:>13,.2f}"
        f"  {row['revenue_percent']:>5.1f}%"
        f"  £{row['avg_revenue_per_customer']:>9,.2f}"
    )

lines.append("\n" + "-" * 72)
lines.append("  CUSTOMER VALUE TIER DISTRIBUTION")
lines.append("-" * 72)
val_order = ["Top 10% Value", "High Value", "Mid Value", "Low Value"]
vt = df["customer_value_tier"].value_counts().reindex(val_order, fill_value=0)
for tier, count in vt.items():
    lines.append(f"  {tier:<16} : {count:,} ({count/len(df)*100:.1f}%)")

lines.append("\n" + "-" * 72)
lines.append("  ACTION PRIORITY DISTRIBUTION")
lines.append("-" * 72)
for _, row in action_df.iterrows():
    lines.append(
        f"  {row['action_priority']}  |  {int(row['customer_count']):,} customers"
        f"  |  £{row['total_revenue']:,.2f}"
    )

lines.append("\n" + "-" * 72)
lines.append("  KEY FLAGS")
lines.append("-" * 72)
lines.append(f"  Retention target customers : {retention_count:,}")
lines.append(f"  VIP customers              : {vip_count:,}")

lines.append("\n" + "-" * 72)
lines.append("  TOP 10 CUSTOMERS BY TOTAL REVENUE")
lines.append("-" * 72)
for _, row in top10_rev.iterrows():
    lines.append(
        f"  Customer {int(row['customer_id'])}  |  {row['rfm_segment']:<20}"
        f"  |  £{row['total_revenue']:>10,.2f}  |  Health: {row['customer_health_score']:.1f} ({row['health_tier']})"
    )

lines.append("\n" + "-" * 72)
lines.append("  TOP 10 HIGHEST-RISK CUSTOMERS BY CHURN PROBABILITY")
lines.append("-" * 72)
for _, row in top10_risk.iterrows():
    lines.append(
        f"  Customer {int(row['customer_id'])}  |  {row['rfm_segment']:<20}"
        f"  |  Churn prob: {row['churn_probability']:.3f}  |  {row['churn_risk_tier']}"
    )

lines.append("\n" + "-" * 72)
lines.append("  BUSINESS INTERPRETATION")
lines.append("-" * 72)
lines.append(
    f"  • {(df['health_tier'].isin(['Critical','At Risk'])).sum():,} customers in Critical/At Risk tiers"
    f" need immediate retention action."
)
lines.append(
    f"  • {(df['health_tier'] == 'Watchlist').sum():,} Watchlist customers are growth opportunities —"
    " targeted nurture can move them to Healthy."
)
lines.append(
    f"  • {(df['health_tier'].isin(['Excellent','Healthy'])).sum():,} Excellent/Healthy customers"
    " are upsell/loyalty candidates — protect and expand."
)
lines.append(
    f"  • One-time buyers churn at 75.5%: prioritise second-purchase incentives"
    " for low-frequency customers."
)
lines.append(
    f"  • Champions ({(df['rfm_segment']=='Champions').sum():,} customers) drive the majority"
    " of revenue — monitor for any recency drift."
)

lines.append("\n" + "-" * 72)
lines.append("  LIMITATIONS")
lines.append("-" * 72)
lines.append(
    f"  1. Churn predictions exist for {scored:,} of {len(df):,} customers"
    f" (observation-window only)."
)
lines.append(
    f"  2. {not_scored:,} Not Scored customers are retained in Customer 360"
    " for dashboard completeness; churn_component uses neutral midpoint (15/30)."
)
lines.append(
    "  3. Customer health score is a business scoring framework, not a causal metric."
)
lines.append(
    "  4. Health tier boundaries (20/40/60/80) are rule-based; consider"
    " re-calibrating with domain expert input."
)
lines.append(
    "  5. Dataset covers Dec 2009 – Dec 2011. Scores must be recalculated"
    " on fresh data before production use."
)

lines.append("\n" + "=" * 72)
lines.append("  STEP 11 customer health score completed successfully.")
lines.append("=" * 72)

summary_text = "\n".join(lines)
SUMMARY_TXT.write_text(summary_text)
print(summary_text)
print(f"\n  Summary saved -> {SUMMARY_TXT}")

# ── Final assertions ──────────────────────────────────────────────────────────
print("\n── Final assertions ──────────────────────────────────────────────────────")

assert df["customer_id"].is_unique,                       "FAIL: customer_id not unique"
assert len(df) == len(cf),                                "FAIL: row count mismatch"
assert df["rfm_segment"].notna().all(),                   "FAIL: rfm_segment has nulls"
assert df["health_tier"].notna().all(),                   "FAIL: health_tier has nulls"
assert df["action_priority"].notna().all(),               "FAIL: action_priority has nulls"
assert df["final_recommended_action"].notna().all(),      "FAIL: final_recommended_action has nulls"
assert df["customer_health_score"].between(0, 100).all(), "FAIL: health score out of range"
assert set(df["model_scored"].unique()).issubset({0, 1}),  "FAIL: model_scored contains invalid values"
assert df.loc[df["model_scored"] == 0, "churn_probability"].isna().all(), \
    "FAIL: churn_probability is not null for unscored customers"
assert (df.loc[df["model_scored"] == 0, "churn_risk_tier"] == "Not Scored").all(), \
    "FAIL: churn_risk_tier is not 'Not Scored' for unscored customers"
assert abs(rev_c360 - rev_cf) < 1.0,                     "FAIL: revenue mismatch > £1.00"

required_files = [
    OUT_PARQUET, OUT_CSV, SUMMARY_TXT,
    TIER_COUNTS, TIER_REV, ACTION_PLAN,
]
for f in required_files:
    assert f.exists(), f"FAIL: missing file {f}"

print("  All assertions passed ✓")
print("\n================================================================")
print("  STEP 11 customer health score completed successfully.")
print("================================================================")
