"""
05_eda.py
---------
STEP 5: Exploratory Data Analysis

Loads clean_transactions.parquet and produces:
  - Printed metrics summary
  - 7 CSV report tables under reports/
  - 6 interactive Plotly HTML charts under reports/figures/
  - reports/eda_summary.txt

Run from project root:
    python src/05_eda.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Output paths ─────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/processed/clean_transactions.parquet")
REPORTS_DIR  = Path("reports")
FIGURES_DIR  = Path("reports/figures")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper: save plotly chart ────────────────────────────────────────────────
def save_chart(fig, filename: str):
    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    print(f"  Chart saved -> {path}")


# ── Helper: save CSV ─────────────────────────────────────────────────────────
def save_csv(df, filename: str):
    path = REPORTS_DIR / filename
    df.to_csv(str(path), index=False)
    print(f"  Table saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
def run():
    summary_lines = []

    def log(line: str = ""):
        print(line)
        summary_lines.append(line)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading data ─────────────────────────────────")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

    # ── 1-8: Dataset overview ─────────────────────────────────────────────────
    print("\n── Dataset overview ─────────────────────────────")
    log("=" * 55)
    log("  STEP 5 — EDA SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 55)
    log(f"\n  Shape                 : {df.shape[0]:,} rows x {df.shape[1]} columns")
    log(f"  Columns               : {list(df.columns)}")
    log(f"\n  Date range            : {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    log(f"  Unique customers      : {df['customer_id'].nunique():,}")
    log(f"  Unique invoices       : {df['invoice_id'].nunique():,}")
    log(f"  Unique stock codes    : {df['stock_code'].nunique():,}")
    log(f"  Unique countries      : {df['country'].nunique():,}")

    # ── 9-11: Revenue metrics ──────────────────────────────────────────────────
    print("\n── Revenue metrics ──────────────────────────────")
    total_revenue   = df["revenue"].sum()
    total_qty       = df["quantity"].sum()
    invoice_revenue = df.groupby("invoice_id")["revenue"].sum()
    avg_order_value = invoice_revenue.mean()
    median_order    = invoice_revenue.median()

    log(f"\n  Total revenue         : £{total_revenue:,.2f}")
    log(f"  Total quantity sold   : {total_qty:,}")
    log(f"  Total invoices        : {df['invoice_id'].nunique():,}")
    log(f"  Avg order value       : £{avg_order_value:,.2f}")
    log(f"  Median order value    : £{median_order:,.2f}")

    # ── 12: Monthly revenue trend ─────────────────────────────────────────────
    print("\n── Monthly revenue trend ────────────────────────")
    monthly_rev = (
        df.groupby("invoice_yearmonth")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_yearmonth": "year_month", "revenue": "total_revenue"})
        .sort_values("year_month")
    )
    save_csv(monthly_rev, "monthly_revenue.csv")

    fig_mr = px.line(
        monthly_rev,
        x="year_month",
        y="total_revenue",
        title="Monthly Revenue Trend",
        labels={"year_month": "Month", "total_revenue": "Revenue (£)"},
        markers=True,
    )
    fig_mr.update_layout(xaxis_tickangle=-45)
    save_chart(fig_mr, "monthly_revenue_trend.html")

    log(f"\n  Peak revenue month    : {monthly_rev.loc[monthly_rev['total_revenue'].idxmax(), 'year_month']}"
        f"  (£{monthly_rev['total_revenue'].max():,.2f})")
    log(f"  Lowest revenue month  : {monthly_rev.loc[monthly_rev['total_revenue'].idxmin(), 'year_month']}"
        f"  (£{monthly_rev['total_revenue'].min():,.2f})")

    # ── 13: Monthly unique customers ─────────────────────────────────────────
    print("\n── Monthly unique customers ─────────────────────")
    monthly_cust = (
        df.groupby("invoice_yearmonth")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"invoice_yearmonth": "year_month", "customer_id": "unique_customers"})
        .sort_values("year_month")
    )
    save_csv(monthly_cust, "monthly_customers.csv")

    fig_mc = px.bar(
        monthly_cust,
        x="year_month",
        y="unique_customers",
        title="Monthly Unique Active Customers",
        labels={"year_month": "Month", "unique_customers": "Unique Customers"},
    )
    fig_mc.update_layout(xaxis_tickangle=-45)
    save_chart(fig_mc, "monthly_customers_trend.html")

    log(f"\n  Peak active customers : {monthly_cust['unique_customers'].max():,}"
        f"  ({monthly_cust.loc[monthly_cust['unique_customers'].idxmax(), 'year_month']})")
    log(f"  Avg monthly customers : {monthly_cust['unique_customers'].mean():,.0f}")

    # ── 14: Top 10 countries by revenue ──────────────────────────────────────
    print("\n── Top 10 countries by revenue ──────────────────")
    top_countries = (
        df.groupby("country")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )
    save_csv(top_countries, "top_countries_by_revenue.csv")

    fig_tc = px.bar(
        top_countries,
        x="country",
        y="total_revenue",
        title="Top 10 Countries by Revenue",
        labels={"country": "Country", "total_revenue": "Revenue (£)"},
        color="total_revenue",
        color_continuous_scale="Blues",
    )
    save_chart(fig_tc, "top_countries_by_revenue.html")

    log("\n  Top 10 countries by revenue:")
    for _, row in top_countries.iterrows():
        log(f"    {row['country']:<25} £{row['total_revenue']:>12,.2f}")

    # ── 15: Top 10 products by revenue ───────────────────────────────────────
    print("\n── Top 10 products by revenue ───────────────────")
    top_products = (
        df.groupby(["stock_code", "description"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )
    save_csv(top_products, "top_products_by_revenue.csv")

    fig_tp = px.bar(
        top_products,
        x="total_revenue",
        y="description",
        orientation="h",
        title="Top 10 Products by Revenue",
        labels={"total_revenue": "Revenue (£)", "description": "Product"},
        color="total_revenue",
        color_continuous_scale="Greens",
    )
    fig_tp.update_layout(yaxis={"categoryorder": "total ascending"})
    save_chart(fig_tp, "top_products_by_revenue.html")

    log("\n  Top 10 products by revenue:")
    for _, row in top_products.iterrows():
        log(f"    [{row['stock_code']}] {row['description'][:40]:<40} £{row['total_revenue']:>10,.2f}")

    # ── 16: Top 10 customers by revenue ──────────────────────────────────────
    print("\n── Top 10 customers by revenue ──────────────────")
    top_customers = (
        df.groupby("customer_id")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )
    save_csv(top_customers, "top_customers_by_revenue.csv")

    log("\n  Top 10 customers by revenue:")
    for _, row in top_customers.iterrows():
        log(f"    Customer {int(row['customer_id']):<8} £{row['total_revenue']:>12,.2f}")

    # ── 17: Customer purchase frequency ──────────────────────────────────────
    print("\n── Customer purchase frequency ──────────────────")
    cust_freq = (
        df.groupby("customer_id")["invoice_id"]
        .nunique()
        .reset_index()
        .rename(columns={"invoice_id": "num_orders"})
    )
    save_csv(cust_freq, "customer_purchase_frequency.csv")

    # Clip the x-axis at 95th percentile so the chart is readable
    freq_cap = int(cust_freq["num_orders"].quantile(0.95))
    fig_cf = px.histogram(
        cust_freq[cust_freq["num_orders"] <= freq_cap],
        x="num_orders",
        nbins=50,
        title=f"Customer Purchase Frequency Distribution (capped at {freq_cap} orders)",
        labels={"num_orders": "Number of Orders", "count": "Number of Customers"},
    )
    save_chart(fig_cf, "customer_frequency_distribution.html")

    log(f"\n  Customer order frequency:")
    log(f"    Min orders per customer   : {cust_freq['num_orders'].min()}")
    log(f"    Max orders per customer   : {cust_freq['num_orders'].max()}")
    log(f"    Median orders per customer: {cust_freq['num_orders'].median():.1f}")
    log(f"    Mean orders per customer  : {cust_freq['num_orders'].mean():.2f}")

    one_time = (cust_freq["num_orders"] == 1).sum()
    repeat   = (cust_freq["num_orders"] > 1).sum()
    log(f"    One-time buyers           : {one_time:,} ({one_time/len(cust_freq)*100:.1f}%)")
    log(f"    Repeat buyers             : {repeat:,} ({repeat/len(cust_freq)*100:.1f}%)")

    # ── 18: Revenue distribution by customer ────────────────────────────────
    print("\n── Revenue distribution by customer ─────────────")
    cust_rev = (
        df.groupby("customer_id")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
    )
    save_csv(cust_rev, "customer_revenue_distribution.csv")

    # Cap at 95th percentile for readability
    rev_cap = cust_rev["total_revenue"].quantile(0.95)
    fig_cr = px.histogram(
        cust_rev[cust_rev["total_revenue"] <= rev_cap],
        x="total_revenue",
        nbins=60,
        title=f"Customer Revenue Distribution (capped at £{rev_cap:,.0f})",
        labels={"total_revenue": "Total Revenue (£)", "count": "Number of Customers"},
    )
    save_chart(fig_cr, "customer_revenue_distribution.html")

    log(f"\n  Customer revenue distribution:")
    log(f"    Min revenue per customer  : £{cust_rev['total_revenue'].min():,.2f}")
    log(f"    Max revenue per customer  : £{cust_rev['total_revenue'].max():,.2f}")
    log(f"    Median rev per customer   : £{cust_rev['total_revenue'].median():,.2f}")
    log(f"    Mean rev per customer     : £{cust_rev['total_revenue'].mean():,.2f}")

    # ── 19: Repeat purchase behavior ────────────────────────────────────────
    print("\n── Repeat purchase behavior ─────────────────────")
    total_customers = df["customer_id"].nunique()
    repeat_customers = (
        df.groupby("customer_id")["invoice_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    repeat_rate = repeat_customers / total_customers * 100

    # Revenue contribution from repeat vs one-time buyers
    one_time_ids    = cust_freq[cust_freq["num_orders"] == 1]["customer_id"]
    repeat_ids      = cust_freq[cust_freq["num_orders"] > 1]["customer_id"]
    rev_one_time    = df[df["customer_id"].isin(one_time_ids)]["revenue"].sum()
    rev_repeat      = df[df["customer_id"].isin(repeat_ids)]["revenue"].sum()

    log(f"\n  Repeat purchase behavior:")
    log(f"    Total customers           : {total_customers:,}")
    log(f"    Repeat buyers             : {repeat_customers:,} ({repeat_rate:.1f}%)")
    log(f"    One-time buyers           : {total_customers - repeat_customers:,} ({100 - repeat_rate:.1f}%)")
    log(f"    Revenue from repeat buyers: £{rev_repeat:,.2f} ({rev_repeat/total_revenue*100:.1f}% of total)")
    log(f"    Revenue from one-timers   : £{rev_one_time:,.2f} ({rev_one_time/total_revenue*100:.1f}% of total)")

    # ── 20: Save EDA summary text ────────────────────────────────────────────
    summary_path = REPORTS_DIR / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  EDA summary saved -> {summary_path}")

    # ── Final success message ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 EDA completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    run()
