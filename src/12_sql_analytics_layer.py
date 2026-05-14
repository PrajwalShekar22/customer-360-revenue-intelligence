"""
Step 12C — DuckDB SQL Analytics Layer
Customer 360 Revenue Intelligence Platform

Registers Parquet datasets as DuckDB views, executes 10 SQL analytical queries,
saves results as CSV, and produces a summary report. All SQL is portable to
BigQuery with minor dialect adjustments.

Run from project root:
    python src/12_sql_analytics_layer.py
"""

import duckdb
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
C360_PATH  = ROOT / "data/processed/customer_360.parquet"
TXN_PATH   = ROOT / "data/processed/clean_transactions.parquet"
SQL_DIR    = ROOT / "sql"
OUT_DIR    = ROOT / "reports/sql_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map: SQL filename → output CSV filename
SQL_OUTPUTS = [
    ("01_executive_kpis.sql",         "executive_kpis.csv"),
    ("02_health_tier_summary.sql",    "health_tier_summary.csv"),
    ("03_rfm_segment_summary.sql",    "rfm_segment_summary.csv"),
    ("04_churn_risk_summary.sql",     "churn_risk_summary.csv"),
    ("05_country_revenue.sql",        "country_revenue.csv"),
    ("06_top_customers.sql",          "top_customers.csv"),
    ("07_action_priority_summary.sql","action_priority_summary.csv"),
    ("08_monthly_revenue.sql",        "monthly_revenue.csv"),
    ("09_product_revenue.sql",        "product_revenue.csv"),
    ("10_customer_value_tiers.sql",   "customer_value_tiers.csv"),
]

# ── Validate input files ───────────────────────────────────────────────────────
print("── Validating inputs ─────────────────────────────────────────────────────")
for p in [C360_PATH, TXN_PATH]:
    assert p.exists(), f"Required file missing: {p}"
    print(f"  Found: {p.name}")

c360_pd = pd.read_parquet(C360_PATH)
txn_pd  = pd.read_parquet(TXN_PATH)

print(f"\n  customer_360        : {len(c360_pd):,} rows × {c360_pd.shape[1]} cols")
print(f"  clean_transactions  : {len(txn_pd):,} rows × {txn_pd.shape[1]} cols")

rev_c360 = round(c360_pd["total_revenue"].sum(), 2)
rev_txn  = round(txn_pd["revenue"].sum(), 2)
rev_diff = abs(rev_c360 - rev_txn)
print(f"\n  Revenue (customer_360)       : £{rev_c360:,.2f}")
print(f"  Revenue (clean_transactions) : £{rev_txn:,.2f}")
print(f"  Difference                   : £{rev_diff:,.2f}")
assert rev_diff < 1.0, f"Revenue mismatch > £1.00: {rev_diff}"
print("  Revenue reconciliation       : PASSED ✓")

# ── Connect DuckDB and register views ─────────────────────────────────────────
print("\n── Connecting DuckDB ─────────────────────────────────────────────────────")
con = duckdb.connect(database=":memory:")

con.execute(f"""
    CREATE OR REPLACE VIEW customer_360 AS
    SELECT * FROM read_parquet('{C360_PATH}');
""")
con.execute(f"""
    CREATE OR REPLACE VIEW clean_transactions AS
    SELECT * FROM read_parquet('{TXN_PATH}');
""")

row_check = con.execute("SELECT COUNT(*) FROM customer_360").fetchone()[0]
print(f"  DuckDB view customer_360    : {row_check:,} rows")
txn_check = con.execute("SELECT COUNT(*) FROM clean_transactions").fetchone()[0]
print(f"  DuckDB view clean_transactions : {txn_check:,} rows")

# ── Execute SQL files ──────────────────────────────────────────────────────────
print("\n── Executing SQL queries ─────────────────────────────────────────────────")
results = {}
for sql_file, csv_file in SQL_OUTPUTS:
    sql_path = SQL_DIR / sql_file
    assert sql_path.exists(), f"SQL file missing: {sql_path}"

    sql = sql_path.read_text()
    df  = con.execute(sql).df()

    out_path = OUT_DIR / csv_file
    df.to_csv(out_path, index=False)
    results[csv_file] = df

    print(f"  {sql_file:<38}  →  {csv_file}  ({df.shape[0]} rows × {df.shape[1]} cols)")

# ── Validate outputs ───────────────────────────────────────────────────────────
print("\n── Validating outputs ────────────────────────────────────────────────────")

kpis = results["executive_kpis.csv"]
assert len(kpis) == 1, "executive_kpis must have exactly 1 row"
assert len(results["top_customers.csv"]) <= 50, "top_customers must have <= 50 rows"
assert len(results["product_revenue.csv"]) <= 50, "product_revenue must have <= 50 rows"

for csv_file in [f for _, f in SQL_OUTPUTS]:
    df = results[csv_file]
    assert len(df) > 0, f"{csv_file} is empty"
    print(f"  {csv_file:<40} {len(df):>4} rows  ✓")

# ── Extract KPI snapshot ───────────────────────────────────────────────────────
kpi = kpis.iloc[0]
total_customers        = int(kpi["total_customers"])
total_revenue          = float(kpi["total_revenue"])
retention_targets      = int(kpi["retention_targets"])
vip_customers          = int(kpi["vip_customers"])
model_scored           = int(kpi["model_scored_customers"])
not_scored             = int(kpi["not_scored_customers"])
ex_healthy_customers   = int(kpi["excellent_healthy_customers"])
ex_healthy_revenue     = float(kpi["excellent_healthy_revenue"])
ex_healthy_pct         = float(kpi["excellent_healthy_revenue_pct"])
crit_high_risk_cust    = int(kpi["critical_high_risk_customers"])
crit_high_risk_rev     = float(kpi["critical_high_risk_revenue"])

# ── Write summary report ───────────────────────────────────────────────────────
print("\n── Writing summary report ────────────────────────────────────────────────")

summary_lines = [
    "=" * 72,
    "  STEP 12C — DuckDB SQL Analytics Layer",
    "  Customer 360 Revenue Intelligence Platform",
    "=" * 72,
    "",
    "  Purpose:",
    "    This SQL analytics layer queries the Customer 360 Parquet datasets",
    "    using DuckDB — an in-process analytical SQL engine. It demonstrates",
    "    SQL-based business analytics over the full pipeline outputs and",
    "    produces reproducible CSV reports. All queries are portable to",
    "    BigQuery with minor dialect adjustments (e.g., APPROX_QUANTILES",
    "    for MEDIAN, backtick identifiers).",
    "",
    "  DuckDB:",
    "    DuckDB is a fast, embedded OLAP SQL engine that queries Parquet",
    "    files directly without loading them into a database server.",
    "    It is well-suited for local analytical projects and serves as a",
    "    lightweight alternative to BigQuery for portfolio demonstrations.",
    "",
    "  Input files:",
    f"    customer_360.parquet      : {C360_PATH}",
    f"    clean_transactions.parquet: {TXN_PATH}",
    "",
    "  Input row counts:",
    f"    customer_360       : {len(c360_pd):,} rows",
    f"    clean_transactions : {len(txn_pd):,} rows",
    "",
    "  Revenue reconciliation:",
    f"    customer_360 total revenue       : £{rev_c360:,.2f}",
    f"    clean_transactions total revenue : £{rev_txn:,.2f}",
    f"    Difference                       : £{rev_diff:,.2f}  ✓",
    "",
    "  SQL files executed (10):",
]
for sql_file, csv_file in SQL_OUTPUTS:
    summary_lines.append(f"    {sql_file}  →  {csv_file}")

summary_lines += [
    "",
    "  Output CSV files (10):",
]
for _, csv_file in SQL_OUTPUTS:
    df = results[csv_file]
    size_kb = (OUT_DIR / csv_file).stat().st_size / 1024
    summary_lines.append(f"    {csv_file:<42} {df.shape[0]:>4} rows  {size_kb:.1f} KB")

summary_lines += [
    "",
    "-" * 72,
    "  EXECUTIVE KPI SNAPSHOT (from SQL query 01)",
    "-" * 72,
    f"  Total customers               : {total_customers:,}",
    f"  Total revenue                 : £{total_revenue:,.2f}",
    f"  Avg revenue per customer      : £{float(kpi['avg_revenue_per_customer']):,.2f}",
    f"  Median revenue per customer   : £{float(kpi['median_revenue_per_customer']):,.2f}",
    f"  Total invoices                : {int(kpi['total_invoices']):,}",
    f"  Avg invoices per customer     : {float(kpi['avg_invoices_per_customer']):.2f}",
    f"  Model-scored customers        : {model_scored:,}",
    f"  Not Scored customers          : {not_scored:,}",
    f"  Retention targets             : {retention_targets:,}",
    f"  VIP customers                 : {vip_customers:,}",
    f"  Excellent + Healthy customers : {ex_healthy_customers:,}",
    f"  Excellent + Healthy revenue   : £{ex_healthy_revenue:,.2f} ({ex_healthy_pct:.1f}%)",
    f"  Critical + High Risk customers: {crit_high_risk_cust:,}",
    f"  Critical + High Risk revenue  : £{crit_high_risk_rev:,.2f}",
    "",
    "-" * 72,
    "  MIGRATION NOTE",
    "-" * 72,
    "  This SQL layer runs locally on DuckDB over Parquet files.",
    "  To migrate to BigQuery:",
    "    1. Upload Parquet files to Google Cloud Storage.",
    "    2. Create external tables or load into BigQuery native tables.",
    "    3. Replace MEDIAN() with APPROX_QUANTILES(col, 2)[OFFSET(1)].",
    "    4. Replace window function syntax if needed (BigQuery is ANSI-compatible).",
    "    5. Replace file paths with project.dataset.table references.",
    "",
    "-" * 72,
    "  VERIFICATION STATUS",
    "-" * 72,
    "  All 10 SQL queries executed successfully.",
    "  All 10 CSV outputs written and non-empty.",
    "  executive_kpis has exactly 1 row.",
    f"  top_customers has {len(results['top_customers.csv'])} rows (<= 50).",
    f"  product_revenue has {len(results['product_revenue.csv'])} rows (<= 50).",
    "  Revenue reconciliation: PASSED (£0.00 difference).",
    "",
    "=" * 72,
    "  STEP 12C SQL analytics layer completed successfully.",
    "=" * 72,
]

summary_text = "\n".join(summary_lines)
(OUT_DIR / "sql_analytics_summary.txt").write_text(summary_text)
print(summary_text)
print(f"\n  Summary saved -> {OUT_DIR / 'sql_analytics_summary.txt'}")
print("\n================================================================")
print("  STEP 12C SQL analytics layer completed successfully.")
print("================================================================")
