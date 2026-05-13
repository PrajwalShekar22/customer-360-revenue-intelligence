"""
04_verify_clean_outputs.py
--------------------------
Verification script for Step 4 outputs.

Runs assertions against clean_transactions.csv and clean_transactions.parquet
to confirm the cleaning pipeline produced correct, consistent results.

Run from project root:
    python src/04_verify_clean_outputs.py
"""

import pandas as pd
import os

CSV_PATH     = "data/processed/clean_transactions.csv"
PARQUET_PATH = "data/processed/clean_transactions.parquet"
RETURNS_PATH = "data/processed/returns.csv"

PASS = "  PASS"
FAIL = "  FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {label}{' | ' + detail if detail else ''}")
    return condition


def run():
    all_passed = True

    # ── File existence ────────────────────────────────────
    print("\n── File existence ───────────────────────────────")
    for path in [CSV_PATH, PARQUET_PATH, RETURNS_PATH]:
        exists = os.path.isfile(path)
        size   = f"{os.path.getsize(path) / 1e6:.1f} MB" if exists else "missing"
        all_passed &= check(path, exists, size)

    # ── Load both formats ─────────────────────────────────
    print("\n── Format consistency (CSV vs Parquet) ──────────")
    df_csv = pd.read_csv(CSV_PATH)
    df_pq  = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

    all_passed &= check("Row count matches CSV vs Parquet",
                        len(df_csv) == len(df_pq),
                        f"CSV={len(df_csv):,}  Parquet={len(df_pq):,}")
    all_passed &= check("Column count matches",
                        df_csv.shape[1] == df_pq.shape[1],
                        f"{df_csv.shape[1]} cols")

    # Use CSV for all remaining checks
    df = df_csv
    df["invoice_id"] = df["invoice_id"].astype(str)
    ret = pd.read_csv(RETURNS_PATH)

    # ── Null checks ───────────────────────────────────────
    print("\n── Null checks ──────────────────────────────────")
    total_nulls = df.isnull().sum().sum()
    all_passed &= check("Zero nulls in clean_transactions", total_nulls == 0, f"found {total_nulls}")

    # ── Business rules ────────────────────────────────────
    print("\n── Business rule checks ─────────────────────────")
    all_passed &= check("No quantity <= 0",    (df["quantity"]   <= 0).sum() == 0)
    all_passed &= check("No unit_price <= 0",  (df["unit_price"] <= 0).sum() == 0)
    all_passed &= check("No revenue <= 0",     (df["revenue"]    <= 0).sum() == 0)
    all_passed &= check("No C-prefix invoices",(df["invoice_id"].str.startswith("C")).sum() == 0)
    all_passed &= check("No duplicate rows",   df.duplicated().sum() == 0)

    # ── Returns file ──────────────────────────────────────
    print("\n── Returns file checks ──────────────────────────")
    all_passed &= check("returns.csv exists and non-empty", len(ret) > 0, f"{len(ret):,} rows")
    all_passed &= check("All returns have C-prefix invoice",
                        ret["invoice_id"].astype(str).str.startswith("C").all())

    # ── Column presence ───────────────────────────────────
    print("\n── Required columns present ─────────────────────")
    required = ["invoice_id","stock_code","description","quantity","invoice_date",
                "unit_price","customer_id","country","revenue",
                "invoice_year","invoice_month","invoice_yearmonth","invoice_date_only"]
    for col in required:
        all_passed &= check(f"Column: {col}", col in df.columns)

    # ── Key metrics ───────────────────────────────────────
    print("\n── Key metrics (from data, not hardcoded) ───────")
    print(f"  Total rows           : {len(df):,}")
    print(f"  Unique customers     : {df['customer_id'].nunique():,}")
    print(f"  Unique invoices      : {df['invoice_id'].nunique():,}")
    print(f"  Unique stock codes   : {df['stock_code'].nunique():,}")
    print(f"  Unique countries     : {df['country'].nunique():,}")
    print(f"  Date range           : {df['invoice_date'].min()[:10]} -> {df['invoice_date'].max()[:10]}")
    print(f"  Total revenue        : £{df['revenue'].sum():,.2f}")
    print(f"  Avg order value      : £{df.groupby('invoice_id')['revenue'].sum().mean():,.2f}")
    print(f"  Returns rows         : {len(ret):,}")

    # ── Final verdict ─────────────────────────────────────
    print("\n" + "="*52)
    if all_passed:
        print("  ALL CHECKS PASSED — clean outputs are valid.")
    else:
        print("  ONE OR MORE CHECKS FAILED — review output above.")
    print("="*52)


if __name__ == "__main__":
    run()
