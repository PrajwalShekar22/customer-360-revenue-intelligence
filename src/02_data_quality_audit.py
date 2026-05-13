"""
02_data_quality_audit.py
------------------------
STEP 3: Data quality audit — inspect only, do NOT clean.

Checks and reports:
  - Missing values per column
  - Duplicate rows
  - Negative / zero quantities
  - Negative / zero prices
  - Missing Customer ID
  - Cancelled invoices (C-prefix)
  - Unique customers, invoices, products, countries

Outputs:
  reports/data_quality_summary.csv

Run from project root:
    python src/02_data_quality_audit.py
"""

import pandas as pd
import os

RAW_PATH   = "data/raw/online_retail_II.xlsx"
REPORT_CSV = "reports/data_quality_summary.csv"


def run():
    os.makedirs("reports", exist_ok=True)

    print("Loading raw data (both sheets)...")
    df0 = pd.read_excel(RAW_PATH, sheet_name="Year 2009-2010", engine="openpyxl")
    df1 = pd.read_excel(RAW_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
    df  = pd.concat([df0, df1], ignore_index=True)
    print(f"  Total rows loaded: {len(df):,}")

    issues = {}

    # Missing values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues[f"null_{col}"] = null_count

    # Duplicates
    issues["duplicate_rows"] = df.duplicated().sum()

    # Quantity issues
    issues["quantity_negative"] = (df["Quantity"] < 0).sum()
    issues["quantity_zero"]     = (df["Quantity"] == 0).sum()

    # Price issues
    issues["price_negative"] = (df["Price"] < 0).sum()
    issues["price_zero"]     = (df["Price"] == 0).sum()

    # Missing Customer ID
    issues["missing_customer_id"] = df["Customer ID"].isnull().sum()

    # Cancellations
    issues["cancelled_invoices"] = df["Invoice"].astype(str).str.startswith("C").sum()

    # Uniqueness stats (not issues, but useful context)
    issues["unique_customers"] = df["Customer ID"].nunique()
    issues["unique_invoices"]  = df["Invoice"].nunique()
    issues["unique_products"]  = df["StockCode"].nunique()
    issues["unique_countries"] = df["Country"].nunique()
    issues["total_rows"]       = len(df)
    issues["date_min"]         = str(df["InvoiceDate"].min())
    issues["date_max"]         = str(df["InvoiceDate"].max())

    # Print summary
    print("\n=== DATA QUALITY AUDIT SUMMARY ===")
    for k, v in issues.items():
        print(f"  {k:<30}: {v}")

    # Save as CSV
    summary_df = pd.DataFrame(
        list(issues.items()), columns=["check", "value"]
    )
    summary_df.to_csv(REPORT_CSV, index=False)
    print(f"\nAudit saved -> {REPORT_CSV}")


if __name__ == "__main__":
    run()
