"""
03_clean_data.py
----------------
STEP 4: Data cleaning pipeline.

Cleaning decisions (each justified by the data quality audit):
  1. Combine both Excel sheets.
  2. Standardize column names to snake_case.
  3. Drop fully duplicate rows (34,335 found in raw data).
  4. Separate cancelled / return transactions (C-prefix invoices) → returns file.
  5. Drop rows with null customer_id (required for all customer-level analysis).
  6. Drop rows where quantity <= 0 (returns already separated; residuals are errors).
  7. Drop rows where unit_price <= 0 (5 negative + ~70 zero after nulls removed).
  8. Drop rows with null description (all overlap with null customer_id after step 5).
  9. Add derived columns: total_price, invoice_year, invoice_month, invoice_yearmonth.
  10. Cast customer_id to int.

Outputs:
  data/processed/clean_transactions.csv
  data/processed/clean_transactions.parquet
  data/processed/returns.csv

Run from project root:
    python src/03_clean_data.py
"""

import pandas as pd
import os

RAW_PATH        = "data/raw/online_retail_II.xlsx"
CLEAN_CSV       = "data/processed/clean_transactions.csv"
CLEAN_PARQUET   = "data/processed/clean_transactions.parquet"
RETURNS_OUT     = "data/processed/returns.csv"
PROCESSED_DIR   = "data/processed"


def load_raw(path: str) -> pd.DataFrame:
    print("Loading Year 2009-2010...")
    df0 = pd.read_excel(path, sheet_name="Year 2009-2010", engine="openpyxl")
    print("Loading Year 2010-2011...")
    df1 = pd.read_excel(path, sheet_name="Year 2010-2011", engine="openpyxl")
    df  = pd.concat([df0, df1], ignore_index=True)
    print(f"  Combined rows: {len(df):,}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Invoice":     "invoice_id",
        "StockCode":   "stock_code",
        "Description": "description",
        "Quantity":    "quantity",
        "InvoiceDate": "invoice_date",
        "Price":       "unit_price",
        "Customer ID": "customer_id",
        "Country":     "country",
    })


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"  [3] Dropped duplicates    : {before - len(df):,} rows")
    return df


def separate_returns(df: pd.DataFrame):
    """C-prefix invoices are cancellations. Keep separately, never in sales."""
    mask    = df["invoice_id"].astype(str).str.startswith("C")
    returns = df[mask].copy()
    sales   = df[~mask].copy()
    print(f"  [4] Returns separated     : {len(returns):,} rows -> returns.csv")
    print(f"      Sales rows remaining  : {len(sales):,}")
    return sales, returns


def drop_missing_customer(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["customer_id"])
    print(f"  [5] Dropped null cust_id  : {before - len(df):,} rows")
    return df


def drop_bad_quantity(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["quantity"] > 0]
    print(f"  [6] Dropped qty <= 0      : {before - len(df):,} rows")
    return df


def drop_bad_price(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["unit_price"] > 0]
    print(f"  [7] Dropped price <= 0    : {before - len(df):,} rows")
    return df


def drop_missing_description(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["description"])
    print(f"  [8] Dropped null desc     : {before - len(df):,} rows")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["customer_id"]       = df["customer_id"].astype(int)
    df["invoice_id"]        = df["invoice_id"].astype(str)
    df["stock_code"]        = df["stock_code"].astype(str)   # pyarrow needs explicit str
    df["invoice_date"]      = pd.to_datetime(df["invoice_date"])
    df["revenue"]           = df["quantity"] * df["unit_price"]   # line-level revenue
    df["invoice_year"]      = df["invoice_date"].dt.year
    df["invoice_month"]     = df["invoice_date"].dt.month
    df["invoice_yearmonth"] = df["invoice_date"].dt.to_period("M").astype(str)
    # Store as string — date objects are not parquet-portable across all engines
    df["invoice_date_only"] = df["invoice_date"].dt.strftime("%Y-%m-%d")
    return df


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n── [1] Load ─────────────────────────────────────")
    df = load_raw(RAW_PATH)

    print("\n── [2] Standardize columns ──────────────────────")
    df = standardize_columns(df)

    print("\n── Cleaning steps ───────────────────────────────")
    df            = drop_duplicates(df)
    df, returns   = separate_returns(df)
    df            = drop_missing_customer(df)
    df            = drop_bad_quantity(df)
    df            = drop_bad_price(df)
    df            = drop_missing_description(df)

    print("\n── [9] Add derived columns ──────────────────────")
    df = add_derived_columns(df)

    print("\n── [10] Save outputs ────────────────────────────")
    df.to_csv(CLEAN_CSV, index=False)
    df.to_parquet(CLEAN_PARQUET, index=False, engine="pyarrow")
    returns.to_csv(RETURNS_OUT, index=False)
    print(f"  Saved: {CLEAN_CSV}")
    print(f"  Saved: {CLEAN_PARQUET}")
    print(f"  Saved: {RETURNS_OUT}")

    print("\n── Final validation ─────────────────────────────")
    print(f"  Total clean rows      : {len(df):,}")
    print(f"  Columns               : {list(df.columns)}")
    print(f"  Nulls                 : {df.isnull().sum().sum()}")
    print(f"  Unique customers      : {df['customer_id'].nunique():,}")
    print(f"  Unique invoices       : {df['invoice_id'].nunique():,}")
    print(f"  Date range            : {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    print(f"  Total revenue         : £{df['revenue'].sum():,.2f}")
    assert df.isnull().sum().sum() == 0,          "FAIL: nulls remain"
    assert (df["quantity"] <= 0).sum() == 0,      "FAIL: bad quantity"
    assert (df["unit_price"] <= 0).sum() == 0,    "FAIL: bad price"
    assert (df["revenue"] <= 0).sum() == 0,       "FAIL: bad revenue"
    print("  All assertions passed ✓")


if __name__ == "__main__":
    run()
