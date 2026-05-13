"""
01_load_and_inspect.py
----------------------
STEP 2: Dataset placement and initial inspection.

Reads both sheets of online_retail_II.xlsx, prints schema info,
and saves a raw inspection summary to reports/data_inspection.txt.

Run from project root:
    python src/01_load_and_inspect.py
"""

import pandas as pd
import os

RAW_PATH     = "data/raw/online_retail_II.xlsx"
REPORTS_DIR  = "reports"
OUTPUT_TXT   = "reports/data_inspection.txt"


def inspect_sheet(name: str, df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"\n{'='*55}")
    lines.append(f"  SHEET: {name}")
    lines.append(f"{'='*55}")
    lines.append(f"  Rows          : {len(df):,}")
    lines.append(f"  Columns       : {df.shape[1]}")
    lines.append(f"  Column names  : {list(df.columns)}")
    lines.append(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    lines.append(f"\n  Sample (top 3):\n{df.head(3).to_string()}")
    if "InvoiceDate" in df.columns:
        lines.append(f"\n  Date range    : {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    lines.append(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
    return "\n".join(lines)


def run():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Reading: {RAW_PATH}")
    xl = pd.ExcelFile(RAW_PATH, engine="openpyxl")
    print(f"Sheets found: {xl.sheet_names}")

    all_output = [f"RAW DATA INSPECTION REPORT", f"File: {RAW_PATH}"]

    dfs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        report_text = inspect_sheet(sheet, df)
        print(report_text)
        all_output.append(report_text)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined_text = (
        f"\n{'='*55}\n  COMBINED (all sheets)\n{'='*55}\n"
        f"  Total rows    : {len(combined):,}\n"
        f"  Unique Customers : {combined['Customer ID'].nunique():,}\n"
        f"  Unique Invoices  : {combined['Invoice'].nunique():,}\n"
        f"  Unique Products  : {combined['StockCode'].nunique():,}\n"
        f"  Unique Countries : {combined['Country'].nunique():,}\n"
        f"  Date range    : {combined['InvoiceDate'].min()} -> {combined['InvoiceDate'].max()}"
    )
    print(combined_text)
    all_output.append(combined_text)

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(all_output))
    print(f"\nInspection report saved -> {OUTPUT_TXT}")


if __name__ == "__main__":
    run()
