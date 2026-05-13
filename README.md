# Customer 360 Revenue Intelligence Platform

> End-to-end customer analytics portfolio project built on the UCI Online Retail II dataset.

---

## Project Summary

This project transforms raw e-commerce transaction data into actionable customer intelligence.  
It covers the full analytics engineering lifecycle: data cleaning → EDA → feature engineering →  
RFM segmentation → churn prediction → explainability → interactive dashboard.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, openpyxl |
| Visualization | Plotly, Matplotlib, Seaborn |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |
| Optional (Cloud) | Google BigQuery |

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Size:** ~1M transactions, 8 columns
- **Period:** December 2009 – December 2011
- **Format:** `.xlsx` (two sheets: Year 2009-2010, Year 2010-2011)

---

## Project Structure

```
CUSTOMER 360 ANALYSIS/
├── data/
│   ├── raw/                  # Original unmodified dataset (not committed to Git)
│   └── processed/            # Cleaned and feature-engineered data
├── notebooks/                # Exploratory analysis notebooks
├── src/                      # Reusable Python modules (cleaning, features, model)
├── app/                      # Streamlit dashboard
├── reports/
│   ├── figures/              # Saved chart images
│   └── screenshots/          # Dashboard screenshots for README / resume
├── models/                   # Trained model artifacts (.pkl files)
├── sql/                      # SQL queries (for BigQuery integration)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Steps

| Step | Description | Status |
|---|---|---|
| 1 | Project setup & environment | ✅ Complete |
| 2 | Data ingestion & cleaning | ⬜ Pending |
| 3 | Exploratory data analysis | ⬜ Pending |
| 4 | RFM segmentation | ⬜ Pending |
| 5 | Customer feature engineering | ⬜ Pending |
| 6 | Churn label creation | ⬜ Pending |
| 7 | Churn prediction model | ⬜ Pending |
| 8 | SHAP explainability | ⬜ Pending |
| 9 | Customer health score | ⬜ Pending |
| 10 | Streamlit dashboard | ⬜ Pending |

---

## Setup Instructions

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd "CUSTOMER 360 ANALYSIS"

# 2. Create and activate conda environment
conda create -n customer360 python=3.11 -y
conda activate customer360

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Download online_retail_II.xlsx from UCI and put it in:
#    data/raw/online_retail_II.xlsx

# 5. Run the Streamlit app (after completing all steps)
streamlit run app/dashboard.py
```

---

## Author

**Prajwal Shekar**  
Data Analyst | Data Engineer  
[prajwalshekar22@gmail.com](mailto:prajwalshekar22@gmail.com)

---

*Built as a portfolio project to demonstrate end-to-end customer analytics engineering.*
