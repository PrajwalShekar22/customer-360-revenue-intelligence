================================================================
  Customer 360 Revenue Intelligence Platform
  Project Summary — README Reference
================================================================

PROJECT TITLE
  Customer 360 Revenue Intelligence Platform

BUSINESS PROBLEM
  How can a business use customer transaction history to identify high-value customers,
  predict churn risk, and prioritize retention and growth actions?

SOLUTION
  An end-to-end Python analytics pipeline that transforms raw Excel transaction data into
  RFM segments, churn predictions, health scores, and action priorities, surfaced through
  a 7-tab interactive Streamlit dashboard.

DATASET
  UCI Online Retail II
  - Raw rows        : 1,067,371
  - Clean rows      : 779,425
  - Date range      : December 2009 – December 2011
  - Source          : UK-based online retailer
  - Currency        : All values assumed GBP (£) — no exchange rate data available
  - Limitation      : Does not contain a true churn label, demographics, or campaign data

TECH STACK
  Python 3.11 · pandas 2.2.2 · NumPy 1.26.4 · scikit-learn 1.4.2 · XGBoost 2.0.3
  SHAP 0.45.1 · Plotly 5.22 · Streamlit 1.35.0 · SQL · DuckDB 1.1.3 · Parquet/CSV · Git

KEY METRICS
  Total revenue             : £17,374,804.27
  Customers analyzed        : 5,878
  Repeat buyer rate         : 72.4% (4,255 customers)
  Repeat buyer revenue share: 96.8% (£16,814,532)
  Champion customers        : 1,297 (22.1% of customers, 68.3% of revenue)
  Retention targets         : 2,952 customers
  VIP customers             : 1,511 customers
  Health score range        : 13.49 – 99.99

CHURN LABELING
  Method        : Time-window — observation window Dec 2009 – Jun 2011
                                prediction window  Jul 2011 – Dec 2011
  Labeled rows  : 5,041 customers
  Churned       : 2,512 (49.8%)
  Retained      : 2,529 (50.2%)
  Not Scored    : 837 (prediction-window-only customers)
  Leakage       : All features prefixed obs_ — observation period only

MODEL PERFORMANCE
  Best model    : Logistic Regression (selected for ROC-AUC + interpretability)
  ROC-AUC       : 0.8148
  Avg Precision : 0.8004
  F1 Score      : 0.7305
  Recall        : 0.7435

  Comparison:
  Logistic Regression : ROC-AUC 0.8148 · F1 0.7305 ← BEST
  Random Forest       : ROC-AUC 0.8114 · F1 0.7417
  XGBoost             : ROC-AUC 0.7978 · F1 0.7301

SHAP EXPLAINABILITY
  Method    : shap.LinearExplainer on 1,000-row sample
  Outputs   : beeswarm.png, bar.png, churn_global_feature_importance.csv
  Top positive drivers (increase churn risk): obs_recency_days, obs_average_unit_price
  Top negative drivers (reduce churn risk)  : obs_unique_purchase_days, obs_m_score

HEALTH SCORE COMPONENTS
  RFM component    : 40 pts — (rfm_total_score / 15) * 40
  Churn safety     : 30 pts — (1 - churn_probability) * 30; neutral 15 for Not Scored
  Revenue          : 20 pts — revenue percentile rank * 20
  Engagement       : 10 pts — (is_repeat_buyer * 0.5 + freq_percentile * 0.5) * 10

HEALTH TIER DISTRIBUTION
  Excellent  : 1,099 customers (18.7%)
  Healthy    : 1,374 customers (23.4%)
  Watchlist  : 1,643 customers (28.0%)
  At Risk    : 1,401 customers (23.8%)
  Critical   :   361 customers  (6.1%)

ACTION PRIORITY DISTRIBUTION
  Priority 1 — Urgent Retention : 1,842 customers · £864K revenue
  Priority 2 — High Value Save  :   616 customers · £1.36M revenue
  Priority 3 — Growth Opportunity: 1,436 customers · £1.11M revenue
  Priority 4 — Loyalty / Upsell : 1,909 customers · £13.94M revenue
  Priority 5 — Low Cost Nurture :    75 customers · £109K revenue

SQL ANALYTICS LAYER (DuckDB)
  DuckDB SQL layer added as Step 12C.
  Engine       : DuckDB 1.1.3 — in-process analytical SQL over Parquet files
  SQL files    : 10 (sql/01_executive_kpis.sql … sql/10_customer_value_tiers.sql)
  Outputs      : 10 CSV reporting tables in reports/sql_outputs/
  Reconciliation: £17,374,804.27 vs £17,374,804.27 — £0.00 difference (PASSED)
  Purpose      : SQL analytics demonstration + future BigQuery migration path
  Runner       : src/12_sql_analytics_layer.py

DASHBOARD TABS (7)
  1. Executive Overview
  2. Customer Segments
  3. Churn Risk & Retention
  4. Customer Lookup
  5. Revenue & Products
  6. Action Plan
  7. About Project

DEPLOYMENT STATUS
  - Local: complete (streamlit run app/streamlit_app.py → http://localhost:8501)
  - GitHub push: pending
  - Streamlit Cloud deployment: pending

LIMITATIONS
  - Dataset is historical (2009–2011)
  - Churn is engineered from inactivity, not a company-provided label
  - All monetary values assumed GBP; no exchange rates
  - No demographics or campaign data
  - Country coefficients are associative, not causal
  - Portfolio/educational project — not a production system

FUTURE IMPROVEMENTS
  - Streamlit Community Cloud deployment
  - BigQuery warehouse layer
  - Looker Studio executive dashboard
  - Docker containerisation
  - GitHub Actions CI
  - Model monitoring and drift detection
  - Live data refresh pipeline

AUTHOR
  Prajwal Gorkhar Chandrashekar
  Data Analyst | Business Analytics | Machine Learning | Data Engineering
  GitHub    : https://github.com/PrajwalShekar22
  LinkedIn  : https://www.linkedin.com/in/prajwalshekar
  Portfolio : https://www.datascienceportfol.io/pgc
================================================================
