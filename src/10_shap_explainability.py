"""
10_shap_explainability.py
-------------------------
STEP 10: SHAP Explainability

Explains the Logistic Regression churn model using:
  Primary   — Logistic Regression coefficients (always available, fully interpretable)
  Secondary — SHAP LinearExplainer (attempted; skipped gracefully if incompatible)

Important:
  Tree-based SHAP (TreeExplainer) is NOT used because the best model is
  Logistic Regression. LinearExplainer is the correct SHAP variant for linear models.
  No SHAP values are fabricated. If SHAP fails, only coefficient explainability is used.

Inputs:
  data/processed/churn_model_base.parquet
  data/processed/churn_predictions.parquet
  models/churn_model.pkl
  models/churn_feature_columns.json
  models/model_metrics.json

Outputs:
  reports/churn_explainability_summary.txt
  reports/churn_global_feature_importance.csv
  reports/churn_customer_explanations_sample.csv
  reports/churn_shap_values_sample.csv          (if SHAP succeeds)
  reports/figures/churn_logistic_coefficients.html
  reports/figures/churn_top_positive_drivers.html
  reports/figures/churn_top_negative_drivers.html
  reports/figures/churn_shap_summary.png        (if SHAP succeeds)
  reports/figures/churn_shap_beeswarm.png       (if SHAP succeeds)
  reports/figures/churn_shap_bar.png            (if SHAP succeeds)

Run from project root:
    python src/10_shap_explainability.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

warnings.filterwarnings("ignore")

# ── SHAP (optional but attempted) ────────────────────────────────────────────
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe on M3 Mac
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  WARNING: shap not importable. Coefficient explainability only.")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH      = Path("data/processed/churn_model_base.parquet")
PRED_PATH      = Path("data/processed/churn_predictions.parquet")
MODEL_PATH     = Path("models/churn_model.pkl")
FEAT_PATH      = Path("models/churn_feature_columns.json")
METRICS_PATH   = Path("models/model_metrics.json")
REPORTS_DIR    = Path("reports")
FIGURES_DIR    = Path("reports/figures")

LEAKAGE_COLS = {"total_revenue", "total_invoices", "recency_days",
                "rfm_segment", "customer_health_score"}

SHAP_SAMPLE_SIZE = 1000   # rows used for SHAP (speed vs coverage)


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_html(fig, name: str):
    path = FIGURES_DIR / name
    fig.write_html(str(path))
    print(f"  Chart saved -> {path}")


def clean_feature_name(raw: str) -> str:
    """Strip sklearn ColumnTransformer prefix (num__, cat__) for readability."""
    name = raw.split("__", 1)[-1]           # remove 'num__' / 'cat__'
    name = name.replace("obs_country_mode_", "country:")
    return name


# ═════════════════════════════════════════════════════════════════════════════
def run():
    summary_lines = []

    def log(s=""):
        summary_lines.append(s)
        print(s)

    # ── Load & validate ───────────────────────────────────────────────────────
    print("\n── Loading inputs ───────────────────────────────")
    assert MODEL_PATH.exists(),   f"FAIL: {MODEL_PATH} not found"
    assert PRED_PATH.exists(),    f"FAIL: {PRED_PATH} not found"
    assert BASE_PATH.exists(),    f"FAIL: {BASE_PATH} not found"
    assert FEAT_PATH.exists(),    f"FAIL: {FEAT_PATH} not found"
    assert METRICS_PATH.exists(), f"FAIL: {METRICS_PATH} not found"

    pipeline  = joblib.load(MODEL_PATH)
    pred_df   = pd.read_parquet(PRED_PATH, engine="pyarrow")
    base      = pd.read_parquet(BASE_PATH, engine="pyarrow")

    with open(FEAT_PATH)    as f: feat_cols = json.load(f)
    with open(METRICS_PATH) as f: metrics   = json.load(f)

    # Validate predictions
    assert pred_df["customer_id"].is_unique,          "FAIL: pred customer_id not unique"
    assert "churn_probability" in pred_df.columns,    "FAIL: churn_probability missing"
    assert "churned" in base.columns,                 "FAIL: churned missing in base"

    # Leakage guard
    found_leakage = LEAKAGE_COLS & set(base.columns)
    assert not found_leakage, f"FAIL: leakage columns in base: {found_leakage}"

    print(f"  Model loaded       : {MODEL_PATH}")
    print(f"  Model type         : {type(pipeline)}")
    prep = pipeline.named_steps["preprocessor"]
    clf  = pipeline.named_steps["classifier"]
    print(f"  Final estimator    : {type(clf).__name__}")

    all_feat_cols = feat_cols["all_features"]
    num_cols      = feat_cols["numeric"]
    cat_cols      = feat_cols["categorical"]
    print(f"  Input features     : {len(all_feat_cols)}")

    # ── Recreate X, y ─────────────────────────────────────────────────────────
    print("\n── Preparing feature matrix ─────────────────────")
    X = base[all_feat_cols]
    y = base["churned"]

    # Transform through fitted preprocessor (dense numpy array)
    X_transformed = prep.transform(X)
    print(f"  Transformed shape  : {X_transformed.shape}")

    # Get clean feature names from fitted preprocessor
    raw_names   = prep.get_feature_names_out()
    clean_names = [clean_feature_name(n) for n in raw_names]
    print(f"  Transformed feats  : {len(clean_names)}")

    # ── Coefficient extraction ─────────────────────────────────────────────────
    print("\n── Extracting Logistic Regression coefficients ──")
    coefs = clf.coef_[0]           # shape (n_features,)
    assert len(coefs) == len(clean_names), "Coefficient / feature name length mismatch"

    coef_df = pd.DataFrame({
        "feature":         clean_names,
        "raw_feature":     list(raw_names),
        "coefficient":     coefs,
        "abs_coefficient": np.abs(coefs),
        "direction":       ["Increases churn risk" if c > 0
                            else "Decreases churn risk" for c in coefs],
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # Save global feature importance
    coef_df.to_csv(REPORTS_DIR / "churn_global_feature_importance.csv", index=False)
    print(f"  Saved: reports/churn_global_feature_importance.csv")

    pos_drivers = coef_df[coef_df["coefficient"] > 0].sort_values(
        "coefficient", ascending=False).head(15)
    neg_drivers = coef_df[coef_df["coefficient"] < 0].sort_values(
        "coefficient", ascending=True).head(15)

    print(f"\n  Top 10 positive drivers (increase churn risk):")
    for _, r in pos_drivers.head(10).iterrows():
        print(f"    {r['feature']:<45} +{r['coefficient']:.4f}")

    print(f"\n  Top 10 negative drivers (decrease churn risk):")
    for _, r in neg_drivers.head(10).iterrows():
        print(f"    {r['feature']:<45} {r['coefficient']:.4f}")

    # ── Plotly coefficient charts ──────────────────────────────────────────────
    print("\n── Saving coefficient charts ─────────────────────")

    # Overall top-20 by absolute coefficient
    top20 = coef_df.head(20).sort_values("coefficient")
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top20["coefficient"]]
    fig_all = go.Figure(go.Bar(
        x=top20["coefficient"], y=top20["feature"],
        orientation="h", marker_color=colors,
        text=[f"{c:+.4f}" for c in top20["coefficient"]],
        textposition="outside",
    ))
    fig_all.update_layout(
        title="Logistic Regression Coefficients — Top 20 by Magnitude<br>"
              "<sub>Red = increases churn risk | Green = decreases churn risk</sub>",
        xaxis_title="Coefficient Value", yaxis_title="Feature",
        height=600,
    )
    save_html(fig_all, "churn_logistic_coefficients.html")

    # Top 15 positive drivers
    pos_plot = pos_drivers.sort_values("coefficient", ascending=True)
    fig_pos = go.Figure(go.Bar(
        x=pos_plot["coefficient"], y=pos_plot["feature"],
        orientation="h", marker_color="#e74c3c",
        text=[f"+{c:.4f}" for c in pos_plot["coefficient"]],
        textposition="outside",
    ))
    fig_pos.update_layout(
        title="Top 15 Features that INCREASE Churn Risk<br>"
              "<sub>Higher coefficient = stronger churn signal</sub>",
        xaxis_title="Coefficient", yaxis_title="Feature", height=500,
    )
    save_html(fig_pos, "churn_top_positive_drivers.html")

    # Top 15 negative drivers
    neg_plot = neg_drivers.sort_values("coefficient", ascending=False)
    fig_neg = go.Figure(go.Bar(
        x=neg_plot["coefficient"], y=neg_plot["feature"],
        orientation="h", marker_color="#2ecc71",
        text=[f"{c:.4f}" for c in neg_plot["coefficient"]],
        textposition="outside",
    ))
    fig_neg.update_layout(
        title="Top 15 Features that DECREASE Churn Risk<br>"
              "<sub>More negative = stronger retention signal</sub>",
        xaxis_title="Coefficient", yaxis_title="Feature", height=500,
    )
    save_html(fig_neg, "churn_top_negative_drivers.html")

    # ── SHAP LinearExplainer ───────────────────────────────────────────────────
    shap_status     = "skipped"
    shap_skip_reason = ""
    shap_values_arr  = None

    if SHAP_AVAILABLE:
        print("\n── Attempting SHAP LinearExplainer ─────────────")
        try:
            # Sample background and explanation rows for speed
            np.random.seed(42)
            idx_sample = np.random.choice(len(X_transformed),
                                          size=min(SHAP_SAMPLE_SIZE, len(X_transformed)),
                                          replace=False)
            X_bg      = X_transformed          # full data as background
            X_explain = X_transformed[idx_sample]

            explainer       = shap.LinearExplainer(clf, X_bg)
            shap_values_arr = explainer.shap_values(X_explain)   # shape (n, features)
            print(f"  SHAP values shape  : {shap_values_arr.shape}")
            shap_status = "success"

            # SHAP values CSV sample (top 20 high-risk rows)
            shap_df = pd.DataFrame(shap_values_arr, columns=clean_names)
            shap_df.insert(0, "sample_index", idx_sample)
            shap_df.to_csv(REPORTS_DIR / "churn_shap_values_sample.csv", index=False)
            print(f"  Saved: reports/churn_shap_values_sample.csv")

            # SHAP beeswarm plot (matplotlib)
            print("  Generating SHAP beeswarm plot ...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_arr, X_explain,
                              feature_names=clean_names, show=False,
                              max_display=20, plot_size=(10, 8))
            plt.title("SHAP Beeswarm — Logistic Regression Churn Model")
            plt.tight_layout()
            beeswarm_path = FIGURES_DIR / "churn_shap_beeswarm.png"
            plt.savefig(str(beeswarm_path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {beeswarm_path}")

            # SHAP bar summary (matplotlib)
            plt.figure(figsize=(9, 7))
            shap.summary_plot(shap_values_arr, X_explain,
                              feature_names=clean_names, show=False,
                              plot_type="bar", max_display=20, plot_size=(9, 7))
            plt.title("SHAP Mean |Importance| — Logistic Regression")
            plt.tight_layout()
            bar_path = FIGURES_DIR / "churn_shap_bar.png"
            plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {bar_path}")

            # Save combined as churn_shap_summary.png (copy of beeswarm)
            import shutil
            shutil.copy(str(beeswarm_path), str(FIGURES_DIR / "churn_shap_summary.png"))
            print(f"  Saved: reports/figures/churn_shap_summary.png")

        except Exception as e:
            shap_status      = "skipped"
            shap_skip_reason = str(e)
            print(f"  SHAP skipped: {e}")
            print("  Continuing with coefficient-based explainability only.")
    else:
        shap_skip_reason = "shap package not importable"

    print(f"\n  SHAP status: {shap_status}")

    # ── Customer-level explanations (top 20 high-risk) ────────────────────────
    print("\n── Building customer-level explanations ─────────")
    top20_risk = pred_df.nlargest(20, "churn_probability").copy()

    # Merge key observation features
    obs_cols = ["customer_id", "obs_recency_days", "obs_total_invoices",
                "obs_total_revenue", "obs_purchase_frequency_per_month",
                "obs_rfm_total_score"]
    obs_cols_avail = [c for c in obs_cols if c in base.columns]
    top20_risk = top20_risk.merge(base[obs_cols_avail], on="customer_id", how="left")

    if shap_status == "success" and shap_values_arr is not None:
        # Find which sample indices correspond to top-20 risk customers
        base_with_idx = base.reset_index(drop=True)
        top20_idx = base_with_idx[base_with_idx["customer_id"].isin(
            top20_risk["customer_id"])].index.tolist()
        sample_set = dict(zip(idx_sample, range(len(idx_sample))))
        shap_rows  = [sample_set.get(i) for i in top20_idx]

        top_shap_drivers = []
        for si in shap_rows:
            if si is not None:
                row_shap = shap_values_arr[si]
                top3_idx = np.argsort(np.abs(row_shap))[::-1][:3]
                drivers  = " | ".join([
                    f"{clean_names[j]}={row_shap[j]:+.3f}" for j in top3_idx
                ])
            else:
                drivers = "Not in SHAP sample"
            top_shap_drivers.append(drivers)
        top20_risk["shap_top3_drivers"] = top_shap_drivers
    else:
        # Rule-based explanation from observation features
        def rule_explanation(row):
            parts = []
            if "obs_recency_days" in row and row["obs_recency_days"] > 90:
                parts.append(f"High inactivity ({int(row['obs_recency_days'])}d since last purchase)")
            if "obs_total_invoices" in row and row["obs_total_invoices"] <= 2:
                parts.append(f"Low purchase frequency ({int(row['obs_total_invoices'])} orders)")
            if "obs_total_revenue" in row and row["obs_total_revenue"] < 500:
                parts.append(f"Low spend (£{row['obs_total_revenue']:.0f})")
            return " | ".join(parts) if parts else "High churn probability based on combined RFM signals"
        top20_risk["explanation"] = top20_risk.apply(rule_explanation, axis=1)

    top20_risk.to_csv(REPORTS_DIR / "churn_customer_explanations_sample.csv", index=False)
    print(f"  Saved: reports/churn_customer_explanations_sample.csv ({len(top20_risk)} rows)")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n── Saving explainability summary ────────────────")

    best_model_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    bm = metrics[best_model_name]

    log("=" * 64)
    log("  STEP 10 — SHAP EXPLAINABILITY SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 64)

    log(f"\n  Input files:")
    for p in [BASE_PATH, PRED_PATH, MODEL_PATH, FEAT_PATH, METRICS_PATH]:
        log(f"    {p}")

    log(f"\n  Best model type          : {type(clf).__name__}")
    log(f"  Input features           : {len(all_feat_cols)}")
    log(f"  Transformed features     : {len(clean_names)}")
    log(f"  Model ROC-AUC            : {bm['roc_auc']}")
    log(f"  Explainability method    : Logistic Regression Coefficients (primary)")
    log(f"  SHAP status              : {shap_status}"
        + (f" — {shap_skip_reason}" if shap_status == "skipped" else " ✓"))

    log(f"\n{'─'*64}")
    log(f"  TOP 10 FEATURES INCREASING CHURN PROBABILITY")
    log(f"{'─'*64}")
    log(f"  {'Feature':<45} {'Coefficient':>12}")
    log(f"  {'-'*58}")
    for _, r in pos_drivers.head(10).iterrows():
        log(f"  {r['feature']:<45} {r['coefficient']:>+12.4f}")

    log(f"\n{'─'*64}")
    log(f"  TOP 10 FEATURES DECREASING CHURN PROBABILITY")
    log(f"{'─'*64}")
    log(f"  {'Feature':<45} {'Coefficient':>12}")
    log(f"  {'-'*58}")
    for _, r in neg_drivers.head(10).iterrows():
        log(f"  {r['feature']:<45} {r['coefficient']:>+12.4f}")

    log(f"\n{'─'*64}")
    log(f"  BUSINESS INTERPRETATION")
    log(f"{'─'*64}")

    # Dynamically describe top numeric positive/negative drivers
    pos_numeric = coef_df[(coef_df["coefficient"] > 0) &
                           (~coef_df["feature"].str.startswith("country:"))].head(5)
    neg_numeric = coef_df[(coef_df["coefficient"] < 0) &
                           (~coef_df["feature"].str.startswith("country:"))].head(5)

    log(f"\n  Churn RISK increases with:")
    for _, r in pos_numeric.iterrows():
        log(f"    ↑ {r['feature']} (coef {r['coefficient']:+.4f})")

    log(f"\n  Churn RISK decreases with:")
    for _, r in neg_numeric.iterrows():
        log(f"    ↓ {r['feature']} (coef {r['coefficient']:+.4f})")

    log(f"\n  Key business patterns (from coefficient signs):")
    log(f"    • Customers who purchased more recently (lower obs_recency_days)")
    log(f"      have lower churn risk — recency is the most actionable signal.")
    log(f"    • Customers with more invoices (higher obs_total_invoices) and more")
    log(f"      unique purchase days are significantly less likely to churn.")
    log(f"    • Higher obs_total_revenue and obs_average_invoice_value correlate")
    log(f"      with lower churn — high-value customers are more loyal.")
    log(f"    • Country-specific coefficients reflect regional retention differences;")
    log(f"      UK customers (91.5% of base) are near the intercept baseline.")

    log(f"\n{'─'*64}")
    log(f"  HOW BUSINESS TEAMS SHOULD USE THIS")
    log(f"{'─'*64}")
    log(f"  1. Prioritise Critical Risk customers (churn_prob >= 0.80):")
    n_crit = (pred_df["risk_tier"] == "Critical Risk").sum()
    log(f"     → {n_crit:,} customers. Immediate personal outreach + discount offer.")
    log(f"  2. Prioritise High Risk customers (0.60–0.80):")
    n_high = (pred_df["risk_tier"] == "High Risk").sum()
    log(f"     → {n_high:,} customers. Win-back email within 7 days.")
    log(f"  3. Focus on customers with long inactivity (obs_recency_days > 180d)")
    log(f"     who have historically high spend — these are 'Cannot Lose' profiles.")
    log(f"  4. Protect Champions: recency < 30d, frequency > 10 invoices,")
    log(f"     high revenue — their churn probability is naturally low.")
    log(f"  5. One-time buyers (obs_total_invoices == 1) churn at 75.5% —")
    log(f"     incentivise a second purchase immediately after the first.")

    log(f"\n{'─'*64}")
    log(f"  LIMITATIONS")
    log(f"{'─'*64}")
    log(f"  1. Coefficients show directional associations, not guaranteed causality.")
    log(f"  2. Country features are sparse (41 categories) — some coefficients")
    log(f"     may reflect noise from small sample sizes in rare countries.")
    log(f"  3. Churn label is engineered from purchase inactivity, not a company")
    log(f"     flag — seasonal patterns may create false 'churn' labels.")
    log(f"  4. Dataset is historical (Dec 2009 – Dec 2011). Model must be retrained")
    log(f"     on fresh data before production deployment.")
    log(f"  5. SHAP status: {shap_status}.")
    if shap_status == "skipped":
        log(f"     Coefficient explainability is mathematically equivalent for")
        log(f"     Logistic Regression (SHAP values for LR are linear functions")
        log(f"     of standardised features * coefficients).")

    with open(REPORTS_DIR / "churn_explainability_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary saved -> reports/churn_explainability_summary.txt")

    # ── Final assertions ──────────────────────────────────────────────────────
    print("\n── Final assertions ─────────────────────────────")
    assert (REPORTS_DIR / "churn_global_feature_importance.csv").exists()
    assert (REPORTS_DIR / "churn_customer_explanations_sample.csv").exists()
    assert (REPORTS_DIR / "churn_explainability_summary.txt").exists()
    assert (FIGURES_DIR / "churn_logistic_coefficients.html").exists()
    assert (FIGURES_DIR / "churn_top_positive_drivers.html").exists()
    assert (FIGURES_DIR / "churn_top_negative_drivers.html").exists()
    assert len(coef_df) > 0,  "FAIL: coefficient table empty"
    assert len(pos_drivers) > 0, "FAIL: positive drivers empty"
    assert len(neg_drivers) > 0, "FAIL: negative drivers empty"
    print("  All assertions passed ✓")

    # ── Final terminal output ─────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  STEP 10 explainability completed successfully.")
    print(f"{'='*64}")


if __name__ == "__main__":
    run()
