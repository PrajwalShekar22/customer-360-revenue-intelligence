"""
09_train_churn_model.py
-----------------------
STEP 9: Churn Model Training

Trains, evaluates, and compares three models on observation-window features only:
  1. Logistic Regression (numeric scaled)
  2. Random Forest       (unscaled)
  3. XGBoost             (unscaled, if installed)

Best model is selected by ROC-AUC (recall used as tiebreaker).

Leakage guarantee:
  Input = churn_model_base.parquet, which contains only obs_* prefixed features.
  No full-period columns (total_revenue, recency_days, rfm_segment, etc.) are present.

Outputs:
  models/churn_model.pkl
  models/churn_preprocessor.pkl
  models/churn_feature_columns.json
  models/model_metrics.json
  data/processed/churn_predictions.csv / .parquet
  reports/churn_model_summary.txt
  reports/churn_model_metrics.csv
  reports/churn_feature_importance.csv
  reports/figures/churn_confusion_matrix.html
  reports/figures/churn_roc_curve.html
  reports/figures/churn_precision_recall_curve.html
  reports/figures/churn_feature_importance.html
  reports/figures/churn_probability_distribution.html

Run from project root:
    python src/09_train_churn_model.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── XGBoost (optional) ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("  WARNING: xgboost not importable. Skipping XGBoost model.")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH    = Path("data/processed/churn_model_base.parquet")
MODELS_DIR   = Path("models")
REPORTS_DIR  = Path("reports")
FIGURES_DIR  = Path("reports/figures")
DATA_OUT     = Path("data/processed")

for p in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DATA_OUT]:
    p.mkdir(parents=True, exist_ok=True)

# ── Columns ───────────────────────────────────────────────────────────────────
TARGET      = "churned"
ID_COL      = "customer_id"
EXCLUDE_COLS = {
    "customer_id", "churned", "purchased_in_prediction",
    "obs_first_purchase_date", "obs_last_purchase_date",
    # explicit leakage guard
    "total_revenue", "total_invoices", "recency_days",
    "rfm_segment", "customer_health_score",
}

NUMERIC_COLS    = []  # filled dynamically after load
CATEGORICAL_COLS = []


# ── Chart helper ───────────────────────────────────────────────────────────────
def save_chart(fig, filename: str):
    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    print(f"  Chart saved -> {path}")


# ── Metric helper ──────────────────────────────────────────────────────────────
def evaluate(name: str, pipeline, X_test, y_test) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "model":             name,
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "precision":         round(precision_score(y_test, y_pred), 4),
        "recall":            round(recall_score(y_test, y_pred), 4),
        "f1":                round(f1_score(y_test, y_pred), 4),
        "roc_auc":           round(roc_auc_score(y_test, y_proba), 4),
        "average_precision": round(average_precision_score(y_test, y_proba), 4),
        "true_negative":     int(tn),
        "false_positive":    int(fp),
        "false_negative":    int(fn),
        "true_positive":     int(tp),
    }


# ── Preprocessors ──────────────────────────────────────────────────────────────
def make_preprocessor(scale_numeric: bool, num_cols, cat_cols):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(transformers=[
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline(cat_steps), cat_cols),
    ], remainder="drop")


# ── Feature name extractor ─────────────────────────────────────────────────────
def get_feature_names(preprocessor, num_cols, cat_cols) -> list:
    """Extract human-readable feature names after ColumnTransformer fit."""
    cat_names = (preprocessor.named_transformers_["cat"]
                 .named_steps["onehot"]
                 .get_feature_names_out(cat_cols))
    return list(num_cols) + list(cat_names)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run():
    summary_lines = []

    def log(s=""):
        summary_lines.append(s)
        print(s)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n── Loading churn model base ─────────────────────")
    df = pd.read_parquet(BASE_PATH, engine="pyarrow")
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {df.columns.tolist()}")
    print(f"  Churn dist     : {df[TARGET].value_counts().to_dict()}")
    print(f"  Churn rate     : {df[TARGET].mean()*100:.1f}%")

    # ── Feature columns ───────────────────────────────────────────────────────
    all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    cat_cols  = [c for c in all_feature_cols if df[c].dtype == object]
    num_cols  = [c for c in all_feature_cols if c not in cat_cols]

    NUMERIC_COLS.extend(num_cols)
    CATEGORICAL_COLS.extend(cat_cols)

    print(f"\n  Numeric features ({len(num_cols)})   : {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n── Pre-training assertions ──────────────────────")
    assert TARGET in df.columns,            "FAIL: target column missing"
    assert ID_COL in df.columns,            "FAIL: customer_id missing"
    assert df[ID_COL].is_unique,            "FAIL: customer_id not unique"
    assert set(df[TARGET].unique()) <= {0,1},"FAIL: target not binary"
    leakage = EXCLUDE_COLS & set(all_feature_cols)
    leakage -= {"obs_first_purchase_date", "obs_last_purchase_date"}  # already excluded above
    assert not leakage,                     f"FAIL: leakage cols in features: {leakage}"
    assert all(c.startswith("obs_") for c in all_feature_cols), \
        "FAIL: non-obs_ feature found"
    print("  All assertions passed ✓")

    # ── Train / test split ────────────────────────────────────────────────────
    print("\n── Train / test split (80/20, stratified) ───────")
    X = df[all_feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train rows     : {len(X_train):,}  (churned={y_train.sum()}, retained={len(y_train)-y_train.sum()})")
    print(f"  Test rows      : {len(X_test):,}   (churned={y_test.sum()}, retained={len(y_test)-y_test.sum()})")

    # ── Build & train models ──────────────────────────────────────────────────
    print("\n── Training models ──────────────────────────────")
    models_to_train = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(True,  num_cols, cat_cols)),
            ("classifier",   LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(False, num_cols, cat_cols)),
            ("classifier",   RandomForestClassifier(n_estimators=200, max_depth=None,
                                                    min_samples_leaf=5, random_state=42,
                                                    n_jobs=-1)),
        ]),
    }
    if XGBOOST_OK:
        models_to_train["XGBoost"] = Pipeline([
            ("preprocessor", make_preprocessor(False, num_cols, cat_cols)),
            ("classifier",   XGBClassifier(n_estimators=200, max_depth=6,
                                           learning_rate=0.1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           use_label_encoder=False,
                                           eval_metric="logloss",
                                           random_state=42, n_jobs=-1)),
        ])

    results   = {}
    pipelines = {}

    for name, pipeline in models_to_train.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(name, pipeline, X_test, y_test)
        results[name]   = metrics
        pipelines[name] = pipeline
        print(f"ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

    # ── Print metrics table ───────────────────────────────────────────────────
    print("\n── Model comparison table ───────────────────────")
    hdr = f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ROC-AUC':>8} {'AvgPrec':>8}"
    print(hdr)
    print("  " + "-" * 68)
    for m in results.values():
        print(f"  {m['model']:<22} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>8.4f} "
              f"{m['average_precision']:>8.4f}")

    # ── Select best model ─────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: (results[k]["roc_auc"], results[k]["recall"]))
    best_metrics  = results[best_name]
    best_pipeline = pipelines[best_name]
    print(f"\n  Best model     : {best_name}")
    print(f"  ROC-AUC        : {best_metrics['roc_auc']:.4f}")
    print(f"  Recall (churn) : {best_metrics['recall']:.4f}")

    # ── Save model artifacts ──────────────────────────────────────────────────
    print("\n── Saving model artifacts ───────────────────────")
    joblib.dump(best_pipeline, MODELS_DIR / "churn_model.pkl")
    joblib.dump(best_pipeline.named_steps["preprocessor"],
                MODELS_DIR / "churn_preprocessor.pkl")

    with open(MODELS_DIR / "churn_feature_columns.json", "w") as f:
        json.dump({"numeric": num_cols, "categorical": cat_cols,
                   "all_features": all_feature_cols}, f, indent=2)

    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: models/churn_model.pkl")
    print(f"  Saved: models/churn_preprocessor.pkl")
    print(f"  Saved: models/churn_feature_columns.json")
    print(f"  Saved: models/model_metrics.json")

    metrics_df = pd.DataFrame(results.values())
    metrics_df.to_csv(REPORTS_DIR / "churn_model_metrics.csv", index=False)
    print(f"  Saved: reports/churn_model_metrics.csv")

    # ── Predictions for all customers ─────────────────────────────────────────
    print("\n── Generating predictions (all customers) ───────")
    X_all   = df[all_feature_cols]
    y_proba = best_pipeline.predict_proba(X_all)[:, 1]
    y_pred  = best_pipeline.predict(X_all)

    def risk_tier(p):
        if p < 0.30:   return "Low Risk"
        if p < 0.60:   return "Medium Risk"
        if p < 0.80:   return "High Risk"
        return "Critical Risk"

    pred_df = pd.DataFrame({
        "customer_id":      df[ID_COL].values,
        "actual_churned":   df[TARGET].values,
        "predicted_churned": y_pred,
        "churn_probability": y_proba.round(4),
        "risk_tier":        [risk_tier(p) for p in y_proba],
    })

    pred_df.to_csv(DATA_OUT  / "churn_predictions.csv",     index=False)
    pred_df.to_parquet(DATA_OUT / "churn_predictions.parquet", index=False, engine="pyarrow")
    print(f"  Saved: data/processed/churn_predictions.csv")
    print(f"  Saved: data/processed/churn_predictions.parquet")
    print(f"\n  Risk tier distribution:")
    tier_order = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    for tier in tier_order:
        n = (pred_df["risk_tier"] == tier).sum()
        print(f"    {tier:<15}: {n:>4,}  ({n/len(pred_df)*100:.1f}%)")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\n── Extracting feature importance ────────────────")
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    clf = best_pipeline.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        imp_type    = "gain"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        imp_type    = "coefficient_magnitude"
    else:
        importances = np.ones(len(feature_names))
        imp_type    = "unknown"

    # Guard: trim to min length in case OHE expanded feature count
    min_len     = min(len(importances), len(feature_names))
    importances = importances[:min_len]
    feature_names = feature_names[:min_len]

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["importance_type"] = imp_type
    imp_df.to_csv(REPORTS_DIR / "churn_feature_importance.csv", index=False)
    print(f"  Top 10 features ({imp_type}):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:<45} {row['importance']:.4f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n── Saving charts ────────────────────────────────")

    # 1. Confusion matrix heatmap
    cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred Retained", "Pred Churned"],
        y=["Actual Retained", "Actual Churned"],
        text=cm, texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig_cm.update_layout(
        title=f"Confusion Matrix — {best_name}",
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    save_chart(fig_cm, "churn_confusion_matrix.html")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, best_pipeline.predict_proba(X_test)[:, 1])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{best_name} (AUC={best_metrics['roc_auc']:.4f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="grey"),
                                  name="Random baseline"))
    fig_roc.update_layout(title=f"ROC Curve — {best_name}",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
    save_chart(fig_roc, "churn_roc_curve.html")

    # 3. Precision-recall curve
    prec, rec, _ = precision_recall_curve(y_test, best_pipeline.predict_proba(X_test)[:, 1])
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=best_name))
    fig_pr.update_layout(title=f"Precision-Recall Curve — {best_name}",
                          xaxis_title="Recall", yaxis_title="Precision")
    save_chart(fig_pr, "churn_precision_recall_curve.html")

    # 4. Feature importance bar chart (top 20)
    fig_imp = px.bar(
        imp_df.head(20).sort_values("importance"),
        x="importance", y="feature", orientation="h",
        title=f"Top 20 Feature Importances — {best_name} ({imp_type})",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance", color_continuous_scale="Oranges",
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    save_chart(fig_imp, "churn_feature_importance.html")

    # 5. Churn probability distribution by actual class
    fig_dist = px.histogram(
        pred_df, x="churn_probability", color="actual_churned",
        nbins=50, barmode="overlay", opacity=0.7,
        title="Churn Probability Distribution by Actual Class",
        labels={"churn_probability": "Predicted Churn Probability",
                "actual_churned": "Actual Churned"},
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
    )
    save_chart(fig_dist, "churn_probability_distribution.html")

    # ── Final assertions ──────────────────────────────────────────────────────
    print("\n── Post-save assertions ─────────────────────────")
    assert (MODELS_DIR / "churn_model.pkl").exists(),        "FAIL: model not saved"
    assert (MODELS_DIR / "model_metrics.json").exists(),     "FAIL: metrics not saved"
    assert len(pred_df) == len(df),                          "FAIL: prediction count mismatch"
    assert pred_df["customer_id"].is_unique,                 "FAIL: pred customer_id not unique"
    assert pred_df["churn_probability"].between(0,1).all(),  "FAIL: probability out of range"
    assert pred_df["risk_tier"].notna().all(),               "FAIL: null risk tier"
    print("  All post-save assertions passed ✓")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n── Saving summary report ────────────────────────")
    log("=" * 64)
    log("  STEP 9 — CHURN MODEL TRAINING SUMMARY")
    log("  Customer 360 Revenue Intelligence Platform")
    log("=" * 64)
    log(f"\n  Input file        : {BASE_PATH}")
    log(f"  Dataset shape     : {df.shape}")
    log(f"  Target (churned)  : 1={df[TARGET].sum():,}  0={(df[TARGET]==0).sum():,}  "
        f"rate={df[TARGET].mean()*100:.1f}%")
    log(f"\n  Train rows        : {len(X_train):,}")
    log(f"  Test rows         : {len(X_test):,}")

    log(f"\n  Feature columns used ({len(all_feature_cols)}):")
    for c in all_feature_cols:
        log(f"    {c}")

    log(f"\n  Excluded columns  :")
    for c in sorted(EXCLUDE_COLS):
        log(f"    {c}")

    log(f"\n  Leakage prevention: confirmed — all features are obs_* prefixed. ✓")

    log(f"\n{'─'*64}")
    log(f"  MODEL METRICS TABLE")
    log(f"{'─'*64}")
    log(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'ROC-AUC':>8} {'AvgPrec':>8}")
    log(f"  {'-'*68}")
    for m in results.values():
        log(f"  {m['model']:<22} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>8.4f} "
            f"{m['average_precision']:>8.4f}")

    log(f"\n  Best model selected: {best_name}")
    log(f"\n  Confusion matrix ({best_name} on test set):")
    log(f"    True Negative  (retained, predicted retained) : {best_metrics['true_negative']}")
    log(f"    False Positive (retained, predicted churned)  : {best_metrics['false_positive']}")
    log(f"    False Negative (churned, predicted retained)  : {best_metrics['false_negative']}")
    log(f"    True Positive  (churned, predicted churned)   : {best_metrics['true_positive']}")

    log(f"\n{'─'*64}")
    log(f"  METRIC INTERPRETATION ({best_name})")
    log(f"{'─'*64}")
    log(f"  Accuracy  {best_metrics['accuracy']:.4f}  — overall correct predictions")
    log(f"  Precision {best_metrics['precision']:.4f}  — of customers predicted to churn,")
    log(f"            {best_metrics['precision']*100:.1f}% actually churned (false alarm rate)")
    log(f"  Recall    {best_metrics['recall']:.4f}  — of customers who churned,")
    log(f"            {best_metrics['recall']*100:.1f}% were correctly identified (catch rate)")
    log(f"  F1        {best_metrics['f1']:.4f}  — harmonic mean of precision/recall")
    log(f"  ROC-AUC   {best_metrics['roc_auc']:.4f}  — probability model ranks churner above")
    log(f"            non-churner; 1.0 is perfect, 0.5 is random")

    log(f"\n{'─'*64}")
    log(f"  RISK TIER DISTRIBUTION")
    log(f"{'─'*64}")
    for tier in tier_order:
        n   = (pred_df["risk_tier"] == tier).sum()
        pct = n / len(pred_df) * 100
        log(f"  {tier:<15}: {n:>4,}  ({pct:.1f}%)")

    log(f"\n{'─'*64}")
    log(f"  TOP 10 FEATURE IMPORTANCES ({imp_type})")
    log(f"{'─'*64}")
    for _, row in imp_df.head(10).iterrows():
        log(f"  {row['feature']:<45} {row['importance']:.4f}")

    log(f"\n{'─'*64}")
    log(f"  BUSINESS INTERPRETATION")
    log(f"{'─'*64}")
    log(f"  High churn probability means the model estimates the customer")
    log(f"  is unlikely to return to purchase within the prediction window.")
    log(f"  Risk tiers translate probabilities into actionable categories:")
    log(f"    Critical Risk (>=0.80) : Immediate personalised outreach + discount")
    log(f"    High Risk    (0.60-0.80): Win-back email campaign within 7 days")
    log(f"    Medium Risk  (0.30-0.60): Monitor; trigger if no purchase in 30d")
    log(f"    Low Risk     (<0.30)    : Business-as-usual; loyalty rewards")

    log(f"\n{'─'*64}")
    log(f"  LIMITATIONS")
    log(f"{'─'*64}")
    log(f"  1. Churn label is engineered from purchase-inactivity, not a real")
    log(f"     company-provided churn flag.")
    log(f"  2. Dataset is historical (Dec 2009 – Dec 2011). Customer behaviour")
    log(f"     patterns may differ in newer periods.")
    log(f"  3. Model should be retrained periodically on fresh transaction data.")
    log(f"  4. 837 customers who only appeared post-cutoff are excluded; their")
    log(f"     risk profile is unknown.")

    with open(REPORTS_DIR / "churn_model_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary saved -> reports/churn_model_summary.txt")

    # ── Final terminal summary ────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  STEP 9 churn model training completed successfully.")
    print(f"{'='*64}")


if __name__ == "__main__":
    run()
