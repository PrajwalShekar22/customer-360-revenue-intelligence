"""
Customer 360 Revenue Intelligence Platform — Streamlit Dashboard
Step 12: Interactive business dashboard powered by customer_360.parquet

Developer: Prajwal Gorkhar Chandrashekar
Role: Data Analyst | Business Analytics | Machine Learning | Data Engineering
GitHub: https://github.com/PrajwalShekar22
LinkedIn: https://www.linkedin.com/in/prajwalshekar
Portfolio: https://www.datascienceportfol.io/pgc
Email: prajwalshekar22@gmail.com
"""

# ── Developer contact info (single source of truth) ──────────────────────────
DEV_NAME      = "Prajwal Gorkhar Chandrashekar"
DEV_ROLE      = "Data Analyst | Business Analytics | Machine Learning | Data Engineering"
DEV_GITHUB    = "https://github.com/PrajwalShekar22"
DEV_LINKEDIN  = "https://www.linkedin.com/in/prajwalshekar"
DEV_PORTFOLIO = "https://www.datascienceportfol.io/pgc"
DEV_EMAIL     = "prajwalshekar22@gmail.com"

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer 360 Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ── Header card ── */
    .header-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
        border-radius: 12px;
        padding: 24px 30px 20px 30px;
        margin-bottom: 18px;
        color: white;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .header-subtitle {
        font-size: 0.92rem;
        color: #bfdbfe;
        margin: 0 0 8px 0;
    }
    .header-owner {
        font-size: 0.82rem;
        color: #e0f2fe;
        margin: 0 0 12px 0;
    }
    .header-links {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .header-link {
        background: rgba(255,255,255,0.15);
        color: #ffffff !important;
        padding: 5px 14px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.25);
        transition: background 0.2s;
    }
    .header-link:hover { background: rgba(255,255,255,0.28); }
    .header-currency-note {
        font-size: 0.75rem;
        color: #93c5fd;
        border-top: 1px solid rgba(255,255,255,0.15);
        padding-top: 10px;
        margin-top: 4px;
    }

    /* ── Section heading ── */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #1d4ed8;
        padding-left: 10px;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ── KPI card ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        height: 100%;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-delta {
        font-size: 0.76rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* ── Insight / alert boxes ── */
    .insight-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #1e40af;
        line-height: 1.5;
    }
    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #9a3412;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #166534;
    }
    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #991b1b;
    }

    /* ── Contact cards ── */
    .contact-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }
    .contact-card-icon { font-size: 1.5rem; margin-bottom: 6px; }
    .contact-card-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .contact-card-link {
        font-size: 0.88rem;
        color: #1d4ed8 !important;
        text-decoration: none;
        font-weight: 500;
    }
    .contact-card-link:hover { text-decoration: underline; }

    /* ── Footer ── */
    .site-footer {
        margin-top: 48px;
        padding: 24px 0 12px 0;
        border-top: 1px solid #e5e7eb;
        text-align: center;
    }
    .footer-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 8px;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 10px;
    }
    .footer-link {
        font-size: 0.82rem;
        color: #1d4ed8 !important;
        text-decoration: none;
    }
    .footer-link:hover { text-decoration: underline; }
    .footer-note {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* ── Sidebar contact ── */
    .sidebar-contact {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 0.8rem;
    }
    .sidebar-contact-name {
        font-weight: 600;
        color: #1e3a5f;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    .sidebar-contact-role {
        color: #64748b;
        font-size: 0.72rem;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .sidebar-contact-links {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .sidebar-link {
        background: #1d4ed8;
        color: #fff !important;
        padding: 3px 10px;
        border-radius: 12px;
        text-decoration: none;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .sidebar-link:hover { background: #1e40af; }

    /* Hide default Streamlit footer/menu */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

HEALTH_TIER_ORDER  = ["Excellent", "Healthy", "Watchlist", "At Risk", "Critical"]
ACTION_PRIO_ORDER  = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5"]
VALUE_TIER_ORDER   = ["Top 10% Value", "High Value", "Mid Value", "Low Value"]
CHURN_TIER_ORDER   = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk", "Not Scored"]
RFM_SEG_ORDER = [
    "Champions", "Loyal Customers", "Big Spenders",
    "Potential Loyalists", "New Customers",
    "Needs Attention", "At Risk", "Cannot Lose",
    "Low Value", "Hibernating",
]

HEALTH_COLORS = {
    "Excellent": "#22c55e", "Healthy": "#84cc16",
    "Watchlist": "#f59e0b", "At Risk": "#f97316", "Critical": "#ef4444",
}
CHURN_COLORS = {
    "Low Risk": "#22c55e", "Medium Risk": "#84cc16",
    "High Risk": "#f97316", "Critical Risk": "#ef4444", "Not Scored": "#94a3b8",
}
PRIO_COLORS = {
    "Priority 1": "#ef4444", "Priority 2": "#f97316",
    "Priority 3": "#f59e0b", "Priority 4": "#22c55e", "Priority 5": "#94a3b8",
}

PLOTLY_TEMPLATE = "plotly_white"

# ── Formatting helpers ────────────────────────────────────────────────────────
def format_currency(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"£{value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"£{value/1_000:.1f}K"
    return f"£{value:,.2f}"

def format_percent(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"

def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"

def safe_mean(series):
    clean = series.dropna()
    return clean.mean() if len(clean) > 0 else np.nan

def kpi_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_customer_360():
    path = ROOT / "data/processed/customer_360.parquet"
    if not path.exists():
        return None, f"Required file not found: {path}"
    df = pd.read_parquet(path)
    required_cols = [
        "customer_id", "total_revenue", "health_tier", "rfm_segment",
        "action_priority", "customer_health_score", "churn_risk_tier",
        "final_recommended_action",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"
    return df, None

@st.cache_data
def load_transactions():
    path = ROOT / "data/processed/clean_transactions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_data
def load_feature_importance():
    path = ROOT / "reports/churn_global_feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_model_metrics():
    path = ROOT / "reports/churn_model_metrics.csv" if (ROOT / "reports/churn_model_metrics.csv").exists() \
        else ROOT / "reports/model_metrics.json"
    csv_path = ROOT / "reports/churn_model_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# ── Load data ─────────────────────────────────────────────────────────────────
df_all, load_error = load_customer_360()

if load_error:
    st.error(f"Data load error: {load_error}")
    st.stop()

txn_df = load_transactions()
feat_imp = load_feature_importance()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-card">
    <div class="header-title">📊 Customer 360 Revenue Intelligence Platform</div>
    <div class="header-subtitle">
        Customer Segmentation &nbsp;·&nbsp; Churn Risk &nbsp;·&nbsp;
        Health Scoring &nbsp;·&nbsp; Revenue Action Planning
    </div>
    <div class="header-owner">
        Built by <strong>{DEV_NAME}</strong> &nbsp;—&nbsp; {DEV_ROLE}
    </div>
    <div class="header-links">
        <a class="header-link" href="{DEV_GITHUB}" target="_blank">GitHub</a>
        <a class="header-link" href="{DEV_LINKEDIN}" target="_blank">LinkedIn</a>
        <a class="header-link" href="{DEV_PORTFOLIO}" target="_blank">Portfolio</a>
        <a class="header-link" href="mailto:{DEV_EMAIL}">📩 Email</a>
    </div>
    <div class="header-currency-note">
        💷 All monetary values are shown in <strong>GBP (£)</strong>, based on the original
        UK retailer transaction dataset (UCI Online Retail II, 2009–2011).
        Country filter applies to customer location only — currency symbol never changes.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Leave a filter empty to include all options. Expand to narrow your selection.")

    st.markdown(
        """
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;
                    padding:9px 12px;margin:6px 0 10px 0;font-size:0.78rem;color:#166534;
                    line-height:1.45">
        <b>Currency note:</b> All monetary values are shown in <b>GBP (£)</b>, based on the
        original UK retailer transaction dataset. Country filters customer location only.
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_health  = sorted(df_all["health_tier"].dropna().unique().tolist())
    all_rfm     = sorted(df_all["rfm_segment"].dropna().unique().tolist())
    all_churn   = sorted(df_all["churn_risk_tier"].dropna().unique().tolist())
    all_prio    = sorted(df_all["action_priority"].dropna().unique().tolist())
    all_value   = sorted(df_all["customer_value_tier"].dropna().unique().tolist())
    all_country = sorted(df_all["country_mode"].dropna().unique().tolist())

    # Reset button — clears all filter widget state
    if st.button("Reset All Filters", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

    with st.expander("Health Tier", expanded=False):
        sel_health = st.multiselect(
            "health_tier", all_health, default=[],
            placeholder="All health tiers",
            label_visibility="collapsed",
            key="flt_health",
        )

    with st.expander("RFM Segment", expanded=False):
        sel_rfm = st.multiselect(
            "rfm_segment", all_rfm, default=[],
            placeholder="All RFM segments",
            label_visibility="collapsed",
            key="flt_rfm",
        )

    with st.expander("Churn Risk Tier", expanded=False):
        sel_churn = st.multiselect(
            "churn_risk_tier", all_churn, default=[],
            placeholder="All risk tiers",
            label_visibility="collapsed",
            key="flt_churn",
        )

    with st.expander("Action Priority", expanded=False):
        sel_prio = st.multiselect(
            "action_priority", all_prio, default=[],
            placeholder="All priorities",
            label_visibility="collapsed",
            key="flt_prio",
        )

    with st.expander("Customer Value Tier", expanded=False):
        sel_value = st.multiselect(
            "customer_value_tier", all_value, default=[],
            placeholder="All value tiers",
            label_visibility="collapsed",
            key="flt_value",
        )

    with st.expander("Country", expanded=False):
        sel_country = st.multiselect(
            "country", all_country, default=[],
            placeholder="All countries",
            label_visibility="collapsed",
            key="flt_country",
        )

    st.divider()

    scored_opts = {"All Customers": "all", "Scored Only": 1, "Not Scored Only": 0}
    sel_scored_label = st.selectbox(
        "Model Scored Status", list(scored_opts.keys()), key="flt_scored"
    )
    sel_scored = scored_opts[sel_scored_label]

    rev_min = float(df_all["total_revenue"].min())
    rev_max = float(df_all["total_revenue"].max())
    rev_range = st.slider(
        "Revenue Range (£)",
        min_value=rev_min, max_value=rev_max,
        value=(rev_min, rev_max),
        format="£%.0f",
        key="flt_rev",
    )

    scored_mask = df_all["model_scored"] == 1
    cp_range = st.slider(
        "Churn Probability (scored only)",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        key="flt_cp",
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all.copy()

if sel_health:  df = df[df["health_tier"].isin(sel_health)]
if sel_rfm:     df = df[df["rfm_segment"].isin(sel_rfm)]
if sel_churn:   df = df[df["churn_risk_tier"].isin(sel_churn)]
if sel_prio:    df = df[df["action_priority"].isin(sel_prio)]
if sel_value:   df = df[df["customer_value_tier"].isin(sel_value)]
if sel_country: df = df[df["country_mode"].isin(sel_country)]

if sel_scored == 1:
    df = df[df["model_scored"] == 1]
elif sel_scored == 0:
    df = df[df["model_scored"] == 0]

df = df[(df["total_revenue"] >= rev_range[0]) & (df["total_revenue"] <= rev_range[1])]

# Apply churn probability range only to scored customers
scored_in_df = df[df["model_scored"] == 1]
mask_cp = (
    scored_in_df["churn_probability"].between(cp_range[0], cp_range[1])
)
keep_ids = set(scored_in_df.loc[mask_cp, "customer_id"]) | set(df.loc[df["model_scored"] == 0, "customer_id"])
df = df[df["customer_id"].isin(keep_ids)]

if len(df) == 0:
    st.markdown(
        '<div class="warning-box">No customers match the selected filters. Please adjust your filters.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Filter stats + contact in sidebar
with st.sidebar:
    st.markdown(f"**{len(df):,}** customers selected ({len(df)/len(df_all)*100:.1f}% of total)")
    st.divider()
    st.markdown(f"""
    <div class="sidebar-contact">
        <div class="sidebar-contact-name">{DEV_NAME}</div>
        <div class="sidebar-contact-role">Data Analyst · Business Analytics · ML · Data Engineering</div>
        <div class="sidebar-contact-links">
            <a class="sidebar-link" href="mailto:{DEV_EMAIL}">📩 Email</a>
            <a class="sidebar-link" href="{DEV_LINKEDIN}" target="_blank">LinkedIn</a>
            <a class="sidebar-link" href="{DEV_GITHUB}" target="_blank">GitHub</a>
            <a class="sidebar-link" href="{DEV_PORTFOLIO}" target="_blank">Portfolio</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Executive Overview",
    "Customer Segments",
    "Churn Risk & Retention",
    "Customer Lookup",
    "Revenue & Products",
    "Action Plan",
    "Contact Developer",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-heading">Key Performance Indicators</div>', unsafe_allow_html=True)

    total_customers   = len(df)
    total_revenue     = df["total_revenue"].sum()
    avg_rev           = df["total_revenue"].mean()
    vip_count         = df["vip_flag"].sum()
    retention_count   = df["retention_target_flag"].sum()
    scored_count      = (df["model_scored"] == 1).sum()
    crit_high_count   = df["churn_risk_tier"].isin(["Critical Risk", "High Risk"]).sum()
    rev_excellent_healthy = df.loc[df["health_tier"].isin(["Excellent", "Healthy"]), "total_revenue"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Customers", format_number(total_customers)), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Total Revenue", format_currency(total_revenue)), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Revenue / Customer", format_currency(avg_rev)), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("VIP Customers", format_number(vip_count),
                             f"{vip_count/total_customers*100:.1f}% of customer base"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(kpi_card("Retention Targets", format_number(retention_count),
                             f"{retention_count/total_customers*100:.1f}% need attention"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("Model-Scored Customers", format_number(scored_count),
                             f"{scored_count/total_customers*100:.1f}% of selected"), unsafe_allow_html=True)
    with c7:
        st.markdown(kpi_card("Critical + High Risk", format_number(crit_high_count),
                             f"{crit_high_count/total_customers*100:.1f}% of selected"), unsafe_allow_html=True)
    with c8:
        pct_eh = rev_excellent_healthy / total_revenue * 100 if total_revenue > 0 else 0
        st.markdown(kpi_card("Revenue: Excellent + Healthy", format_currency(rev_excellent_healthy),
                             f"{pct_eh:.1f}% of selected revenue"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-heading">Revenue by Health Tier</div>', unsafe_allow_html=True)
        rev_by_tier = (
            df.groupby("health_tier")["total_revenue"].sum()
            .reindex([t for t in HEALTH_TIER_ORDER if t in df["health_tier"].unique()], fill_value=0)
            .reset_index()
        )
        rev_by_tier.columns = ["health_tier", "total_revenue"]
        fig = px.bar(
            rev_by_tier, x="health_tier", y="total_revenue",
            color="health_tier", color_discrete_map=HEALTH_COLORS,
            text=rev_by_tier["total_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"health_tier": "Health Tier", "total_revenue": "Revenue (£)"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-heading">Customers by Health Tier</div>', unsafe_allow_html=True)
        cnt_by_tier = (
            df["health_tier"].value_counts()
            .reindex([t for t in HEALTH_TIER_ORDER if t in df["health_tier"].unique()], fill_value=0)
            .reset_index()
        )
        cnt_by_tier.columns = ["health_tier", "count"]
        fig2 = px.bar(
            cnt_by_tier, x="health_tier", y="count",
            color="health_tier", color_discrete_map=HEALTH_COLORS,
            text="count",
            template=PLOTLY_TEMPLATE,
            labels={"health_tier": "Health Tier", "count": "Customers"},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2
    col_left2, col_mid2, col_right2 = st.columns([2, 2, 1.5])

    with col_left2:
        st.markdown('<div class="section-heading">Revenue by Action Priority</div>', unsafe_allow_html=True)
        rev_by_prio = (
            df.groupby("action_priority")["total_revenue"].sum()
            .reindex([p for p in ACTION_PRIO_ORDER if p in df["action_priority"].unique()], fill_value=0)
            .reset_index()
        )
        rev_by_prio.columns = ["action_priority", "total_revenue"]
        fig3 = px.bar(
            rev_by_prio, x="action_priority", y="total_revenue",
            color="action_priority", color_discrete_map=PRIO_COLORS,
            text=rev_by_prio["total_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"action_priority": "Action Priority", "total_revenue": "Revenue (£)"},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, margin=dict(t=10, b=10), height=310)
        st.plotly_chart(fig3, use_container_width=True)

    with col_mid2:
        st.markdown('<div class="section-heading">Customers by Action Priority</div>', unsafe_allow_html=True)
        cnt_by_prio = (
            df["action_priority"].value_counts()
            .reindex([p for p in ACTION_PRIO_ORDER if p in df["action_priority"].unique()], fill_value=0)
            .reset_index()
        )
        cnt_by_prio.columns = ["action_priority", "count"]
        fig4 = px.bar(
            cnt_by_prio, x="action_priority", y="count",
            color="action_priority", color_discrete_map=PRIO_COLORS,
            text="count",
            template=PLOTLY_TEMPLATE,
            labels={"action_priority": "Action Priority", "count": "Customers"},
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, margin=dict(t=10, b=10), height=310)
        st.plotly_chart(fig4, use_container_width=True)

    with col_right2:
        st.markdown('<div class="section-heading">Churn Risk Distribution</div>', unsafe_allow_html=True)
        churn_cnt = (
            df["churn_risk_tier"].value_counts()
            .reindex([c for c in CHURN_TIER_ORDER if c in df["churn_risk_tier"].unique()], fill_value=0)
            .reset_index()
        )
        churn_cnt.columns = ["churn_risk_tier", "count"]
        fig5 = px.pie(
            churn_cnt, names="churn_risk_tier", values="count",
            color="churn_risk_tier", color_discrete_map=CHURN_COLORS,
            template=PLOTLY_TEMPLATE,
            hole=0.42,
        )
        fig5.update_traces(textposition="inside", textinfo="percent+label")
        fig5.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=310,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Auto-generated business insights
    st.markdown('<div class="section-heading">Business Insights</div>', unsafe_allow_html=True)

    pct_rev_top = rev_excellent_healthy / total_revenue * 100 if total_revenue > 0 else 0
    pct_retention = retention_count / total_customers * 100 if total_customers > 0 else 0
    p4_rev = df.loc[df["action_priority"] == "Priority 4", "total_revenue"].sum()
    pct_p4 = p4_rev / total_revenue * 100 if total_revenue > 0 else 0
    not_scored_count = (df["model_scored"] == 0).sum()
    pct_crit = df["churn_risk_tier"].eq("Critical Risk").sum() / total_customers * 100

    insights = [
        f"Excellent and Healthy customers represent <b>{pct_rev_top:.1f}%</b> of revenue "
        f"({format_currency(rev_excellent_healthy)}) — protecting these accounts is the highest-priority action.",
        f"<b>{format_number(retention_count)}</b> customers ({pct_retention:.1f}%) are flagged as retention targets "
        f"based on churn risk, health tier, or RFM segment.",
        f"Priority 4 (Loyalty/Upsell) customers hold <b>{format_currency(p4_rev)}</b> "
        f"({pct_p4:.1f}% of revenue) — the highest revenue concentration across action priorities.",
        f"<b>{format_number(crit_high_count)}</b> customers are rated Critical or High churn risk "
        f"({pct_crit:.1f}% are in the Critical Risk tier alone).",
        f"<b>{format_number(not_scored_count)}</b> customers could not be scored by the churn model "
        f"(entered during the prediction window) — their churn component uses a neutral baseline.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    valid_segs = [s for s in RFM_SEG_ORDER if s in df["rfm_segment"].unique()]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-heading">Customers by RFM Segment</div>', unsafe_allow_html=True)
        seg_cnt = (
            df.groupby("rfm_segment").size()
            .reindex(valid_segs, fill_value=0)
            .reset_index()
        )
        seg_cnt.columns = ["rfm_segment", "count"]
        fig_s1 = px.bar(
            seg_cnt, y="rfm_segment", x="count",
            orientation="h", text="count",
            template=PLOTLY_TEMPLATE,
            labels={"rfm_segment": "", "count": "Customers"},
            color="rfm_segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_s1.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
        fig_s1.update_traces(textposition="outside")
        st.plotly_chart(fig_s1, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-heading">Revenue by RFM Segment</div>', unsafe_allow_html=True)
        seg_rev = (
            df.groupby("rfm_segment")["total_revenue"].sum()
            .reindex(valid_segs, fill_value=0)
            .reset_index()
        )
        seg_rev.columns = ["rfm_segment", "total_revenue"]
        fig_s2 = px.bar(
            seg_rev, y="rfm_segment", x="total_revenue",
            orientation="h",
            text=seg_rev["total_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"rfm_segment": "", "total_revenue": "Revenue (£)"},
            color="rfm_segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_s2.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
        fig_s2.update_traces(textposition="outside")
        st.plotly_chart(fig_s2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-heading">Avg Revenue per Customer by Segment</div>', unsafe_allow_html=True)
        seg_avg = (
            df.groupby("rfm_segment")["total_revenue"].mean()
            .reindex(valid_segs, fill_value=0)
            .reset_index()
        )
        seg_avg.columns = ["rfm_segment", "avg_revenue"]
        fig_s3 = px.bar(
            seg_avg, y="rfm_segment", x="avg_revenue",
            orientation="h",
            text=seg_avg["avg_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"rfm_segment": "", "avg_revenue": "Avg Revenue (£)"},
            color="avg_revenue",
            color_continuous_scale="Blues",
        )
        fig_s3.update_layout(showlegend=False, coloraxis_showscale=False, height=380, margin=dict(t=10, b=10))
        fig_s3.update_traces(textposition="outside")
        st.plotly_chart(fig_s3, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-heading">Health Tier by RFM Segment</div>', unsafe_allow_html=True)
        stacked = (
            df.groupby(["rfm_segment", "health_tier"])
            .size()
            .reset_index(name="count")
        )
        stacked = stacked[stacked["rfm_segment"].isin(valid_segs)]
        fig_s4 = px.bar(
            stacked, y="rfm_segment", x="count",
            color="health_tier", orientation="h",
            color_discrete_map=HEALTH_COLORS,
            category_orders={"rfm_segment": valid_segs, "health_tier": HEALTH_TIER_ORDER},
            template=PLOTLY_TEMPLATE,
            labels={"rfm_segment": "", "count": "Customers", "health_tier": "Health Tier"},
        )
        fig_s4.update_layout(height=380, margin=dict(t=10, b=10), legend_title_text="Health Tier")
        st.plotly_chart(fig_s4, use_container_width=True)

    col_l3, col_r3 = st.columns([1, 1])
    with col_l3:
        st.markdown('<div class="section-heading">Customer Value Tier Distribution</div>', unsafe_allow_html=True)
        vt_cnt = (
            df["customer_value_tier"].value_counts()
            .reindex([v for v in VALUE_TIER_ORDER if v in df["customer_value_tier"].unique()], fill_value=0)
            .reset_index()
        )
        vt_cnt.columns = ["customer_value_tier", "count"]
        fig_vt = px.pie(
            vt_cnt, names="customer_value_tier", values="count",
            color_discrete_sequence=["#1d4ed8", "#3b82f6", "#93c5fd", "#dbeafe"],
            template=PLOTLY_TEMPLATE, hole=0.4,
        )
        fig_vt.update_traces(textinfo="percent+label")
        fig_vt.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_vt, use_container_width=True)

    with col_r3:
        st.markdown('<div class="section-heading">Segment Summary Table</div>', unsafe_allow_html=True)
        seg_summary = (
            df.groupby("rfm_segment")
            .agg(
                customer_count        = ("customer_id",        "count"),
                total_revenue         = ("total_revenue",      "sum"),
                avg_revenue           = ("total_revenue",      "mean"),
                avg_churn_probability = ("churn_probability",  "mean"),
                retention_targets     = ("retention_target_flag", "sum"),
                vip_count             = ("vip_flag",           "sum"),
            )
            .reindex(valid_segs)
            .reset_index()
        )
        seg_summary["total_revenue"]         = seg_summary["total_revenue"].apply(format_currency)
        seg_summary["avg_revenue"]           = seg_summary["avg_revenue"].apply(format_currency)
        seg_summary["avg_churn_probability"] = seg_summary["avg_churn_probability"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        seg_summary.columns = [
            "Segment", "Customers", "Total Revenue", "Avg Revenue",
            "Avg Churn Prob", "Retention Targets", "VIP Count",
        ]
        st.dataframe(seg_summary, hide_index=True, use_container_width=True, height=300)

    # Business insight
    top_seg      = df.groupby("rfm_segment")["total_revenue"].sum().idxmax()
    top_seg_rev  = df.groupby("rfm_segment")["total_revenue"].sum().max()
    retain_segs  = df[df["rfm_segment"].isin(["At Risk", "Cannot Lose"])].shape[0]
    st.markdown(
        f'<div class="insight-box">'
        f'The <b>{top_seg}</b> segment generates the highest revenue at <b>{format_currency(top_seg_rev)}</b>. '
        f'<b>{format_number(retain_segs)}</b> customers are in high-priority retention segments '
        f'(At Risk + Cannot Lose) and require immediate outreach.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHURN RISK & RETENTION
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        '<div class="insight-box">Churn predictions are available only for customers eligible in the '
        'observation window (obs_cutoff = 2011-07-01). Prediction-window-only customers are labeled '
        '<b>Not Scored</b> and excluded from probability charts below.</div>',
        unsafe_allow_html=True,
    )

    scored_df = df[df["model_scored"] == 1].copy()

    if len(scored_df) == 0:
        st.markdown(
            '<div class="warning-box">No scored customers in current filter. Adjust filters to include scored customers.</div>',
            unsafe_allow_html=True,
        )
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-heading">Churn Probability Distribution</div>', unsafe_allow_html=True)
            fig_c1 = px.histogram(
                scored_df, x="churn_probability", nbins=40,
                color_discrete_sequence=["#3b82f6"],
                template=PLOTLY_TEMPLATE,
                labels={"churn_probability": "Churn Probability", "count": "Customers"},
            )
            fig_c1.update_layout(bargap=0.05, height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_c1, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-heading">Churn Risk Tier — Customer Count</div>', unsafe_allow_html=True)
            valid_churn_tiers = [t for t in CHURN_TIER_ORDER if t in df["churn_risk_tier"].unique()]
            churn_tier_cnt = (
                df["churn_risk_tier"].value_counts()
                .reindex(valid_churn_tiers, fill_value=0)
                .reset_index()
            )
            churn_tier_cnt.columns = ["churn_risk_tier", "count"]
            fig_c2 = px.bar(
                churn_tier_cnt, x="churn_risk_tier", y="count",
                color="churn_risk_tier", color_discrete_map=CHURN_COLORS,
                text="count",
                template=PLOTLY_TEMPLATE,
                labels={"churn_risk_tier": "Risk Tier", "count": "Customers"},
            )
            fig_c2.update_traces(textposition="outside")
            fig_c2.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_c2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown('<div class="section-heading">Churn Risk Tier — Revenue at Stake</div>', unsafe_allow_html=True)
            churn_tier_rev = (
                df.groupby("churn_risk_tier")["total_revenue"].sum()
                .reindex(valid_churn_tiers, fill_value=0)
                .reset_index()
            )
            churn_tier_rev.columns = ["churn_risk_tier", "total_revenue"]
            fig_c3 = px.bar(
                churn_tier_rev, x="churn_risk_tier", y="total_revenue",
                color="churn_risk_tier", color_discrete_map=CHURN_COLORS,
                text=churn_tier_rev["total_revenue"].apply(format_currency),
                template=PLOTLY_TEMPLATE,
                labels={"churn_risk_tier": "Risk Tier", "total_revenue": "Revenue (£)"},
            )
            fig_c3.update_traces(textposition="outside")
            fig_c3.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_c3, use_container_width=True)

        with col_r2:
            st.markdown('<div class="section-heading">Health Score vs Churn Probability</div>', unsafe_allow_html=True)
            fig_c4 = px.scatter(
                scored_df,
                x="churn_probability", y="customer_health_score",
                color="health_tier", color_discrete_map=HEALTH_COLORS,
                hover_data=["customer_id", "rfm_segment", "total_revenue"],
                opacity=0.5,
                template=PLOTLY_TEMPLATE,
                labels={
                    "churn_probability": "Churn Probability",
                    "customer_health_score": "Health Score",
                    "health_tier": "Health Tier",
                },
                category_orders={"health_tier": HEALTH_TIER_ORDER},
            )
            fig_c4.update_layout(height=300, margin=dict(t=10, b=10), legend_title_text="Health Tier")
            st.plotly_chart(fig_c4, use_container_width=True)

        # Top churn risk table
        st.markdown('<div class="section-heading">Top 20 Highest Churn Probability Customers</div>', unsafe_allow_html=True)
        top_risk_cols = [
            "customer_id", "country_mode", "total_revenue", "rfm_segment",
            "health_tier", "churn_probability", "churn_risk_tier",
            "action_priority", "final_recommended_action",
        ]
        top_risk = (
            scored_df.nlargest(20, "churn_probability")[top_risk_cols].copy()
        )
        top_risk["total_revenue"]    = top_risk["total_revenue"].apply(format_currency)
        top_risk["churn_probability"] = top_risk["churn_probability"].apply(lambda x: f"{x:.3f}")
        top_risk.columns = [
            "Customer ID", "Country", "Revenue", "RFM Segment", "Health Tier",
            "Churn Prob", "Risk Tier", "Action Priority", "Recommended Action",
        ]
        st.dataframe(top_risk, hide_index=True, use_container_width=True)

        # Feature importance if available
        if feat_imp is not None:
            st.markdown('<div class="section-heading">Top Churn Drivers (Logistic Regression Coefficients)</div>', unsafe_allow_html=True)
            top_pos = feat_imp[feat_imp["coefficient"] > 0].nlargest(10, "abs_coefficient")
            top_neg = feat_imp[feat_imp["coefficient"] < 0].nlargest(10, "abs_coefficient")
            fi_plot = pd.concat([top_pos, top_neg]).sort_values("coefficient")
            fi_plot["color"] = fi_plot["coefficient"].apply(lambda x: "#ef4444" if x > 0 else "#22c55e")
            fig_fi = go.Figure(go.Bar(
                x=fi_plot["coefficient"],
                y=fi_plot["feature"],
                orientation="h",
                marker_color=fi_plot["color"],
                text=fi_plot["coefficient"].apply(lambda x: f"{x:+.3f}"),
                textposition="outside",
            ))
            fig_fi.update_layout(
                template=PLOTLY_TEMPLATE,
                height=460,
                margin=dict(t=10, b=10),
                xaxis_title="Coefficient (positive = higher churn risk)",
                yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-heading">Customer 360 Profile Lookup</div>', unsafe_allow_html=True)

    lookup_ids = sorted(df["customer_id"].unique().tolist())
    selected_id = st.selectbox(
        "Select Customer ID",
        options=lookup_ids,
        format_func=lambda x: f"Customer {x}",
    )

    cust = df[df["customer_id"] == selected_id].iloc[0]

    # Header
    is_vip     = cust["vip_flag"] == 1
    is_target  = cust["retention_target_flag"] == 1
    is_scored  = cust["model_scored"] == 1

    badges = []
    if is_vip:     badges.append('<span style="background:#1d4ed8;color:white;padding:3px 10px;border-radius:12px;font-size:0.78rem;margin-right:6px">VIP</span>')
    if is_target:  badges.append('<span style="background:#dc2626;color:white;padding:3px 10px;border-radius:12px;font-size:0.78rem;margin-right:6px">Retention Target</span>')
    if not is_scored: badges.append('<span style="background:#94a3b8;color:white;padding:3px 10px;border-radius:12px;font-size:0.78rem">Not Scored</span>')

    st.markdown(
        f'<div style="margin:8px 0 16px 0"><b>Customer {selected_id}</b> &nbsp; {"".join(badges)}</div>',
        unsafe_allow_html=True,
    )

    # KPI row 1
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Country", str(cust["country_mode"])), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Total Revenue", format_currency(cust["total_revenue"])), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Total Invoices", format_number(cust["total_invoices"])), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Recency (days)", format_number(cust["recency_days"])), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # KPI row 2
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.markdown(kpi_card("RFM Segment", str(cust["rfm_segment"])), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card("RFM Score", str(cust["rfm_score"])), unsafe_allow_html=True)
    with k7:
        st.markdown(kpi_card("Health Score", f"{cust['customer_health_score']:.1f} / 100"), unsafe_allow_html=True)
    with k8:
        st.markdown(kpi_card("Health Tier", str(cust["health_tier"])), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # KPI row 3
    k9, k10, k11, k12 = st.columns(4)
    with k9:
        churn_prob_display = f"{cust['churn_probability']:.1%}" if is_scored else "Not Scored"
        st.markdown(kpi_card("Churn Probability", churn_prob_display), unsafe_allow_html=True)
    with k10:
        st.markdown(kpi_card("Churn Risk Tier", str(cust["churn_risk_tier"])), unsafe_allow_html=True)
    with k11:
        st.markdown(kpi_card("Customer Value Tier", str(cust["customer_value_tier"])), unsafe_allow_html=True)
    with k12:
        st.markdown(kpi_card("Action Priority", str(cust["action_priority"])), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Recommended action
    st.markdown(
        f'<div class="insight-box"><b>Recommended Action:</b> {cust["final_recommended_action"]}</div>',
        unsafe_allow_html=True,
    )

    # Not scored message
    if not is_scored:
        st.markdown(
            '<div class="warning-box">Churn model score is not available for this customer because they '
            'entered during the prediction window or were not eligible for training-label creation. '
            'The health score churn component uses a neutral midpoint (15/30).</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_info = st.columns([1, 1])

    with col_chart:
        st.markdown('<div class="section-heading">Customer Score Profile</div>', unsafe_allow_html=True)

        r = float(cust["r_score"])
        f = float(cust["f_score"])
        m = float(cust["m_score"])
        health_norm = float(cust["customer_health_score"]) / 20  # scale to 0-5
        churn_safety = (1 - float(cust["churn_probability"])) * 5 if is_scored else 2.5

        radar_cats   = ["Recency (R)", "Frequency (F)", "Monetary (M)", "Health Score", "Churn Safety"]
        radar_vals   = [r, f, m, health_norm, churn_safety]
        radar_vals_c = radar_vals + [radar_vals[0]]
        radar_cats_c = radar_cats + [radar_cats[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals_c,
            theta=radar_cats_c,
            fill="toself",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
            ),
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-heading">Purchase & Return Details</div>', unsafe_allow_html=True)

        details = {
            "First Purchase Date":       str(cust["first_purchase_date"])[:10],
            "Last Purchase Date":        str(cust["last_purchase_date"])[:10],
            "Tenure (days)":             format_number(cust["customer_tenure_days"]),
            "Unique Purchase Days":      format_number(cust["unique_purchase_days"]),
            "Avg Days Between Purchases": f"{cust['avg_days_between_purchases']:.1f}",
            "Purchase Freq / Month":     f"{cust['purchase_frequency_per_month']:.2f}",
            "Unique Products":           format_number(cust["unique_products"]),
            "Avg Invoice Value":         format_currency(cust["average_invoice_value"]),
            "Max Invoice Value":         format_currency(cust["max_invoice_value"]),
            "Has Returned":              "Yes" if cust["has_returned"] == 1 else "No",
            "Total Returns":             format_number(cust["total_returns"]),
            "Return Rate":               f"{cust['return_rate']:.1%}",
        }
        detail_df = pd.DataFrame(list(details.items()), columns=["Metric", "Value"])
        st.dataframe(detail_df, hide_index=True, use_container_width=True, height=360)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — REVENUE & PRODUCTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if txn_df is None:
        st.markdown(
            '<div class="warning-box">clean_transactions.parquet not found. '
            'Revenue trend and product charts are unavailable.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Monthly revenue trend
        monthly_rev = (
            txn_df.groupby("invoice_yearmonth")
            .agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"))
            .reset_index()
            .sort_values("invoice_yearmonth")
        )

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-heading">Monthly Revenue Trend</div>', unsafe_allow_html=True)
            fig_t1 = px.line(
                monthly_rev, x="invoice_yearmonth", y="revenue",
                markers=True,
                template=PLOTLY_TEMPLATE,
                labels={"invoice_yearmonth": "Month", "revenue": "Revenue (£)"},
            )
            fig_t1.update_traces(line_color="#3b82f6", line_width=2)
            fig_t1.update_layout(height=310, margin=dict(t=10, b=10), xaxis_tickangle=-45)
            st.plotly_chart(fig_t1, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-heading">Monthly Active Customers</div>', unsafe_allow_html=True)
            fig_t2 = px.bar(
                monthly_rev, x="invoice_yearmonth", y="customers",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=["#6366f1"],
                labels={"invoice_yearmonth": "Month", "customers": "Active Customers"},
            )
            fig_t2.update_layout(height=310, margin=dict(t=10, b=10), xaxis_tickangle=-45)
            st.plotly_chart(fig_t2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown('<div class="section-heading">Top 10 Countries by Revenue</div>', unsafe_allow_html=True)
            country_rev = (
                txn_df.groupby("country")["revenue"].sum()
                .nlargest(10)
                .reset_index()
            )
            country_rev.columns = ["country", "revenue"]
            fig_t3 = px.bar(
                country_rev, y="country", x="revenue",
                orientation="h",
                text=country_rev["revenue"].apply(format_currency),
                color="revenue", color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                labels={"country": "", "revenue": "Revenue (£)"},
            )
            fig_t3.update_traces(textposition="outside")
            fig_t3.update_layout(showlegend=False, coloraxis_showscale=False,
                                  height=360, margin=dict(t=10, b=10))
            st.plotly_chart(fig_t3, use_container_width=True)

        with col_r2:
            st.markdown('<div class="section-heading">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
            prod_rev = (
                txn_df.groupby("description")["revenue"].sum()
                .nlargest(10)
                .reset_index()
            )
            prod_rev.columns = ["description", "revenue"]
            prod_rev["description"] = prod_rev["description"].str[:40]
            fig_t4 = px.bar(
                prod_rev, y="description", x="revenue",
                orientation="h",
                text=prod_rev["revenue"].apply(format_currency),
                color="revenue", color_continuous_scale="Greens",
                template=PLOTLY_TEMPLATE,
                labels={"description": "", "revenue": "Revenue (£)"},
            )
            fig_t4.update_traces(textposition="outside")
            fig_t4.update_layout(showlegend=False, coloraxis_showscale=False,
                                  height=360, margin=dict(t=10, b=10))
            st.plotly_chart(fig_t4, use_container_width=True)

        # Avg order value trend
        st.markdown('<div class="section-heading">Average Order Value Trend</div>', unsafe_allow_html=True)
        aov = (
            txn_df.groupby(["invoice_yearmonth", "invoice_id"])["revenue"].sum()
            .reset_index()
            .groupby("invoice_yearmonth")["revenue"].mean()
            .reset_index()
            .rename(columns={"revenue": "avg_order_value"})
            .sort_values("invoice_yearmonth")
        )
        fig_t5 = px.line(
            aov, x="invoice_yearmonth", y="avg_order_value",
            markers=True, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#f59e0b"],
            labels={"invoice_yearmonth": "Month", "avg_order_value": "Avg Order Value (£)"},
        )
        fig_t5.update_traces(line_width=2)
        fig_t5.update_layout(height=280, margin=dict(t=10, b=10), xaxis_tickangle=-45)
        st.plotly_chart(fig_t5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — ACTION PLAN
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-heading">Customer Count by Action Priority</div>', unsafe_allow_html=True)
        ap_cnt = (
            df["action_priority"].value_counts()
            .reindex([p for p in ACTION_PRIO_ORDER if p in df["action_priority"].unique()], fill_value=0)
            .reset_index()
        )
        ap_cnt.columns = ["action_priority", "count"]
        fig_ap1 = px.bar(
            ap_cnt, x="action_priority", y="count",
            color="action_priority", color_discrete_map=PRIO_COLORS,
            text="count", template=PLOTLY_TEMPLATE,
            labels={"action_priority": "Priority", "count": "Customers"},
        )
        fig_ap1.update_traces(textposition="outside")
        fig_ap1.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_ap1, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-heading">Revenue at Stake by Action Priority</div>', unsafe_allow_html=True)
        ap_rev = (
            df.groupby("action_priority")["total_revenue"].sum()
            .reindex([p for p in ACTION_PRIO_ORDER if p in df["action_priority"].unique()], fill_value=0)
            .reset_index()
        )
        ap_rev.columns = ["action_priority", "total_revenue"]
        fig_ap2 = px.bar(
            ap_rev, x="action_priority", y="total_revenue",
            color="action_priority", color_discrete_map=PRIO_COLORS,
            text=ap_rev["total_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"action_priority": "Priority", "total_revenue": "Revenue (£)"},
        )
        fig_ap2.update_traces(textposition="outside")
        fig_ap2.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_ap2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-heading">Retention Target Revenue by Health Tier</div>', unsafe_allow_html=True)
        ret_rev = (
            df[df["retention_target_flag"] == 1]
            .groupby("health_tier")["total_revenue"].sum()
            .reindex([t for t in HEALTH_TIER_ORDER if t in df["health_tier"].unique()], fill_value=0)
            .reset_index()
        )
        ret_rev.columns = ["health_tier", "total_revenue"]
        fig_ap3 = px.bar(
            ret_rev, x="health_tier", y="total_revenue",
            color="health_tier", color_discrete_map=HEALTH_COLORS,
            text=ret_rev["total_revenue"].apply(format_currency),
            template=PLOTLY_TEMPLATE,
            labels={"health_tier": "Health Tier", "total_revenue": "Revenue (£)"},
        )
        fig_ap3.update_traces(textposition="outside")
        fig_ap3.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_ap3, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-heading">VIP Customer Revenue by RFM Segment</div>', unsafe_allow_html=True)
        vip_rev = (
            df[df["vip_flag"] == 1]
            .groupby("rfm_segment")["total_revenue"].sum()
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )
        vip_rev.columns = ["rfm_segment", "total_revenue"]
        fig_ap4 = px.bar(
            vip_rev, y="rfm_segment", x="total_revenue",
            orientation="h",
            text=vip_rev["total_revenue"].apply(format_currency),
            color="total_revenue", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
            labels={"rfm_segment": "", "total_revenue": "Revenue (£)"},
        )
        fig_ap4.update_traces(textposition="outside")
        fig_ap4.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_ap4, use_container_width=True)

    # Action plan table
    st.markdown('<div class="section-heading">Action Plan Summary</div>', unsafe_allow_html=True)
    ap_summary = (
        df.groupby("action_priority")
        .agg(
            customer_count        = ("customer_id",        "count"),
            total_revenue         = ("total_revenue",      "sum"),
            avg_churn_probability = ("churn_probability",  "mean"),
        )
        .reindex([p for p in ACTION_PRIO_ORDER if p in df["action_priority"].unique()])
        .reset_index()
    )
    # Map strategy from the first customer per priority
    strat_map = (
        df.groupby("action_priority")["final_recommended_action"].first()
    )
    ap_summary["recommended_strategy"] = ap_summary["action_priority"].map(strat_map)
    ap_summary["total_revenue"]         = ap_summary["total_revenue"].apply(format_currency)
    ap_summary["avg_churn_probability"] = ap_summary["avg_churn_probability"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    ap_summary.columns = [
        "Action Priority", "Customers", "Total Revenue",
        "Avg Churn Prob", "Recommended Strategy",
    ]
    st.dataframe(ap_summary, hide_index=True, use_container_width=True)

    # Download buttons
    st.markdown('<div class="section-heading">Export Data</div>', unsafe_allow_html=True)
    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        csv_filtered = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Customer List",
            data=csv_filtered,
            file_name="filtered_customers.csv",
            mime="text/csv",
        )

    with dl2:
        high_risk_df = df[df["churn_risk_tier"].isin(["High Risk", "Critical Risk"])]
        csv_highrisk = high_risk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download High-Risk Customers",
            data=csv_highrisk,
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )

    with dl3:
        vip_df = df[df["vip_flag"] == 1]
        csv_vip = vip_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download VIP Customers",
            data=csv_vip,
            file_name="vip_customers.csv",
            mime="text/csv",
        )

    with dl4:
        ap_export = (
            df.groupby("action_priority")
            .agg(
                customer_count=("customer_id", "count"),
                total_revenue=("total_revenue", "sum"),
                avg_churn_probability=("churn_probability", "mean"),
            )
            .reset_index()
        )
        csv_ap = ap_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Action Plan",
            data=csv_ap,
            file_name="action_plan.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — CONTACT DEVELOPER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    # Intro
    st.markdown(f"""
    <div style="max-width:720px;margin:0 auto 28px auto">
        <h2 style="font-size:1.5rem;font-weight:700;color:#1a1a2e;margin-bottom:6px">
            Get in Touch
        </h2>
        <p style="color:#4b5563;font-size:0.95rem;line-height:1.6;margin:0">
            Interested in discussing this project, collaboration, or data analytics opportunities?
            I'd love to hear from you. Use the form below or reach out directly through any of
            the channels listed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Contact cards row
    cc1, cc2, cc3, cc4 = st.columns(4)
    for col, icon, title, label, href in [
        (cc1, "📩", "Email",     DEV_EMAIL,                   f"mailto:{DEV_EMAIL}"),
        (cc2, "💼", "LinkedIn",  "linkedin.com/in/prajwalshekar", DEV_LINKEDIN),
        (cc3, "🐙", "GitHub",    "PrajwalShekar22",             DEV_GITHUB),
        (cc4, "🌐", "Portfolio", "datascienceportfol.io/pgc", DEV_PORTFOLIO),
    ]:
        with col:
            st.markdown(f"""
            <div class="contact-card">
                <div class="contact-card-icon">{icon}</div>
                <div class="contact-card-title">{title}</div>
                <a class="contact-card-link" href="{href}" target="_blank">{label}</a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Contact form
    st.markdown('<div class="section-heading">Send a Message</div>', unsafe_allow_html=True)
    st.caption(
        "Fill in the form below. On submit, you'll get a direct mailto link that opens your "
        "email client with the message pre-filled — no third-party service required."
    )

    with st.form("contact_form", clear_on_submit=False):
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            contact_name = st.text_input("Your Name *")
        with form_col2:
            contact_email = st.text_input("Your Email *")

        contact_subject = st.text_input("Subject *")
        contact_message = st.text_area(
            "Message * (max 150 words)",
            height=150,
            placeholder="Write your message here...",
        )

        # Live word count display (static — updates on rerun)
        word_count = len(contact_message.split()) if contact_message.strip() else 0
        wc_color = "#dc2626" if word_count > 150 else "#166534"
        st.markdown(
            f'<p style="font-size:0.8rem;color:{wc_color};margin-top:-8px">'
            f'Message word count: <b>{word_count} / 150</b></p>',
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button("Send Message", use_container_width=False)

    # Validation & mailto generation (outside the form, triggered by submit)
    if submitted:
        errors = []
        if not contact_name.strip():
            errors.append("Name is required.")
        if not contact_email.strip():
            errors.append("Email is required.")
        elif "@" not in contact_email or "." not in contact_email:
            errors.append("Please enter a valid email address.")
        if not contact_subject.strip():
            errors.append("Subject is required.")
        if not contact_message.strip():
            errors.append("Message is required.")
        elif word_count > 150:
            errors.append(f"Message is {word_count} words — please keep it under 150 words.")

        if errors:
            for err in errors:
                st.markdown(
                    f'<div class="warning-box">⚠️ {err}</div>',
                    unsafe_allow_html=True,
                )
        else:
            import urllib.parse
            subject_enc = urllib.parse.quote(contact_subject)
            body_text   = (
                f"Hi Prajwal,\n\nMy name is {contact_name} ({contact_email}).\n\n"
                f"{contact_message}\n\nBest regards,\n{contact_name}"
            )
            body_enc = urllib.parse.quote(body_text)
            mailto_url = f"mailto:{DEV_EMAIL}?subject={subject_enc}&body={body_enc}"

            st.markdown(
                f'<div class="success-box">'
                f'✅ Thanks, <b>{contact_name}</b>! Your message is ready.<br><br>'
                f'For now, please email me directly at '
                f'<a href="mailto:{DEV_EMAIL}" style="color:#166534;font-weight:600">'
                f'{DEV_EMAIL}</a> or connect with me on '
                f'<a href="{DEV_LINKEDIN}" target="_blank" style="color:#166534;font-weight:600">'
                f'LinkedIn</a>.<br><br>'
                f'<a href="{mailto_url}" style="display:inline-block;background:#1d4ed8;'
                f'color:white;padding:8px 20px;border-radius:6px;text-decoration:none;'
                f'font-weight:600;font-size:0.88rem">📩 Open in Email Client</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Also show read-only message preview
            with st.expander("Preview your message"):
                st.text(body_text)

    # Note about Formspree (Option B hook for future)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Note: This form uses a client-side mailto link (no server required). "
        "For automated email delivery, a Formspree or SMTP integration can be added at deployment time."
    )


# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="site-footer">
    <div class="footer-name">{DEV_NAME}</div>
    <div class="footer-links">
        <a class="footer-link" href="{DEV_GITHUB}" target="_blank">GitHub</a>
        <a class="footer-link" href="{DEV_LINKEDIN}" target="_blank">LinkedIn</a>
        <a class="footer-link" href="{DEV_PORTFOLIO}" target="_blank">Portfolio</a>
        <a class="footer-link" href="mailto:{DEV_EMAIL}">Email</a>
    </div>
    <div class="footer-note">
        Built and maintained by {DEV_NAME} © 2026. Portfolio project for demonstration purposes.
    </div>
    <div class="footer-note">
        All monetary values are shown in GBP (£), based on the original UK retailer transaction
        dataset. Country filters customer location only.
    </div>
    <div class="footer-note">
        This project is built for portfolio and educational demonstration.
        Insights are based on the UCI Online Retail II public historical dataset.
    </div>
</div>
""", unsafe_allow_html=True)
