"""Mortgage Default Prediction — Interactive Dashboard.

Reads the project's already-generated output files only.
No model retraining. No raw performance file loaded into pandas.
No fabricated or hard-coded results.

Run (from project root, venv activated):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── make the project root importable when launched from any cwd ──────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.data_loader import (
    load_balance_proxy,
    load_data_quality,
    load_descriptive_statistics,
    load_model_metrics,
    load_target_distribution,
    load_test_predictions,
    load_validation_threshold_metrics,
)
from dashboard import metrics as M

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dashboard.app")

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mortgage Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS: dark fintech theme ──────────────────────────────────────────
DARK_CSS = """
<style>
/* main background */
.stApp { background-color: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1c2333 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px 14px 22px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.kpi-label { font-size: 0.80rem; color: #8b949e; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
.kpi-sub   { font-size: 0.74rem; color: #8b949e; margin-top: 4px; }

/* Section headers */
h1, h2, h3 { color: #e6edf3 !important; }

/* Divider */
hr { border-color: #30363d; }

/* Table */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* Info / warning / error banners */
.stAlert { border-radius: 8px; }

/* metric delta hide arrow when not needed */
[data-testid="stMetricDelta"] svg { display: none; }

/* streamlit tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px;
    padding: 6px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_DARK = "plotly_dark"
ACCENT = "#1f6feb"       # primary blue
ACCENT2 = "#58a6ff"      # lighter blue
WARN = "#f97316"         # orange for defaults/risk
SAFE = "#3fb950"         # green for non-defaults
NEUTRAL = "#8b949e"

# ── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading model metrics…")
def _metrics() -> dict:
    return load_model_metrics()

@st.cache_data(ttl=3600, show_spinner="Loading validation table…")
def _val_metrics() -> pd.DataFrame:
    return load_validation_threshold_metrics()

@st.cache_data(ttl=3600, show_spinner="Loading test predictions…")
def _predictions() -> pd.DataFrame:
    return load_test_predictions()

@st.cache_data(ttl=3600, show_spinner="Loading data-quality report…")
def _data_quality() -> dict:
    return load_data_quality()

@st.cache_data(ttl=3600, show_spinner="Loading target distribution…")
def _target_dist() -> pd.DataFrame:
    return load_target_distribution()

@st.cache_data(ttl=3600, show_spinner="Loading balance proxy table…")
def _balance_proxy() -> pd.DataFrame:
    return load_balance_proxy()

@st.cache_data(ttl=3600, show_spinner="Loading descriptive statistics…")
def _desc_stats() -> pd.DataFrame:
    return load_descriptive_statistics()

# ── Helper utilities ─────────────────────────────────────────────────────────
def _kpi(label: str, value: str, sub: str = "", colour: str = ACCENT2) -> None:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{colour}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _fmt_pct(v, decimals: int = 2) -> str:
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_f(v, decimals: int = 4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _missing_banner(section: str, reason: str) -> None:
    st.warning(
        f"**{section}** is unavailable — {reason}. "
        "Run `python run_project.py` from the project root to generate outputs.",
        icon="⚠️",
    )


_OCC_LABELS = {"P": "Principal (Owner-occupied)", "S": "Second Home",
               "I": "Investment / Non-owner", "O": "Other"}
_PURPOSE_LABELS = {"P": "Purchase", "C": "Cash-out Refinance",
                   "N": "No-cash-out Refinance", "R": "Refinance (Unspecified)"}

def _label_cat(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(series)


def _plotly_conf(fig: go.Figure) -> go.Figure:
    """Apply consistent dark-theme styling to any figure."""
    fig.update_layout(
        template=PLOTLY_DARK,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font={"color": "#e6edf3", "family": "system-ui, sans-serif"},
        title_font_size=15,
        margin={"t": 50, "b": 40, "l": 10, "r": 10},
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Mortgage Default\nPrediction Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        options=[
            "Executive Overview",
            "Loan Risk Analysis",
            "Model Performance",
            "Prediction Explorer",
            "Project Information",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>"
        "Model: Class-Weighted Logistic Regression<br>"
        "Data: Freddie Mac 2022 Sample<br>"
        "Threshold: 0.50 (validation-selected)<br>"
        "**No live retraining**"
        "</small>",
        unsafe_allow_html=True,
    )

# Load all data once (cached)
mm = _metrics()
preds = _predictions()
val_df = _val_metrics()
dq = _data_quality()
target_df = _target_dist()
balance_df = _balance_proxy()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown(
        "Key figures drawn directly from the most recent pipeline run. "
        "No values are fabricated or hard-coded."
    )

    if not mm:
        _missing_banner("Executive Overview", "model_metrics.json not found")
        st.stop()

    # ── KPI row 1: dataset ───────────────────────────────────────────────────
    total_obs = dq.get("rows", mm.get("training_records", 0) +
                       mm.get("validation_records", 0) + mm.get("test_records", 0))
    total_defaults = None
    non_defaults = None
    if not target_df.empty and {"default", "count"}.issubset(target_df.columns):
        d_row = target_df[target_df["default"] == 1]
        nd_row = target_df[target_df["default"] == 0]
        total_defaults = int(d_row["count"].sum()) if not d_row.empty else None
        non_defaults = int(nd_row["count"].sum()) if not nd_row.empty else None

    prev = mm.get("test_default_prevalence", None)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi("Eligible Observations", f"{total_obs:,}" if total_obs else "—",
             "full dataset after censoring", ACCENT2)
    with k2:
        _kpi("Actual Defaults", f"{total_defaults:,}" if total_defaults else "—",
             "within 24-month horizon", WARN)
    with k3:
        _kpi("Default Prevalence", _fmt_pct(prev) if prev else "—",
             "test-set observed rate", WARN)
    with k4:
        _kpi("Always-Non-Default Acc.", _fmt_pct(mm.get("always_non_default_accuracy")),
             "baseline: recall = 0.00%", NEUTRAL)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI row 2: model ─────────────────────────────────────────────────────
    k5, k6, k7, k8, k9 = st.columns(5)
    with k5:
        _kpi("Accuracy", _fmt_pct(mm.get("accuracy")), "test set", ACCENT2)
    with k6:
        _kpi("Precision", _fmt_f(mm.get("precision")), "test set", ACCENT2)
    with k7:
        _kpi("Recall", _fmt_f(mm.get("recall")), "test set", ACCENT2)
    with k8:
        _kpi("F1 Score", _fmt_f(mm.get("f1")), "test set", ACCENT2)
    with k9:
        _kpi("ROC-AUC", _fmt_f(mm.get("roc_auc")), "continuous scores", SAFE)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col_a, col_b = st.columns(2)

    # ── Default vs Non-Default bar chart ────────────────────────────────────
    with col_a:
        st.subheader("Default vs Non-Default Distribution")
        st.caption("Historical outcomes across all 45,697 eligible loans.")
        if not target_df.empty and {"default", "count"}.issubset(target_df.columns):
            plot_df = target_df.copy()
            plot_df["label"] = plot_df["default"].map({0: "Non-Default", 1: "Default"})
            fig = px.bar(
                plot_df, x="label", y="count",
                color="label",
                color_discrete_map={"Default": WARN, "Non-Default": SAFE},
                text_auto=True,
                title="Loan Outcomes — Full Dataset",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Loan Count")
            st.plotly_chart(_plotly_conf(fig), width='stretch')
        else:
            _missing_banner("Default Distribution", "target_distribution.csv missing or empty")

    # ── Predicted probability distribution ───────────────────────────────────
    with col_b:
        st.subheader("Predicted Risk Score Distribution")
        st.caption(
            "Distribution of predicted default probabilities on the **test set**, "
            "coloured by actual outcome. Higher scores indicate higher estimated risk."
        )
        if not preds.empty and "predicted_default_probability" in preds.columns:
            fig2 = go.Figure()
            for outcome, colour, label in ((0, SAFE, "Actual Non-Default"),
                                           (1, WARN, "Actual Default")):
                sub = preds[preds["default"] == outcome]["predicted_default_probability"].dropna()
                if not sub.empty:
                    fig2.add_trace(go.Histogram(
                        x=sub, name=label, marker_color=colour,
                        opacity=0.75, nbinsx=40,
                        histnorm="probability density",
                    ))
            fig2.update_layout(
                barmode="overlay",
                title="Predicted Default Probability",
                xaxis_title="Predicted Probability",
                yaxis_title="Density",
                legend={"title": "Actual Outcome"},
            )
            st.plotly_chart(_plotly_conf(fig2), width='stretch')
        else:
            _missing_banner("Risk Distribution", "test predictions not found")

    # ── Test-partition summary ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Dataset Partition Summary")
    st.caption("Loan-level 70 / 15 / 15 split, randomly seeded; all loan IDs are disjoint across partitions.")
    p1, p2, p3 = st.columns(3)
    parts = {
        "Training": (mm.get("training_records", 0), "Used to fit the model"),
        "Validation": (mm.get("validation_records", 0), "Selected model & threshold"),
        "Test": (mm.get("test_records", 0), "Final evaluation (once)"),
    }
    total_parts = sum(v[0] for v in parts.values())
    for col, (name, (n, role)) in zip([p1, p2, p3], parts.items()):
        pct = f"{100*n/total_parts:.1f}%" if total_parts else "—"
        with col:
            _kpi(f"{name} Set", f"{n:,}", f"{role} · {pct} of total", ACCENT2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAN RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Loan Risk Analysis":
    st.title("Loan Risk Analysis")
    st.markdown(
        "Analysis of default risk across available loan characteristics. "
        "All statistics are computed from the **test-set predictions** "
        "(6,678 loans). Historical default rate = realised outcomes. "
        "Predicted probability = model estimate."
    )

    if preds.empty:
        _missing_banner("Loan Risk Analysis", "test predictions not found")
        st.stop()

    # ── Sidebar filters (only columns that exist) ─────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        st.caption("Applied to the test-prediction table below.")

        # Occupancy filter
        if "occupancy_status" in preds.columns:
            occ_opts = sorted(preds["occupancy_status"].dropna().unique())
            occ_labels = [_OCC_LABELS.get(o, o) for o in occ_opts]
            sel_occ_labels = st.multiselect(
                "Occupancy Status", occ_labels, default=occ_labels)
            sel_occ = [o for o, l in zip(occ_opts, occ_labels)
                       if l in sel_occ_labels]
        else:
            sel_occ = None

        # Loan purpose filter
        if "loan_purpose" in preds.columns:
            pur_opts = sorted(preds["loan_purpose"].dropna().unique())
            pur_labels = [_PURPOSE_LABELS.get(p, p) for p in pur_opts]
            sel_pur_labels = st.multiselect(
                "Loan Purpose", pur_labels, default=pur_labels)
            sel_pur = [p for p, l in zip(pur_opts, pur_labels)
                       if l in sel_pur_labels]
        else:
            sel_pur = None

        # UPB slider
        if "upb" in preds.columns:
            upb_min = float(preds["upb"].min())
            upb_max = float(preds["upb"].max())
            sel_upb = st.slider(
                "Original UPB ($)",
                min_value=int(upb_min), max_value=int(upb_max),
                value=(int(upb_min), int(upb_max)), step=10000)
        else:
            sel_upb = None

    # Apply filters
    mask = pd.Series(True, index=preds.index)
    if sel_occ is not None:
        mask &= preds["occupancy_status"].isin(sel_occ)
    if sel_pur is not None:
        mask &= preds["loan_purpose"].isin(sel_pur)
    if sel_upb is not None and "upb" in preds.columns:
        mask &= (preds["upb"] >= sel_upb[0]) & (preds["upb"] <= sel_upb[1])
    filtered = preds[mask].copy()

    n_total = len(filtered)
    n_default = int((filtered["default"] == 1).sum()) if "default" in filtered.columns else 0
    st.info(f"**{n_total:,}** loans shown after filters · **{n_default:,}** actual defaults "
            f"({100*n_default/n_total:.1f}% rate)" if n_total else "No loans match the current filters.")
    if filtered.empty:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Categorical Risk", "📈 Numeric Risk", "⚖️ Loan-Balance Proxy", "📋 Prediction Table"])

    # ── Tab 1: Categorical features ─────────────────────────────────────────
    with tab1:
        cat_cols = [c for c in ("occupancy_status", "loan_purpose") if c in filtered.columns]
        if not cat_cols:
            st.info("No categorical features available in prediction output.")
        else:
            for col in cat_cols:
                label_map = _OCC_LABELS if col == "occupancy_status" else _PURPOSE_LABELS
                grp = M.default_rate_by_group(filtered, col)
                if grp.empty:
                    continue
                grp[col] = _label_cat(grp[col], label_map)
                grp = grp.sort_values("observed_default_rate", ascending=False)
                col_title = col.replace("_", " ").title()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=grp[col], y=grp["loan_count"],
                    name="Loan Count", marker_color=ACCENT2,
                    yaxis="y", offsetgroup=1,
                ))
                fig.add_trace(go.Scatter(
                    x=grp[col],
                    y=grp["observed_default_rate"] * 100,
                    name="Default Rate (%)",
                    mode="lines+markers",
                    marker={"color": WARN, "size": 9},
                    line={"color": WARN, "width": 2},
                    yaxis="y2",
                ))
                fig.update_layout(
                    title=f"Loan Count & Default Rate by {col_title}",
                    yaxis={"title": "Loan Count", "showgrid": False},
                    yaxis2={"title": "Observed Default Rate (%)",
                            "overlaying": "y", "side": "right",
                            "showgrid": False, "ticksuffix": "%"},
                    legend={"orientation": "h", "y": 1.1},
                    barmode="group",
                    xaxis_title=col_title,
                )
                st.plotly_chart(_plotly_conf(fig), width='stretch')

    # ── Tab 2: Numeric features ──────────────────────────────────────────────
    with tab2:
        num_cols = [c for c in ("credit_score", "dti", "ltv", "cltv")
                    if c in filtered.columns]
        bins_cfg = {
            "credit_score": (
                [0, 620, 660, 700, 740, 780, 10000],
                ["<620", "620–659", "660–699", "700–739", "740–779", "780+"],
                "Credit Score Band",
            ),
            "dti": (
                [0, 20, 30, 36, 43, 50, 1000],
                ["<20%", "20–29%", "30–35%", "36–42%", "43–49%", "≥50%"],
                "Debt-to-Income (DTI) Band",
            ),
            "ltv": (
                [0, 60, 70, 80, 90, 97, 200],
                ["≤60%", "61–70%", "71–80%", "81–90%", "91–97%", ">97%"],
                "Loan-to-Value (LTV) Band",
            ),
            "cltv": (
                [0, 60, 70, 80, 90, 97, 200],
                ["≤60%", "61–70%", "71–80%", "81–90%", "91–97%", ">97%"],
                "Combined LTV (CLTV) Band",
            ),
        }
        if not num_cols:
            st.info("No numeric features available in prediction output.")
        else:
            pairs = [(num_cols[i], num_cols[i+1])
                     for i in range(0, len(num_cols)-1, 2)]
            if len(num_cols) % 2:
                pairs.append((num_cols[-1], None))
            for left_col, right_col in pairs:
                c1, c2 = st.columns(2)
                for col, container in ((left_col, c1), (right_col, c2)):
                    if col is None:
                        continue
                    bins, labels, title = bins_cfg.get(col, (None, None, col))
                    if bins is None:
                        continue
                    grp = M.numeric_band_default_rate(
                        filtered, col,
                        bins=[-np.inf] + bins[1:-1] + [np.inf],
                        labels=labels)
                    if grp.empty:
                        continue
                    with container:
                        fig = px.bar(
                            grp, x="band", y="observed_default_rate",
                            color="observed_default_rate",
                            color_continuous_scale=["#3fb950", "#f97316", "#ff4d4d"],
                            text_auto=".1%",
                            title=f"Observed Default Rate by {title}",
                        )
                        fig.update_traces(
                            texttemplate="%{y:.1%}", textposition="outside")
                        fig.update_layout(
                            yaxis_tickformat=".1%",
                            yaxis_title="Observed Default Rate",
                            xaxis_title=title,
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(_plotly_conf(fig), width='stretch')

        # Credit score vs probability scatter
        if "credit_score" in filtered.columns and \
                "predicted_default_probability" in filtered.columns:
            st.markdown("#### Credit Score vs Predicted Default Probability")
            st.caption(
                "Each point is one test-set loan. Colour = actual outcome. "
                "Lower credit scores generally correspond to higher model scores."
            )
            sample = filtered.sample(min(2000, len(filtered)), random_state=42)
            sample["outcome"] = sample["default"].map({0: "Non-Default", 1: "Default"})
            fig_sc = px.scatter(
                sample,
                x="credit_score",
                y="predicted_default_probability",
                color="outcome",
                color_discrete_map={"Default": WARN, "Non-Default": SAFE},
                opacity=0.5,
                title="Credit Score vs Predicted Default Probability",
            )
            fig_sc.update_layout(
                xaxis_title="Credit Score",
                yaxis_title="Predicted Default Probability",
            )
            st.plotly_chart(_plotly_conf(fig_sc), width='stretch')

    # ── Tab 3: Loan-balance proxy (UPB bands) ────────────────────────────────
    with tab3:
        st.markdown(
            "> ⚠️ **No borrower income data.** Freddie Mac's public sample layout "
            "does not include borrower income, occupation, or home-ownership. "
            "The bands below group loans by **original unpaid principal balance (UPB)** "
            "— a loan-size proxy, not an income proxy."
        )
        # Build from filtered test predictions (same as the saved table but respects filters)
        if "upb" in filtered.columns and "default" in filtered.columns:
            from dashboard.data_loader import balance_bands_from_predictions
            bb = balance_bands_from_predictions(filtered)
            if not bb.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig_b1 = px.bar(
                        bb, x="balance_band",
                        y="average_predicted_default_probability",
                        color="average_predicted_default_probability",
                        color_continuous_scale=["#3fb950", "#f97316", "#ff4d4d"],
                        text_auto=".3f",
                        title="Average Predicted Default Probability by UPB Band",
                    )
                    fig_b1.update_layout(
                        xaxis_title="Loan-Balance Band (UPB)",
                        yaxis_title="Avg. Predicted Default Probability",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(_plotly_conf(fig_b1), width='stretch')
                with col2:
                    fig_b2 = px.bar(
                        bb, x="balance_band",
                        y="observed_default_rate",
                        color="observed_default_rate",
                        color_continuous_scale=["#3fb950", "#f97316", "#ff4d4d"],
                        text_auto=".3f",
                        title="Observed Default Rate by UPB Band",
                    )
                    fig_b2.update_layout(
                        xaxis_title="Loan-Balance Band (UPB)",
                        yaxis_title="Observed Default Rate",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(_plotly_conf(fig_b2), width='stretch')

                st.markdown("**Summary table** (test-set loans, current filters)")
                disp = bb.copy()
                disp.columns = ["Loan-Balance Band (UPB)", "Loan Count",
                                 "Avg. Predicted Default Prob.",
                                 "Observed Default Rate", "Avg. Credit Score"]
                for c in ("Avg. Predicted Default Prob.", "Observed Default Rate"):
                    disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
                disp["Avg. Credit Score"] = disp["Avg. Credit Score"].map(
                    lambda x: f"{x:.0f}" if pd.notna(x) else "—")
                st.dataframe(disp, hide_index=True, width='stretch')
        else:
            st.info("UPB column not available in prediction output.")

    # ── Tab 4: Prediction table ──────────────────────────────────────────────
    with tab4:
        st.markdown(f"Showing **{len(filtered):,}** loans · Use sidebar filters to narrow results.")
        display_cols = [c for c in (
            "loan_sequence_number", "default", "prediction",
            "predicted_default_probability", "credit_score", "dti",
            "ltv", "cltv", "upb", "occupancy_status", "loan_purpose")
            if c in filtered.columns]
        display = filtered[display_cols].copy()
        # Human-readable labels
        if "occupancy_status" in display.columns:
            display["occupancy_status"] = _label_cat(
                display["occupancy_status"], _OCC_LABELS)
        if "loan_purpose" in display.columns:
            display["loan_purpose"] = _label_cat(
                display["loan_purpose"], _PURPOSE_LABELS)
        if "default" in display.columns:
            display["default"] = display["default"].map(
                {0: "No", 1: "Yes"})
        if "prediction" in display.columns:
            display["prediction"] = display["prediction"].map(
                {0: "Non-Default", 1: "Default"})
        if "predicted_default_probability" in display.columns:
            display["predicted_default_probability"] = display[
                "predicted_default_probability"].round(4)
        st.dataframe(
            display.sort_values("predicted_default_probability",
                                ascending=False, na_position="last"),
            hide_index=True, width='stretch', height=450)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        "All metrics come from the saved `model_metrics.json` and the "
        "test-prediction CSV files. Curves are computed from the actual "
        "predicted probabilities — not estimated or hard-coded."
    )

    if not mm:
        _missing_banner("Model Performance", "model_metrics.json not found")
        st.stop()

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
        "📊 Metrics Summary", "🔲 Confusion Matrix",
        "📈 ROC & PR Curves", "🔀 Threshold Comparison"])

    # ── Tab: Metrics summary ─────────────────────────────────────────────────
    with tab_m1:
        st.subheader("Test-Set Metrics")
        st.caption(
            f"Selected model: **{mm.get('model','—')}** · "
            f"Decision threshold: **{mm.get('selected_threshold','—')}** "
            "(maximum validation F1; threshold selected on validation set, "
            "never the test set)"
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        for col, key, label in (
            (m1, "accuracy", "Accuracy"),
            (m2, "precision", "Precision"),
            (m3, "recall", "Recall"),
            (m4, "f1", "F1 Score"),
            (m5, "roc_auc", "ROC-AUC"),
            (m6, "pr_auc", "PR-AUC"),
        ):
            with col:
                _kpi(label, _fmt_f(mm.get(key)), "test set",
                     SAFE if key in ("roc_auc", "pr_auc") else ACCENT2)

        st.markdown("---")
        st.subheader("Always-Non-Default Baseline Comparison")
        st.caption(
            "A trivial baseline that predicts every loan as non-default achieves "
            "high accuracy purely due to class imbalance. High accuracy alone is "
            "**not** evidence of successful default detection."
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        for col, (label, model_val, baseline_val) in zip(
            [b1, b2, b3, b4, b5],
            [("Accuracy", mm.get("accuracy"), mm.get("always_non_default_accuracy")),
             ("Precision", mm.get("precision"), mm.get("always_non_default_precision", 0)),
             ("Recall", mm.get("recall"), mm.get("always_non_default_recall", 0)),
             ("F1", mm.get("f1"), mm.get("always_non_default_f1", 0)),
             ("ROC-AUC", mm.get("roc_auc"), 0.5)],
        ):
            with col:
                try:
                    delta = float(model_val) - float(baseline_val)
                    delta_str = f"{delta:+.4f} vs baseline"
                except (TypeError, ValueError):
                    delta_str = ""
                _kpi(label, _fmt_f(model_val), delta_str,
                     SAFE if delta > 0 else WARN)  # type: ignore[operator]

        st.markdown("---")
        st.subheader("False Positives & False Negatives")
        col_fp, col_fn = st.columns(2)
        with col_fp:
            fp = mm.get("false_positive", 0)
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>False Positives</div>"
                f"<div class='kpi-value' style='color:{WARN}'>{fp:,}</div>"
                f"<div class='kpi-sub'>Loans flagged as default risk that did not default.<br>"
                f"These trigger unnecessary follow-up but protect against missed risk.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_fn:
            fn = mm.get("false_negative", 0)
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>False Negatives</div>"
                f"<div class='kpi-value' style='color:#ff4d4d'>{fn:,}</div>"
                f"<div class='kpi-sub'>Loans that defaulted but were not detected.<br>"
                f"These are the missed defaults — the higher-cost error for a lender.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Full metrics table
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("All saved metrics (model_metrics.json)"):
            skip = {"validation_selected_metrics"}
            rows = [(k, str(v)) for k, v in mm.items() if k not in skip]
            st.dataframe(
                pd.DataFrame(rows, columns=["Metric", "Value"]),
                hide_index=True, width='stretch')

    # ── Tab: Confusion matrix ────────────────────────────────────────────────
    with tab_m2:
        st.subheader("Confusion Matrix")
        st.caption(
            f"At decision threshold **{mm.get('selected_threshold', 0.50)}** on the held-out test set.")
        tp = mm.get("true_positive", 0)
        tn = mm.get("true_negative", 0)
        fp_v = mm.get("false_positive", 0)
        fn_v = mm.get("false_negative", 0)

        z = [[tn, fp_v], [fn_v, tp]]
        text_cm = [[f"TN = {tn:,}", f"FP = {fp_v:,}"],
                   [f"FN = {fn_v:,}", f"TP = {tp:,}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z, text=text_cm, texttemplate="%{text}",
            colorscale=[[0, "#1c2333"], [1, ACCENT]],
            showscale=False,
            xgap=3, ygap=3,
        ))
        fig_cm.update_layout(
            xaxis={"tickvals": [0, 1],
                   "ticktext": ["Predicted: Non-Default", "Predicted: Default"],
                   "side": "bottom"},
            yaxis={"tickvals": [0, 1],
                   "ticktext": ["Actual: Non-Default", "Actual: Default"],
                   "autorange": "reversed"},
            title=f"Confusion Matrix — Threshold {mm.get('selected_threshold', 0.50)}",
            height=360,
        )
        c1, _, c2 = st.columns([2, 1, 2])
        with c1:
            st.plotly_chart(_plotly_conf(fig_cm), width='stretch')
        with c2:
            total_test = tp + tn + fp_v + fn_v
            st.markdown("#### Interpretation")
            st.markdown(
                f"- **True Positives ({tp:,})**: defaults correctly detected.\n"
                f"- **True Negatives ({tn:,})**: non-defaults correctly cleared.\n"
                f"- **False Positives ({fp_v:,})**: safe loans flagged as risky "
                f"({100*fp_v/total_test:.1f}% of test set).\n"
                f"- **False Negatives ({fn_v:,})**: actual defaults missed "
                f"({100*fn_v/total_test:.1f}% of test set).\n\n"
                f"At threshold **{mm.get('selected_threshold',0.50)}** the model catches "
                f"**{100*tp/(tp+fn_v):.1f}%** of defaults (recall) while "
                f"precision is **{100*tp/(tp+fp_v):.1f}%** — "
                f"for every true default detected, roughly "
                f"**{int(fp_v/max(tp,1))}** non-defaults are also flagged."
                if total_test else ""
            )

    # ── Tab: ROC and PR curves ───────────────────────────────────────────────
    with tab_m3:
        if preds.empty or "predicted_default_probability" not in preds.columns:
            _missing_banner("ROC & PR Curves", "test prediction probabilities unavailable")
        else:
            st.subheader("ROC and Precision–Recall Curves")
            st.caption(
                "Computed from the **actual predicted probabilities** on the "
                "test set. AUC values computed via the trapezoidal rule on "
                "the sorted probability outputs."
            )
            fpr_arr, tpr_arr, roc_auc_calc = M.roc_curve(preds)
            rec_arr, prec_arr, pr_auc_calc = M.pr_curve(preds)

            c1, c2 = st.columns(2)
            with c1:
                if len(fpr_arr):
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_arr, y=tpr_arr,
                        name=f"ROC (AUC = {roc_auc_calc:.4f})",
                        line={"color": ACCENT, "width": 2.5},
                        fill="tozeroy", fillcolor="rgba(31,111,235,0.12)",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        name="Random Classifier",
                        line={"dash": "dash", "color": NEUTRAL, "width": 1.5},
                    ))
                    fig_roc.update_layout(
                        title=f"ROC Curve — AUC = {roc_auc_calc:.4f} "
                              f"(saved: {_fmt_f(mm.get('roc_auc'))})",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        legend={"x": 0.5, "y": 0.1},
                    )
                    st.plotly_chart(_plotly_conf(fig_roc), width='stretch')
            with c2:
                if len(rec_arr):
                    prev_rate = mm.get("test_default_prevalence", 0)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=rec_arr, y=prec_arr,
                        name=f"PR (AUC = {pr_auc_calc:.4f})",
                        line={"color": WARN, "width": 2.5},
                        fill="tozeroy", fillcolor="rgba(249,115,22,0.12)",
                    ))
                    fig_pr.add_shape(
                        type="line",
                        x0=0, y0=float(prev_rate), x1=1, y1=float(prev_rate),
                        line={"dash": "dot", "color": NEUTRAL, "width": 1.5},
                    )
                    fig_pr.add_annotation(
                        x=0.85, y=float(prev_rate) + 0.02,
                        text=f"No-skill baseline ({_fmt_pct(prev_rate)})",
                        showarrow=False, font={"color": NEUTRAL, "size": 11},
                    )
                    fig_pr.update_layout(
                        title=f"Precision–Recall Curve — AUC = {pr_auc_calc:.4f} "
                              f"(saved: {_fmt_f(mm.get('pr_auc'))})",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        legend={"x": 0.5, "y": 0.9},
                    )
                    st.plotly_chart(_plotly_conf(fig_pr), width='stretch')

            st.markdown(
                "> **Note**: computed AUC values may differ slightly from the "
                "saved PySpark values (different numerical integration), but "
                "both are derived from the same prediction probabilities. "
                "The saved metrics file is the canonical evaluation result."
            )

    # ── Tab: Validation threshold comparison ─────────────────────────────────
    with tab_m4:
        st.subheader("Validation Threshold Comparison")
        st.caption(
            "F1, precision, recall and predicted-positive count for both "
            "candidate models across decision thresholds 0.05–0.50. "
            "The threshold and model are selected **on validation only**."
        )
        if val_df.empty:
            _missing_banner("Threshold Comparison",
                            "validation_threshold_metrics.csv not found")
        else:
            model_options = sorted(val_df["model"].dropna().unique()
                                   ) if "model" in val_df.columns else []
            if model_options:
                sel_model = st.selectbox("Model", model_options,
                                         index=len(model_options)-1)
                vm = val_df[val_df["model"] == sel_model].sort_values("threshold")
            else:
                vm = val_df.sort_values("threshold")

            if not vm.empty:
                fig_th = go.Figure()
                for metric, colour in (("f1", SAFE), ("precision", ACCENT2),
                                        ("recall", WARN)):
                    if metric in vm.columns:
                        fig_th.add_trace(go.Scatter(
                            x=vm["threshold"], y=vm[metric],
                            name=metric.title(),
                            mode="lines+markers",
                            line={"color": colour, "width": 2},
                            marker={"size": 7},
                        ))
                # Mark selected threshold
                if mm.get("selected_threshold") is not None:
                    fig_th.add_vline(
                        x=mm["selected_threshold"],
                        line_dash="dash", line_color=ACCENT,
                        annotation_text="Selected",
                        annotation_font_color=ACCENT,
                    )
                fig_th.update_layout(
                    title="Validation Precision / Recall / F1 by Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Score",
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(_plotly_conf(fig_th), width='stretch')

                # Predicted positive count
                if "predicted_positive_count" in vm.columns:
                    fig_pp = px.bar(
                        vm, x="threshold", y="predicted_positive_count",
                        color="predicted_positive_count",
                        color_continuous_scale=[SAFE, WARN],
                        text_auto=True,
                        title="Validation: Predicted Positives by Threshold",
                    )
                    fig_pp.update_layout(
                        xaxis_title="Threshold",
                        yaxis_title="Predicted Positive Count",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(_plotly_conf(fig_pp), width='stretch')

                with st.expander("Full validation threshold table"):
                    st.dataframe(vm.reset_index(drop=True),
                                 hide_index=True, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Explorer":
    st.title("Prediction Explorer")
    st.markdown(
        "Explore individual loan predictions from the **test set**. "
        "Select a loan to view its features, model score, and actual outcome."
    )
    st.info(
        "⚠️ Predicted probability is a **model estimate**, not a guarantee. "
        "A high score means the model assigns elevated risk based on origination "
        "features — it is not a definitive statement about the loan's outcome.",
        icon="ℹ️",
    )

    if preds.empty:
        _missing_banner("Prediction Explorer", "test predictions not found")
        st.stop()

    # ── Sidebar search ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Find a Loan")
        search_by = st.radio("Search by", ["Loan Index", "Loan Sequence Number"])
        if search_by == "Loan Sequence Number" and "loan_sequence_number" in preds.columns:
            loan_ids = sorted(preds["loan_sequence_number"].dropna().unique())
            sel_id = st.selectbox("Loan Sequence Number", loan_ids)
            row_df = preds[preds["loan_sequence_number"] == sel_id]
            row = row_df.iloc[0] if not row_df.empty else None
        else:
            idx = st.number_input(
                "Loan Index (0–{})".format(len(preds)-1),
                min_value=0, max_value=len(preds)-1, value=0, step=1)
            row = preds.iloc[int(idx)]

        st.markdown("### Browse High-Risk Loans")
        if "predicted_default_probability" in preds.columns:
            high_n = st.slider("Show top N by predicted probability", 5, 50, 20)
        else:
            high_n = 20

    if row is None:
        st.warning("Loan not found.")
        st.stop()

    # ── Loan detail card ─────────────────────────────────────────────────────
    actual = int(row.get("default", -1)) if pd.notna(row.get("default")) else None
    pred_label = int(row.get("prediction", -1)) if pd.notna(row.get("prediction")) else None
    prob = float(row.get("predicted_default_probability", np.nan))
    loan_id = row.get("loan_sequence_number", "—")

    actual_str = "Default" if actual == 1 else ("Non-Default" if actual == 0 else "—")
    pred_str = "Default" if pred_label == 1 else ("Non-Default" if pred_label == 0 else "—")
    outcome_colour = WARN if actual == 1 else SAFE
    pred_colour = WARN if pred_label == 1 else SAFE
    correct = (actual == pred_label)
    correct_str = "✅ Correct" if correct else "❌ Incorrect"

    st.markdown(f"### Loan: `{loan_id}`")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Actual Outcome", actual_str, "realised label", outcome_colour)
    with c2:
        _kpi("Model Prediction", pred_str, f"threshold {mm.get('selected_threshold',0.5)}", pred_colour)
    with c3:
        _kpi("Predicted Probability",
             f"{prob:.4f}" if not np.isnan(prob) else "—",
             "estimated default risk", WARN if prob > 0.5 else SAFE)
    with c4:
        _kpi("Prediction Result", correct_str,
             "vs actual outcome", SAFE if correct else WARN)

    # ── Feature display ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Available Input Features")
    st.caption(
        "Only origination-time features are shown. The model was trained on "
        "these fields only. Performance-time data is **never** used as a feature."
    )
    feature_map = {
        "credit_score": ("Credit Score", "Borrower FICO score at origination"),
        "dti": ("Debt-to-Income (%)", "Borrower total monthly debt / gross income"),
        "ltv": ("Loan-to-Value (%)", "Loan amount / appraised value"),
        "cltv": ("Combined LTV (%)", "All liens / appraised value"),
        "upb": ("Unpaid Principal Balance ($)", "Original loan balance"),
        "occupancy_status": ("Occupancy Status", _OCC_LABELS.get(str(row.get("occupancy_status","")), "—")),
        "loan_purpose": ("Loan Purpose", _PURPOSE_LABELS.get(str(row.get("loan_purpose","")), "—")),
    }
    feat_rows = []
    for col, (display_name, description) in feature_map.items():
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if col == "occupancy_status":
                val = _OCC_LABELS.get(str(val), str(val))
            elif col == "loan_purpose":
                val = _PURPOSE_LABELS.get(str(val), str(val))
            elif col == "upb":
                val = f"${float(val):,.0f}"
            elif col in ("dti", "ltv", "cltv"):
                val = f"{float(val):.1f}%"
            else:
                val = f"{float(val):.0f}" if isinstance(row[col], float) else str(val)
            feat_rows.append({"Feature": display_name, "Value": val, "Description": description})
    if feat_rows:
        st.dataframe(pd.DataFrame(feat_rows), hide_index=True, width='stretch')
    else:
        st.info("No feature details available for this loan.")

    # ── Risk gauge ───────────────────────────────────────────────────────────
    if not np.isnan(prob):
        st.markdown("---")
        st.subheader("Predicted Risk Score Gauge")
        st.caption(
            "The gauge shows this loan's predicted default probability. "
            "The dashed line marks the 0.50 decision threshold. "
            "This is a model estimate — not a guaranteed outcome."
        )
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "", "font": {"size": 40, "color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%",
                         "tickcolor": "#8b949e"},
                "bar": {"color": WARN if prob >= 0.5 else ACCENT},
                "bgcolor": "#21262d",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 0.5], "color": "#1c2333"},
                    {"range": [0.5, 1.0], "color": "#2d1a12"},
                ],
                "threshold": {
                    "line": {"color": "#e6edf3", "width": 3},
                    "thickness": 0.8,
                    "value": mm.get("selected_threshold", 0.5),
                },
            },
            title={"text": "Predicted Default Probability",
                   "font": {"color": "#8b949e", "size": 14}},
        ))
        fig_g.update_layout(
            height=300,
            margin={"t": 60, "b": 20, "l": 40, "r": 40},
        )
        _, gc, _ = st.columns([1, 2, 1])
        with gc:
            st.plotly_chart(_plotly_conf(fig_g), width='stretch')

    # ── Top N high-risk table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Top {high_n} Loans by Predicted Default Probability")
    st.caption("Sorted by model score descending. Higher scores = more estimated risk.")
    if "predicted_default_probability" in preds.columns:
        top_n = preds.nlargest(high_n, "predicted_default_probability").copy()
        top_n["actual_outcome"] = top_n["default"].map({0: "Non-Default", 1: "Default"})
        top_n["model_prediction"] = top_n["prediction"].map(
            {0: "Non-Default", 1: "Default"})
        show_cols = [c for c in (
            "loan_sequence_number", "predicted_default_probability",
            "actual_outcome", "model_prediction",
            "credit_score", "dti", "ltv", "cltv", "upb") if c in top_n.columns]
        top_n["predicted_default_probability"] = top_n[
            "predicted_default_probability"].round(4)
        st.dataframe(top_n[show_cols], hide_index=True, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PROJECT INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Project Information":
    st.title("Project Information")

    tab_i1, tab_i2, tab_i3, tab_i4 = st.tabs([
        "📂 Dataset & Limitations", "⚙️ Pipeline & Methodology",
        "📊 Model Evaluation", "⚖️ Disclaimer"])

    with tab_i1:
        st.subheader("Dataset Source")
        st.markdown("""
**Freddie Mac Single-Family Loan-Level Dataset** (2022 Yearly Sample)

- Source: [Freddie Mac Research Datasets](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- Format: Pipe-delimited, headerless text files.
- **Origination file**: 31 fields per loan, ~50,000 records.
- **Performance file**: 35 fields per loan-month, ~2 million rows.
- Joined on the documented `loan_sequence_number` field.
- No synthetic records; no rows added or fabricated.
""")
        st.subheader("Target Definition")
        st.markdown("""
A loan is labelled **default = 1** if, within 24 months of its first reporting
period:
- it reaches **90+ days past due** (`current_loan_delinquency_status ≥ 3`), **or**
- a **zero-balance code 03 or 09** appears (foreclosure or third-party sale).

Loans without a qualifying event require **24 full months of follow-up** to be
labelled `default = 0`. Loans with fewer than 24 months of history that did
not default are **excluded** to prevent censoring bias.
""")
        st.subheader("Data Limitations")
        st.markdown("""
> ⚠️ **No borrower income, occupation, or home-ownership data exists in the
> Freddie Mac public loan-level sample layout.**

The dashboard's "Loan-Balance Proxy" section groups loans by **original unpaid
principal balance (UPB)** as a crude size proxy. These groups are not income
groups and should not be interpreted as such.

Other limitations:
- 2022 sample only — results may not generalise to other vintages.
- Only origination-time features used; no dynamic performance features.
- Academic project — not a production credit-decision system.
""")
        # Data quality table
        if dq:
            st.subheader("Data Quality Report")
            n_rows = dq.get("rows", "—")
            n_cols = dq.get("columns", "—")
            n_dups = dq.get("duplicate_loans", "—")
            q1, q2, q3 = st.columns(3)
            with q1:
                _kpi("Total Rows", f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows), "eligible loans", ACCENT2)
            with q2:
                _kpi("Columns", str(n_cols), "all feature+meta fields", ACCENT2)
            with q3:
                _kpi("Duplicate Loans", str(n_dups), "should be 0", SAFE if n_dups == 0 else WARN)
            missing_df = pd.DataFrame(dq.get("missing", []))
            if not missing_df.empty:
                with st.expander("Missing-value report by column"):
                    st.dataframe(
                        missing_df.sort_values("missing_pct", ascending=False),
                        hide_index=True, width='stretch')

    with tab_i2:
        st.subheader("PySpark Processing Workflow")
        st.markdown("""
```
sample_orig_2022.txt  (50k loans, 31 fields, pipe-delimited)
sample_perf_2022.txt  (2M rows, 35 fields, pipe-delimited)
        │
        ▼  spark_session.get_spark()    local[2], driver memory 2 g,
        │                               project-local .spark_tmp dir
        ▼  data_loader.load_freddie_mac()
        │  ├─ validate field counts (31 / 35) before Spark reads
        │  └─ schema: all columns as StringType (Spark infers nothing)
        │
        ▼  data_loader.build_observations()
        │  ├─ parse monthly_reporting_period → date
        │  ├─ compute first_report / last_report per loan
        │  ├─ flag default = 1 within 24-month horizon
        │  ├─ exclude censored non-defaults (< 24 months follow-up)
        │  ├─ dropDuplicates on loan_sequence_number
        │  └─ cast credit_score, dti, ltv, cltv, upb → DoubleType
        │
        ▼  validate_observations()   checks binary label + no null IDs
        ▼  explore()                 writes outputs/tables/
        │
        ▼  split_by_loan()   randomSplit [0.70, 0.15, 0.15], seed 42
        │                    (loan-level → no loan appears in 2 partitions)
        │
        ▼  fit_model()  × 2 candidates
        │  ├─ Imputer (median, training set only)
        │  ├─ StringIndexer × 4 categorical columns
        │  ├─ OneHotEncoder
        │  ├─ VectorAssembler (23 features total)
        │  └─ LogisticRegression (baseline / class-weighted)
        │
        ▼  evaluate on validation → select model + threshold by max F1
        ▼  evaluate_test() once on test set → outputs/metrics/
        ▼  save_predictions() → outputs/predictions/
        ▼  create_charts()   → outputs/charts/
        ▼  create_report()   → report/project_report.docx
```
""")

        st.subheader("Logistic Regression Methodology")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Algorithm | Spark MLlib LogisticRegression |
| Max iterations | 100 |
| Regularisation (regParam) | 0.05 |
| Elastic net (elasticNetParam) | 0.0 (pure L2) |
| Class weighting | Inverse-frequency weights on training set |
| Weight formula | `class_weight = count(0) / count(1)` for default class |
| Features | {mm.get("training_records","—")} training loans, 23 features |
| Missing values | Median imputation (training set only) |
| Categorical encoding | StringIndexer → OneHotEncoder |

**Why class weighting?**  
At ~1.9% default prevalence, an unweighted model converges to predicting
everything as non-default (0 TP). Inverse-frequency class weights upweight the
minority (default) class during gradient descent, shifting the decision
boundary toward the rare class.
""")

    with tab_i3:
        st.subheader("Model Evaluation Summary")
        if not mm:
            _missing_banner("Evaluation Summary", "model_metrics.json not found")
        else:
            st.markdown(f"""
| Item | Value |
|---|---|
| Selected model | `{mm.get("model","—")}` |
| Decision threshold | `{mm.get("selected_threshold","—")}` |
| Threshold selection | Max validation F1; ties → higher recall |
| Training records | {mm.get("training_records",0):,} |
| Validation records | {mm.get("validation_records",0):,} |
| Test records | {mm.get("test_records",0):,} |
| Test default prevalence | {_fmt_pct(mm.get("test_default_prevalence"))} |
| **Precision** | **{_fmt_f(mm.get("precision"))}** |
| **Recall** | **{_fmt_f(mm.get("recall"))}** |
| **F1 Score** | **{_fmt_f(mm.get("f1"))}** |
| ROC-AUC | {_fmt_f(mm.get("roc_auc"))} |
| PR-AUC | {_fmt_f(mm.get("pr_auc"))} |
| True Positives | {mm.get("true_positive",0):,} |
| False Positives | {mm.get("false_positive",0):,} |
| True Negatives | {mm.get("true_negative",0):,} |
| False Negatives | {mm.get("false_negative",0):,} |
| Always-non-default accuracy | {_fmt_pct(mm.get("always_non_default_accuracy"))} |
| Always-non-default recall | 0.0000 |
""")
            st.markdown("""
**Honest interpretation**

The model genuinely detects defaults (73% recall versus 0% for the trivial
baseline) but precision remains low: for every true default detected,
roughly 22 non-defaults are also flagged as risky. This reflects the
difficulty of the problem — only five numeric and two categorical
origination-time features are available, and borrower income is absent.

This is an academic baseline result. Defensible next experiments include
richer feature sets, tree-based models, or a time-based validation split.
Do not tune the threshold further against the test labels.
""")

    with tab_i4:
        st.subheader("Academic Disclaimer")
        st.markdown("""
> This project is an **academic exercise** in applying PySpark MLlib to
> publicly available Freddie Mac loan-level data.
>
> **It is not a production credit-decision system.**
>
> Results depend on the specific data vintage, observation window,
> available features, and modelling choices used here. They should not be
> used for actual lending decisions without extensive independent validation,
> bias testing, regulatory compliance review, and governance controls.
>
> The Freddie Mac data is subject to Freddie Mac's terms of use. Review and
> comply with those terms before downloading or reproducing any data.
>
> AUC and metric values shown in this dashboard are computed from the saved
> test-prediction probabilities. They may differ trivially from the saved
> PySpark AUC values due to different numerical integration methods; the
> saved `model_metrics.json` is the canonical evaluation record.
""")
        st.markdown("---")
        st.markdown(
            "<small style='color:#8b949e'>"
            "Dashboard: Streamlit · Plotly · Pandas · Python 3.11<br>"
            "Pipeline: PySpark 3.5.1 · MLlib Logistic Regression<br>"
            "Project root: <code>"
            + str(_PROJECT_ROOT).replace("\\", "/") +
            "</code></small>",
            unsafe_allow_html=True,
        )
