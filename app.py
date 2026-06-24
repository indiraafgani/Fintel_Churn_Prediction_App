"""
FINTel — Customer Churn Intelligence Dashboard
Single unified Logistic Regression model.
Churn Segment: Low / Mid / High (binned from ChurnScore).
Recommendations: per-feature, 3 tiers keyed to top-3 SHAP.
"""

import io
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from utils.prediction import (
    FEATURE_COLS,
    predict_single,
    predict_bulk,
    validate_upload,
    prob_to_churnscore,
    get_churn_segment,
)
from utils.visualization import (
    churn_score_gauge,
    shap_individual_bar,
    shap_global_bar,
    churn_distribution_donut,
    segment_bar,
    probability_histogram,
    churn_score_histogram,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FINTel — Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #EBECEF;
    --navy:     #0F1D3D;
    --primary:  #1A3462;
    --secondary:#476996;
    --soft:     #9AADC2;
    --white:    #FFFFFF;
    --success:  #27AE60;
    --warning:  #F39C12;
    --danger:   #E74C3C;
    --muted:    #6B7280;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--navy) !important;
}
[data-testid="stSidebar"]          { background: var(--primary) !important; border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] *        { color: rgba(255,255,255,0.9) !important; }
[data-testid="stSidebar"] hr       { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stAppViewContainer"] > .main { background: var(--bg) !important; }
[data-testid="block-container"]    { padding: 1.25rem 1.75rem !important; background: var(--bg) !important; }

.fin-card          { background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(15,29,61,0.07); border: 1px solid rgba(15,29,61,0.06); margin-bottom: 14px; }
.fin-card-hdr      { font-size: 10px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--secondary); margin-bottom: 10px; }
.page-header       { background: linear-gradient(135deg, var(--navy) 0%, var(--primary) 100%); border-radius: 14px; padding: 26px 30px; margin-bottom: 20px; }
.page-header h1    { font-size: 21px; font-weight: 700; color: white !important; margin: 0 0 4px 0; }
.page-header p     { font-size: 13px; color: rgba(255,255,255,0.65); margin: 0; }
.section-title     { font-size: 10px; font-weight: 600; color: var(--secondary); text-transform: uppercase; letter-spacing: 1.1px; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid var(--bg); }
.info-row          { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--bg); font-size: 12px; }
.info-k            { color: var(--muted); }
.info-v            { color: var(--navy); font-weight: 600; }
.badge             { display: inline-block; padding: 3px 11px; border-radius: 20px; font-size: 10px; font-weight: 600; }
.b-high            { background: #FEE2E2; color: #991B1B; }
.b-mid             { background: #FEF3C7; color: #92400E; }
.b-low             { background: #D1FAE5; color: #065F46; }
.b-model           { background: rgba(26,52,98,0.1); color: #1A3462; border: 1px solid rgba(26,52,98,0.25); }
.prob-bar-wrap     { margin: 8px 0; }
.prob-labels       { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-bottom: 3px; }
.prob-track        { background: var(--bg); border-radius: 5px; height: 9px; overflow: hidden; }
.pred-box          { border-radius: 8px; padding: 9px 14px; text-align: center; margin: 10px 0; }
.fin-divider       { height: 1px; background: linear-gradient(to right, var(--bg), rgba(71,105,150,0.2), var(--bg)); margin: 16px 0; }
.metric-card       { background: white; border: 1px solid rgba(15,29,61,0.06); border-radius: 10px; padding: 16px 18px; text-align: center; }
.metric-label      { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 4px; }
.metric-value      { font-size: 26px; font-weight: 700; color: var(--navy); line-height: 1.1; }
.metric-sub        { font-size: 11px; color: var(--muted); margin-top: 3px; }

.shap-row          { display: flex; align-items: center; gap: 7px; padding: 4px 0; border-bottom: 1px solid var(--bg); font-size: 11px; }
.shap-rank         { flex-shrink: 0; width: 18px; height: 18px; border-radius: 50%; font-size: 9px; font-weight: 700; display: flex; align-items: center; justify-content: center; color: white; }
.shap-name         { flex: 0 0 148px; color: var(--navy); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.shap-track-w      { flex: 1; background: var(--bg); border-radius: 3px; height: 7px; }
.shap-fill         { height: 7px; border-radius: 3px; }
.shap-dir          { flex: 0 0 58px; font-size: 9px; font-weight: 700; padding: 2px 5px; border-radius: 3px; text-align: center; }
.dir-up            { background: #FEE2E2; color: #991B1B; }
.dir-dn            { background: #D1FAE5; color: #065F46; }

.rec-card          { background: white; border: 1px solid rgba(15,29,61,0.06); border-radius: 12px; padding: 16px 18px; margin-bottom: 10px; }
.rec-item          { display: flex; gap: 10px; padding: 10px 12px; border-radius: 8px; margin-bottom: 8px; align-items: flex-start; }
.rec-rank-badge    { flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%; font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center; color: white; }
.rec-content       { flex: 1; }
.rec-feature       { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px; }
.rec-text          { font-size: 12px; color: var(--navy); line-height: 1.5; }
.rec-level-pill    { display: inline-block; font-size: 9px; font-weight: 600; padding: 2px 8px; border-radius: 10px; margin-top: 4px; }
.level-1           { background: #FEE2E2; color: #991B1B; }
.level-2           { background: #FEF3C7; color: #92400E; }
.level-3           { background: #DBEAFE; color: #1E40AF; }

[data-testid="stTabs"] [data-baseweb="tab-list"]  { background: white !important; border-radius: 10px !important; padding: 4px !important; box-shadow: 0 1px 4px rgba(15,29,61,0.06) !important; border: 1px solid rgba(15,29,61,0.06) !important; }
[data-testid="stTabs"] [data-baseweb="tab"]       { border-radius:7px !important; font-weight:500 !important; font-size:13px !important; color:var(--secondary) !important; padding:7px 18px !important; flex:1 !important; text-align:center !important; justify-content:center !important; }
[data-testid="stTabs"] [aria-selected="true"]     { background: var(--primary) !important; color: white !important; }
[data-testid="stButton"] > button                 { background: var(--primary) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stButton"] > button:hover           { background: var(--secondary) !important; }

.sb-logo           { padding-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px; }
.sb-logo-mark      { font-size: 22px; font-weight: 700; color: white !important; letter-spacing: -0.5px; }
.sb-logo-tag       { font-size: 9px; color: rgba(255,255,255,0.42) !important; text-transform: uppercase; letter-spacing: 2px; }
.sb-section        { font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.4px; color: rgba(255,255,255,0.36) !important; margin: 14px 0 6px; }
.sb-pill           { background: rgba(255,255,255,0.07); border: 0.5px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 9px 12px; margin-bottom: 7px; }
.sb-pill-l         { font-size: 9px; color: rgba(255,255,255,0.42) !important; text-transform: uppercase; letter-spacing: 0.7px; }
.sb-pill-v         { font-size: 17px; font-weight: 600; color: white !important; }
.sb-pill-s         { font-size: 10px; color: rgba(255,255,255,0.42) !important; }
.sb-mrow           { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 11px; }
.sb-mk             { color: rgba(255,255,255,0.48) !important; }
.sb-mv             { color: white !important; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.sb-footer         { margin-top: auto; font-size: 9px; color: rgba(255,255,255,0.22) !important; text-align: center; border-top: 1px solid rgba(255,255,255,0.07); padding-top: 10px; line-height: 1.5; }
.seg-legend-row    { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
.seg-dot-item      { display: flex; align-items: center; gap: 5px; font-size: 11px; color: rgba(255,255,255,0.7) !important; }
.seg-dot           { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("Model file not found. Pastikan `models/artifacts.pkl` tersedia.")
    st.stop()

metrics          = artifacts["model_metrics"]
global_shap      = artifacts["global_shap_importance"]
selected_names   = artifacts["selected_feature_names"]


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_ref_data():
    try:
        df = pd.read_csv("data/df_clean.csv")
        if "customerID" not in df.columns:
            df.insert(0, "customerID", [f"CUST-{i+1:05d}" for i in range(len(df))])
        return df
    except FileNotFoundError:
        return None

ref_data = load_ref_data()


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def segment_badge(seg: str) -> str:
    cls = {"High": "b-high", "Mid": "b-mid", "Low": "b-low"}.get(seg, "b-low")
    return f'<span class="badge {cls}">{seg} Churn</span>'


def info_row(label: str, value: str) -> str:
    return (
        f'<div class="info-row">'
        f'<span class="info-k">{label}</span>'
        f'<span class="info-v">{value}</span>'
        f'</div>'
    )


def prob_bar(prob: float) -> str:
    pct   = int(prob * 100)
    color = "#E74C3C" if pct >= 67 else ("#F39C12" if pct >= 33 else "#27AE60")
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-labels"><span>Churn Probability</span>'
        f'<span style="font-weight:700;color:{color};">{pct}%</span></div>'
        f'<div class="prob-track">'
        f'<div class="shap-fill" style="width:{pct}%;background:{color};height:9px;border-radius:5px;"></div>'
        f'</div></div>'
    )


def pred_box(prob: float, threshold: float) -> str:
    if prob >= threshold:
        label, bg, border, color = "Predicted to Churn",   "#FEF2F2", "#E74C3C55", "#991B1B"
    else:
        label, bg, border, color = "Predicted to Stay",     "#F0FDF4", "#27AE6055", "#065F46"
    return (
        f'<div class="pred-box" style="background:{bg};border:1px solid {border};">'
        f'<div style="font-size:13px;font-weight:700;color:{color};">{label}</div>'
        f'<div style="font-size:11px;color:#6B7280;margin-top:3px;">'
        f'Probability: {int(prob*100)}% · Threshold: {threshold:.4f}</div>'
        f'</div>'
    )


def shap_rows_html(shap_display: list) -> str:
    rank_colors  = ["#E74C3C", "#F39C12", "#3B82F6"]
    rank_bg_gray = "#9CA3AF"
    max_abs = max(abs(d["value"]) for d in shap_display) if shap_display else 1
    html = ""
    for i, d in enumerate(shap_display):
        rank_n   = i + 1
        r_color  = rank_colors[i] if i < 3 else rank_bg_gray
        fill_c   = "#E74C3C" if d["direction"] == "up" else "#27AE60"
        dir_cls  = "dir-up" if d["direction"] == "up" else "dir-dn"
        dir_lbl  = "↑ Churn" if d["direction"] == "up" else "↓ Churn"
        bar_w    = int(abs(d["value"]) / max_abs * 100) if max_abs else 0
        html += (
            f'<div class="shap-row">'
            f'<div class="shap-rank" style="background:{r_color};">{rank_n}</div>'
            f'<div class="shap-name" title="{d["name"]}">{d["name"]}</div>'
            f'<div class="shap-track-w">'
            f'<div class="shap-fill" style="width:{bar_w}%;background:{fill_c};"></div>'
            f'</div>'
            f'<div class="shap-dir {dir_cls}">{dir_lbl}</div>'
            f'</div>'
        )
    return html


REC_BG = {1: "rgba(231,76,60,0.07)", 2: "rgba(243,156,18,0.07)", 3: "rgba(59,130,246,0.07)"}
REC_BORDER = {1: "#E74C3C", 2: "#F39C12", 3: "#3B82F6"}
REC_BADGE_BG = {1: "#E74C3C", 2: "#F39C12", 3: "#3B82F6"}
LEVEL_PILL_CLS = {1: "level-1", 2: "level-2", 3: "level-3"}

def render_rec_cards(recommendations: list):
    for rec in recommendations:
        rank  = rec["rank"]
        bg    = REC_BG[rank]
        bdr   = REC_BORDER[rank]
        badge = REC_BADGE_BG[rank]
        pill  = LEVEL_PILL_CLS[rank]
        st.markdown(
            f'<div class="rec-item" style="background:{bg};border-left:3px solid {bdr};">'
            f'<div class="rec-rank-badge" style="background:{badge};">{rank}</div>'
            f'<div class="rec-content">'
            f'<div class="rec-feature" style="color:{bdr};">{rec["feature_name"]} · Rank #{rank}</div>'
            f'<div class="rec-text">{rec["recommendation"]}</div>'
            f'<span class="rec-level-pill {pill}">{rec["level_label"]}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-mark">FINTel</div>
      <div class="sb-logo-tag">Churn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">About Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pill">
      <div class="sb-pill-l">Model</div>
      <div style="font-size:13px;color:white;margin-top:2px;">Customer Churn Prediction</div>
      <div class="sb-pill-s">Logistic Regression · F2-Optimized</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pill">
      <div class="sb-pill-l">Total Customers</div>
      <div class="sb-pill-v">7,032</div>
      <div class="sb-pill-s">Telco Customer Churn · IBM Dataset</div>
    </div>
    <div class="sb-pill">
      <div class="sb-pill-l">Features Used</div>
      <div class="sb-pill-v">20</div>
      <div class="sb-pill-s">Post SelectKBest (ANOVA F-test, k=20)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Churn Segmentation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pill">
      <div class="sb-pill-l">Binning dari ChurnScore</div>
      <div class="seg-legend-row" style="margin-top:6px;">
        <div class="seg-dot-item">
          <div class="seg-dot" style="background:#27AE60;"></div>
          <span>Low (0–33)</span>
        </div>
        <div class="seg-dot-item">
          <div class="seg-dot" style="background:#F39C12;"></div>
          <span>Mid (34–66)</span>
        </div>
        <div class="seg-dot-item">
          <div class="seg-dot" style="background:#E74C3C;"></div>
          <span>High (67–100)</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Evaluation Metrics</div>', unsafe_allow_html=True)
    for k, v in [
        ("F2 (β=2)",  metrics["f2"]),
        ("ROC-AUC",   metrics["roc_auc"]),
        ("Threshold", metrics["threshold"]),
    ]:
        st.markdown(
            f'<div class="sb-mrow">'
            f'<span class="sb-mk">{k}</span>'
            f'<span class="sb-mv">{v:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-section">Business Objective</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:rgba(255,255,255,0.6);line-height:1.65;padding:0 2px;">
      Prediksi churn pelanggan dan berikan rekomendasi retensi personal
      berbasis Top-3 SHAP features per individu.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-footer">
      FINTel <br>
      Akbar K. · Khaerun N. · Indira F.A.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>FINTel Churn Intelligence Dashboard</h1>
  <p>ML-powered churn prediction, Low/Mid/High segmentation & SHAP-based personalised retention recommendations</p>
</div>
""", unsafe_allow_html=True)

tab_existing, tab_new, tab_bulk = st.tabs([
    "  Existing Customer  ",
    "  New Customer  ",
    "  Bulk Prediction  ",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXISTING CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
with tab_existing:
    st.markdown("<br>", unsafe_allow_html=True)

    col_in, col_hint = st.columns([2, 3])
    with col_in:
        st.markdown('<div class="section-title">Masukkan Customer ID</div>', unsafe_allow_html=True)
        cid_input = st.text_input(
            "Customer ID", placeholder="e.g. 7216-EWTRS",
            label_visibility="collapsed",
        )

    if cid_input:
        cid = cid_input.strip().upper()
        found = None
        if ref_data is not None:
            match = ref_data[ref_data["customerID"].str.upper() == cid]
            if not match.empty:
                found = match.iloc[0]

        if found is None:
            st.warning(f"Customer ID **{cid}** tidak ditemukan.")
        else:
            with st.spinner("Menghitung SHAP values..."):
                result = predict_single(found, artifacts)

            seg  = result["segment"]
            prob = result["probability"]
            thr  = result["threshold"]
            cs   = result["churn_score"]

            # ── Header strip ──
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:14px 20px;margin-bottom:14px;
                        border:1px solid rgba(15,29,61,0.07);display:flex;
                        justify-content:space-between;align-items:center;
                        box-shadow:0 2px 8px rgba(15,29,61,0.06);">
              <div>
                <div style="font-size:10px;color:#9AADC2;text-transform:uppercase;letter-spacing:1.2px;">
                  Customer Report</div>
                <div style="font-size:20px;font-weight:700;color:#0F1D3D;
                            font-family:'JetBrains Mono',monospace;">{cid}</div>
              </div>
              <div style="display:flex;gap:8px;align-items:center;">
                <span class="badge b-model">Logistic Regression</span>
                {segment_badge(seg)}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Info cards ──
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Customer Overview</div>', unsafe_allow_html=True)
                st.markdown(
                    info_row("Customer ID", f'<span style="font-family:JetBrains Mono,monospace;font-size:11px">{cid}</span>') +
                    info_row("Gender",       str(found.get("gender", "—"))) +
                    info_row("Senior Citizen", str(found.get("SeniorCitizen", "—"))) +
                    info_row("Partner",       str(found.get("Partner", "—"))) +
                    info_row("Dependents",    str(found.get("Dependents", "—"))),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Account Information</div>', unsafe_allow_html=True)
                st.markdown(
                    info_row("Tenure",          f'{int(found.get("tenure", 0))} month(s)') +
                    info_row("Contract",         str(found.get("Contract",        "—"))) +
                    info_row("Internet Service", str(found.get("InternetService", "—"))) +
                    info_row("Phone Service",    str(found.get("PhoneService",    "—"))) +
                    info_row("Multiple Lines",   str(found.get("MultipleLines",   "—"))),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_c:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Billing Information</div>', unsafe_allow_html=True)
                st.markdown(
                    info_row("Monthly Charges", f'${float(found.get("MonthlyCharges", 0)):.2f}') +
                    info_row("Total Charges",   f'${float(found.get("TotalCharges", 0)):,.2f}') +
                    info_row("Payment Method",  str(found.get("PaymentMethod", "—"))) +
                    info_row("Paperless Billing", str(found.get("PaperlessBilling", "—"))) +
                    info_row("Est. CLTV",       f'${int(found.get("CLTV", 0)):,}'),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)

            # ── Add-Ons card ──
            st.markdown('<div class="fin-card"><div class="fin-card-hdr">Add-On Services</div>', unsafe_allow_html=True)
            col_add1, col_add2, col_add3 = st.columns(3)
            with col_add1:
                st.markdown(
                    info_row("Online Security",   str(found.get("OnlineSecurity",   "—"))) +
                    info_row("Online Backup",     str(found.get("OnlineBackup",     "—"))),
                    unsafe_allow_html=True,
                )
            with col_add2:
                st.markdown(
                    info_row("Device Protection", str(found.get("DeviceProtection", "—"))) +
                    info_row("Tech Support",      str(found.get("TechSupport",      "—"))),
                    unsafe_allow_html=True,
                )
            with col_add3:
                st.markdown(
                    info_row("Streaming TV",      str(found.get("StreamingTV",      "—"))) +
                    info_row("Streaming Movies",  str(found.get("StreamingMovies",  "—"))),
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Churn Analysis Result</div>', unsafe_allow_html=True)

            col_g, col_d = st.columns([1, 1])
            with col_g:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Churn Score</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    churn_score_gauge(cs, seg),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown(
                    f'<div style="text-align:center;margin-top:-10px;">'
                    f'<div style="font-size:9px;color:#9AADC2;margin-bottom:5px;">'
                    f'Piecewise Scaler · Low 0–33 | Mid 34–66 | High 67–100</div>'
                    f'<div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">'
                    f'<span class="badge b-low" style="font-size:9px;">Low (0–33)</span>'
                    f'<span class="badge b-mid" style="font-size:9px;">Mid (34–66)</span>'
                    f'<span class="badge b-high" style="font-size:9px;">High (67–100)</span>'
                    f'</div>'
                    f'<div style="margin-top:8px;">{segment_badge(seg)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_d:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Prediction Detail</div>', unsafe_allow_html=True)
                st.markdown(prob_bar(prob), unsafe_allow_html=True)
                st.markdown(pred_box(prob, thr), unsafe_allow_html=True)
                st.markdown(
                    info_row("Churn Score",   str(cs)) +
                    info_row("Churn Segment", seg) +
                    info_row("Model",         "Logistic Regression") +
                    info_row("Threshold",     f"{thr:.4f}"),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # ── SHAP + Recommendations ──
            st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Individual SHAP + Personalised Recommendations</div>', unsafe_allow_html=True)

            col_shap, col_rec = st.columns([1, 1])
            with col_shap:
                st.markdown('<div class="fin-card"><div class="fin-card-hdr">Top SHAP Features (ranked)</div>', unsafe_allow_html=True)
                st.markdown(shap_rows_html(result["shap_display"]), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_rec:
                st.markdown(
                    f'<div class="fin-card">'
                    f'<div class="fin-card-hdr">Retention Recommendations · {segment_badge(seg)}</div>'
                    f'<div style="font-size:10px;color:#9AADC2;margin-bottom:10px;">'
                    f'Berdasarkan top 3 SHAP feature pelanggan ini</div>',
                    unsafe_allow_html=True,
                )
                render_rec_cards(result["recommendations"])
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Global SHAP chart ──
            with st.expander("Global Feature Importance (semua pelanggan)", expanded=False):
                st.plotly_chart(
                    shap_global_bar(global_shap),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    else:
        st.markdown("""
        <div style="text-align:center;padding:56px 20px;background:white;border-radius:14px;
                    border:2px dashed rgba(71,105,150,0.2);">
          <div style="font-size:44px;margin-bottom:14px;"></div>
          <div style="font-size:15px;font-weight:600;color:#476996;margin-bottom:8px;">
            Masukkan Customer ID untuk melihat laporan churn analysis</div>
          <div style="font-size:12px;color:#9AADC2;">
            Format: <code style="background:#EBECEF;padding:2px 8px;
            border-radius:4px;color:#1A3462;">7590-VHVEG</code> sampai
            <code style="background:#EBECEF;padding:2px 8px;
            border-radius:4px;color:#1A3462;">7216-EWTRS</code></div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — NEW CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
with tab_new:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:white;border-radius:10px;padding:12px 18px;margin-bottom:16px;
                border:1px solid rgba(15,29,61,0.07);font-size:12px;color:#476996;">
      Isi profil pelanggan baru untuk mendapatkan prediksi churn dan rekomendasi retensi personal.
    </div>
    """, unsafe_allow_html=True)

    with st.form("new_customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">Demographic</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender",         ["Female", "Male"])
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner",        ["No", "Yes"])
            dependents = st.selectbox("Dependents",     ["No", "Yes"])
            city       = st.text_input("City", value="Los Angeles",
                                       help="Nama kota pelanggan (digunakan untuk TargetEncoding)")

        with c2:
            st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)
            phone_svc  = st.selectbox("Phone Service",     ["Yes", "No"])
            multi_lines= st.selectbox("Multiple Lines",     ["No", "Yes", "No phone service"])
            internet   = st.selectbox("Internet Service",   ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security",    ["No", "Yes", "No internet service"])
            online_bk  = st.selectbox("Online Backup",      ["No", "Yes", "No internet service"])
            dev_prot   = st.selectbox("Device Protection",  ["No", "Yes", "No internet service"])
            tech_sup   = st.selectbox("Tech Support",       ["No", "Yes", "No internet service"])
            stream_tv  = st.selectbox("Streaming TV",       ["No", "Yes", "No internet service"])
            stream_mov = st.selectbox("Streaming Movies",   ["No", "Yes", "No internet service"])

        with c3:
            st.markdown('<div class="section-title">Account & Billing</div>', unsafe_allow_html=True)
            contract   = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
            paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment    = st.selectbox("Payment Method",    [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            tenure      = st.number_input("Tenure (months)", min_value=0, max_value=72, value=6, step=1)
            monthly_chg = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
            total_chg   = round(monthly_chg * max(int(tenure), 1), 2)
            cltv        = int(monthly_chg * 60)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        input_row = pd.Series({
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": float(tenure),
            "PhoneService": phone_svc, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bk, "DeviceProtection": dev_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_chg),
            "TotalCharges": float(total_chg),
            "CLTV": float(cltv),
        })

        with st.spinner("Menghitung SHAP values..."):
            result = predict_single(input_row, artifacts)

        seg  = result["segment"]
        prob = result["probability"]
        thr  = result["threshold"]
        cs   = result["churn_score"]

        st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        col_g, col_d = st.columns([1, 1])
        with col_g:
            st.markdown('<div class="fin-card"><div class="fin-card-hdr">Churn Score</div>', unsafe_allow_html=True)
            st.plotly_chart(
                churn_score_gauge(cs, seg),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f'<div style="text-align:center;margin-top:-10px;">'
                f'<div style="font-size:9px;color:#9AADC2;margin-bottom:5px;">'
                f'Piecewise Scaler · Low 0–33 | Mid 34–66 | High 67–100</div>'
                f'<div style="margin-top:6px;">{segment_badge(seg)}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="fin-card"><div class="fin-card-hdr">Prediction Summary</div>', unsafe_allow_html=True)
            st.markdown(prob_bar(prob), unsafe_allow_html=True)
            st.markdown(pred_box(prob, thr), unsafe_allow_html=True)
            st.markdown(
                info_row("Churn Score",   str(cs)) +
                info_row("Churn Segment", seg) +
                info_row("Model",         "Logistic Regression") +
                info_row("Threshold",     f"{thr:.4f}"),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Individual SHAP + Personalised Recommendations</div>', unsafe_allow_html=True)

        col_shap, col_rec = st.columns([1, 1])
        with col_shap:
            st.markdown('<div class="fin-card"><div class="fin-card-hdr">Top SHAP Features (ranked)</div>', unsafe_allow_html=True)
            st.markdown(shap_rows_html(result["shap_display"]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_rec:
            st.markdown(
                f'<div class="fin-card">'
                f'<div class="fin-card-hdr">Retention Recommendations · {segment_badge(seg)}</div>'
                f'<div style="font-size:10px;color:#9AADC2;margin-bottom:10px;">'
                f'Berdasarkan top 3 SHAP feature untuk profil ini</div>',
                unsafe_allow_html=True,
            )
            render_rec_cards(result["recommendations"])
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — BULK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_bulk:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Format CSV yang Diperlukan", expanded=False):
        st.markdown(f"""
        File CSV harus memiliki kolom berikut (header harus sama persis):
        ```
        {", ".join(FEATURE_COLS)}
        ```
        Kolom opsional yang ditampilkan jika ada: `CustomerID`
        """)
        template = pd.DataFrame(columns=["customerID"] + FEATURE_COLS)
        st.download_button(
            "⬇ Download Template CSV",
            data=template.to_csv(index=False),
            file_name="fintel_bulk_template.csv",
            mime="text/csv",
        )

    st.markdown('<div class="section-title">Upload File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        label_visibility="collapsed",
        help="CSV dengan kolom sesuai format model FINTel",
    )

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        is_valid, msg = validate_upload(df_up)
        if not is_valid:
            st.error(f"File tidak valid. {msg}")
            st.stop()

        st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Total Rows</div>
              <div class="metric-value">{len(df_up):,}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Columns</div>
              <div class="metric-value">{len(df_up.columns)}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            avg_ten = df_up["tenure"].mean()
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Avg Tenure</div>
              <div class="metric-value">{avg_ten:.1f}</div>
              <div class="metric-sub">months</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_up.head(10), use_container_width=True, hide_index=True)

        st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
        run_btn = st.button("Run Bulk Prediction", use_container_width=True)

        if run_btn:
            with st.spinner("Menjalankan prediksi untuk semua customer..."):
                df_res = predict_bulk(df_up.copy(), artifacts)

            n_total    = len(df_res)
            n_churn    = int((df_res["Predicted_Label"] == "Churn").sum())
            n_no_churn = n_total - n_churn
            churn_rate = n_churn / n_total * 100 if n_total > 0 else 0

            st.markdown('<div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Total Customers</div>
                  <div class="metric-value">{n_total:,}</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Predicted Churn</div>
                  <div class="metric-value" style="color:#E74C3C;">{n_churn:,}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Predicted No Churn</div>
                  <div class="metric-value" style="color:#27AE60;">{n_no_churn:,}</div>
                </div>""", unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Churn Rate</div>
                  <div class="metric-value" style="color:#E74C3C;">{churn_rate:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(churn_distribution_donut(n_churn, n_no_churn),
                                use_container_width=True, config={"displayModeBar": False})
            with v2:
                st.plotly_chart(segment_bar(df_res),
                                use_container_width=True, config={"displayModeBar": False})

            v3, v4 = st.columns(2)
            with v3:
                st.plotly_chart(probability_histogram(df_res),
                                use_container_width=True, config={"displayModeBar": False})
            with v4:
                st.plotly_chart(churn_score_histogram(df_res),
                                use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="fin-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Result Table</div>', unsafe_allow_html=True)

            display_cols = (
                (["customerID"] if "customerID" in df_res.columns else []) +
                ["tenure", "Contract", "MonthlyCharges",
                 "Churn_Probability", "Churn_Score",
                 "Predicted_Label", "Churn_Segment"]
            )
            display_cols = [c for c in display_cols if c in df_res.columns]

            def _color_label(v):
                return "color:#E74C3C;font-weight:600" if v == "Churn" else "color:#27AE60;font-weight:600"

            def _color_seg(v):
                return {"High": "color:#E74C3C;font-weight:600",
                        "Mid":  "color:#F39C12;font-weight:600",
                        "Low":  "color:#27AE60;font-weight:600"}.get(v, "")

            styled = (
                df_res[display_cols]
                .style
                .applymap(_color_label, subset=["Predicted_Label"])
                .applymap(_color_seg,   subset=["Churn_Segment"])
                .format({"Churn_Probability": "{:.4f}",
                         "MonthlyCharges": "${:.2f}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = df_res[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Prediction Result (CSV)",
                data=csv_out,
                file_name="fintel_bulk_result.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div style="text-align:center;padding:56px 20px;background:white;border-radius:14px;
                    border:2px dashed rgba(71,105,150,0.2);">
          <div style="font-size:44px;margin-bottom:14px;"></div>
          <div style="font-size:15px;font-weight:600;color:#476996;margin-bottom:8px;">
            Upload CSV untuk prediksi massal</div>
          <div style="font-size:12px;color:#9AADC2;">
            Gunakan tombol "Format CSV" di atas untuk melihat kolom yang dibutuhkan</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:16px;border-top:1px solid rgba(15,29,61,0.08);
            font-size:11px;color:#9AADC2;">
  FINTel Customer Churn Intelligence Dashboard <br>
  Akbar Kanugraha · Khaerun Nisa'Tri Safaati · Indira Faisa Afgani
</div>
""", unsafe_allow_html=True)
