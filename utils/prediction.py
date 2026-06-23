"""
utils/prediction.py
===================
Helper functions untuk prediksi churn FINTel.
Digunakan oleh app.py.
"""

import numpy as np
import pandas as pd
import shap

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (input raw dari user / CSV upload)
# Sesuaikan urutan ini dengan kolom di df_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "CLTV",
]

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION MAP
# Key: nama feature (setelah encoding, bisa partial match)
# Tier 1 = paling urgen (segment High), 2 = Mid, 3 = Low
# ─────────────────────────────────────────────────────────────────────────────
REC_MAP = {
    "tenure": {
        1: ("Masa kontrak sangat pendek — tawarkan diskon loyalitas 20% dan program reward eksklusif untuk perpanjangan 12 bulan.", "🔴 Prioritas Tinggi"),
        2: ("Tenure menengah — perkenalkan program poin loyalitas dan benefit eksklusif member jangka panjang.", "🟡 Prioritas Sedang"),
        3: ("Pelanggan sudah cukup lama — apresiasi dengan hadiah ulang tahun langganan dan akses fitur premium gratis 1 bulan.", "🔵 Prioritas Rendah"),
    },
    "Contract": {
        1: ("Kontrak month-to-month sangat berisiko — tawarkan diskon 30% upgrade ke One Year atau Two Year contract.", "🔴 Prioritas Tinggi"),
        2: ("Dorong upgrade kontrak dengan cashback atau gratis bulan pertama setelah upgrade.", "🟡 Prioritas Sedang"),
        3: ("Informasikan benefit kontrak jangka panjang melalui email newsletter bulanan.", "🔵 Prioritas Rendah"),
    },
    "MonthlyCharges": {
        1: ("Tagihan bulanan tinggi — review paket dan tawarkan bundling lebih hemat atau diskon retensi 15%.", "🔴 Prioritas Tinggi"),
        2: ("Tawarkan paket yang lebih sesuai kebutuhan pelanggan atau opsi cicilan.", "🟡 Prioritas Sedang"),
        3: ("Kirimkan summary nilai yang didapat pelanggan vs biaya yang dibayar setiap bulan.", "🔵 Prioritas Rendah"),
    },
    "InternetService": {
        1: ("Layanan internet saat ini kurang memuaskan — tawarkan upgrade Fiber Optic dengan harga spesial.", "🔴 Prioritas Tinggi"),
        2: ("Tawaran upgrade layanan internet dengan trial gratis 1 bulan.", "🟡 Prioritas Sedang"),
        3: ("Informasikan opsi upgrade internet yang tersedia di area pelanggan.", "🔵 Prioritas Rendah"),
    },
    "TechSupport": {
        1: ("Tidak ada Tech Support — tawarkan free trial Tech Support Premium 3 bulan.", "🔴 Prioritas Tinggi"),
        2: ("Aktifkan Tech Support dasar gratis selama 2 bulan sebagai benefit retensi.", "🟡 Prioritas Sedang"),
        3: ("Kirimkan panduan self-service dan FAQ untuk meningkatkan kepuasan.", "🔵 Prioritas Rendah"),
    },
    "OnlineSecurity": {
        1: ("Tidak ada perlindungan Online Security — berikan free activation Online Security 3 bulan.", "🔴 Prioritas Tinggi"),
        2: ("Tawarkan paket bundling keamanan digital dengan harga terjangkau.", "🟡 Prioritas Sedang"),
        3: ("Edukasi manfaat keamanan digital melalui konten email.", "🔵 Prioritas Rendah"),
    },
    "PaperlessBilling": {
        1: ("Aktifkan paperless billing dan berikan kredit $5 per bulan sebagai insentif.", "🔴 Prioritas Tinggi"),
        2: ("Dorong transisi ke paperless billing dengan bonus poin loyalty.", "🟡 Prioritas Sedang"),
        3: ("Informasikan keuntungan ekologis dan kemudahan paperless billing.", "🔵 Prioritas Rendah"),
    },
    "PaymentMethod": {
        1: ("Metode pembayaran manual — tawarkan diskon 5% untuk beralih ke auto-pay.", "🔴 Prioritas Tinggi"),
        2: ("Fasilitasi pendaftaran auto-pay dengan panduan mudah dan insentif.", "🟡 Prioritas Sedang"),
        3: ("Ingatkan manfaat auto-pay: tidak pernah telat bayar, poin ekstra.", "🔵 Prioritas Rendah"),
    },
    "TotalCharges": {
        1: ("Total tagihan besar tapi risiko churn tinggi — tawarkan program cicilan atau restrukturisasi tagihan.", "🔴 Prioritas Tinggi"),
        2: ("Berikan summary nilai layanan untuk justifikasi total tagihan.", "🟡 Prioritas Sedang"),
        3: ("Kirimkan laporan penggunaan dan nilai yang diterima pelanggan.", "🔵 Prioritas Rendah"),
    },
    "CLTV": {
        1: ("CLTV tinggi — pelanggan sangat berharga. Assign dedicated account manager dan prioritaskan penanganan keluhan.", "🔴 Prioritas Tinggi"),
        2: ("Tawarkan program VIP dengan akses ke fitur eksklusif.", "🟡 Prioritas Sedang"),
        3: ("Sertakan dalam program loyalitas tier silver/gold.", "🔵 Prioritas Rendah"),
    },
    # Default fallback
    "DEFAULT": {
        1: ("Segera hubungi pelanggan untuk survei kepuasan dan tawarkan solusi retensi personal.", "🔴 Prioritas Tinggi"),
        2: ("Kirimkan penawaran khusus retensi melalui email/SMS.", "🟡 Prioritas Sedang"),
        3: ("Sertakan dalam program loyalty rutin dan pantau terus aktivitas.", "🔵 Prioritas Rendah"),
    },
}

LEVEL_LABELS = {1: "Urgent", 2: "Medium Priority", 3: "Low Priority"}


def _get_recommendation(feature_name: str, segment: str) -> dict:
    """Cari rekomendasi berdasarkan nama fitur dan segment."""
    tier = {"High": 1, "Mid": 2, "Low": 3}.get(segment, 3)

    # Cari exact atau partial match di REC_MAP
    matched_key = None
    for key in REC_MAP:
        if key == "DEFAULT":
            continue
        if key.lower() in feature_name.lower() or feature_name.lower().startswith(key.lower()):
            matched_key = key
            break

    rec_dict = REC_MAP.get(matched_key, REC_MAP["DEFAULT"])
    text, level_label = rec_dict[tier]
    return {
        "feature_name": feature_name,
        "recommendation": text,
        "level_label": level_label,
    }


def prob_to_churnscore(prob: float) -> int:
    """
    Piecewise scaler: probability → ChurnScore 0–100
    Low (0–33) | Mid (34–66) | High (67–100)
    """
    if prob < 0.33:
        score = int(prob / 0.33 * 33)
    elif prob < 0.67:
        score = 33 + int((prob - 0.33) / 0.34 * 33)
    else:
        score = 66 + int((prob - 0.67) / 0.33 * 34)
    return min(max(score, 0), 100)


def get_churn_segment(churn_score: int) -> str:
    if churn_score >= 67:
        return "High"
    elif churn_score >= 34:
        return "Mid"
    return "Low"


def _preprocess_row(row: pd.Series, artifacts: dict) -> np.ndarray:
    """Preprocess single row: encode → scale → select features."""
    scaler   = artifacts["scaler"]
    selector = artifacts["selector"]
    all_feat = artifacts["feature_names_all"]

    # Build DataFrame dengan kolom raw
    df_row = pd.DataFrame([row[FEATURE_COLS]])

    # Encode categorical
    cat_cols = df_row.select_dtypes(include="object").columns.tolist()
    df_enc   = pd.get_dummies(df_row, columns=cat_cols, drop_first=False)

    # Align columns dengan training set
    for col in all_feat:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    df_enc = df_enc[all_feat].astype(float)

    # Scale & select
    X_sc  = scaler.transform(df_enc)
    X_sel = selector.transform(X_sc)
    return X_sel


def predict_single(row: pd.Series, artifacts: dict) -> dict:
    """
    Prediksi satu baris data.

    Returns dict:
        probability, churn_score, segment, threshold,
        shap_display (list of dict), recommendations (list of dict)
    """
    model     = artifacts["model"]
    sel_names = artifacts["selected_feature_names"]
    threshold = artifacts["model_metrics"]["threshold"]

    X_sel = _preprocess_row(row, artifacts)
    prob  = float(model.predict_proba(X_sel)[0, 1])
    cs    = prob_to_churnscore(prob)
    seg   = get_churn_segment(cs)

    # SHAP individual
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_sel)
    if isinstance(sv, list):
        sv = sv[1]  # class 1
    sv = sv[0]  # first (only) row

    # Build shap_display sorted by abs value
    shap_items = sorted(
        [{"name": n, "value": float(v)} for n, v in zip(sel_names, sv)],
        key=lambda x: abs(x["value"]),
        reverse=True,
    )
    for item in shap_items:
        item["direction"] = "up" if item["value"] > 0 else "dn"

    # Top-3 recommendations
    top3 = shap_items[:3]
    recs = []
    for rank, item in enumerate(top3, start=1):
        rec = _get_recommendation(item["name"], seg)
        rec["rank"] = rank
        recs.append(rec)

    return {
        "probability":     prob,
        "churn_score":     cs,
        "segment":         seg,
        "threshold":       threshold,
        "shap_display":    shap_items,
        "recommendations": recs,
    }


def predict_bulk(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Prediksi bulk dari DataFrame.
    Mengembalikan df asli + kolom hasil prediksi.
    """
    model     = artifacts["model"]
    scaler    = artifacts["scaler"]
    selector  = artifacts["selector"]
    all_feat  = artifacts["feature_names_all"]
    threshold = artifacts["model_metrics"]["threshold"]

    # Encode
    cat_cols = df[FEATURE_COLS].select_dtypes(include="object").columns.tolist()
    df_enc   = pd.get_dummies(df[FEATURE_COLS], columns=cat_cols, drop_first=False)

    for col in all_feat:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    df_enc = df_enc[all_feat].astype(float)

    X_sc   = scaler.transform(df_enc)
    X_sel  = selector.transform(X_sc)
    probas = model.predict_proba(X_sel)[:, 1]

    df["Churn_Probability"] = probas
    df["Churn_Score"]       = [prob_to_churnscore(p) for p in probas]
    df["Churn_Segment"]     = [get_churn_segment(s) for s in df["Churn_Score"]]
    df["Predicted_Label"]   = ["Churn" if p >= threshold else "No Churn" for p in probas]
    return df


def validate_upload(df: pd.DataFrame):
    """Validasi kolom CSV yang diupload."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return False, f"Kolom tidak ditemukan: {missing}"
    if len(df) == 0:
        return False, "File kosong."
    return True, "OK"
