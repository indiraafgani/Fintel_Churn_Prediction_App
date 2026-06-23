"""
FINTel — Prediction & Recommendation Engine
Single unified Logistic Regression model (no cold/non-cold split).
Churn segmentation: Low / Mid / High based on ChurnScore binning.
Per-feature SHAP recommendations: top-3 SHAP features → tiered actions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# ─── Feature columns (input to model) ─────────────────────────────────────────
FEATURE_COLS: List[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "CLTV",
]

# ─── Churn segment thresholds (from ChurnScore binning) ───────────────────────
SEGMENT_LOW_MAX  = 33   # score 0–33   → Low
SEGMENT_HIGH_MIN = 67   # score 67–100 → High
                        # score 34–66  → Mid

# ─── Probability → ChurnScore piecewise scaler ────────────────────────────────
# Three anchor points: prob=0 → 0,  prob=threshold → 50,  prob=1 → 100
def prob_to_churnscore(prob: float, threshold: float) -> int:
    if threshold <= 0 or threshold >= 1:
        return int(np.clip(round(prob * 100), 0, 100))
    pivot_score = 50
    if prob <= threshold:
        score = (prob / threshold) * pivot_score
    else:
        score = pivot_score + ((prob - threshold) / (1 - threshold)) * (100 - pivot_score)
    return int(np.clip(round(score), 0, 100))


def get_churn_segment(churn_score: int) -> str:
    """Map churn score (0–100) → Low / Mid / High."""
    if churn_score <= SEGMENT_LOW_MAX:
        return "Low"
    elif churn_score < SEGMENT_HIGH_MIN:
        return "Mid"
    return "High"


# ─── Per-feature SHAP recommendations ─────────────────────────────────────────
# Each feature has 3 tiered actions (level 1 = most urgent / highest-ranked SHAP).
# Keys must match selected_feature_names from the trained model.
# For binary-encoded features, we map the "base" concept back to the original column.
# Recommendations are segment-aware where noted with [Low] [Mid] [High] prefix variants.

FEATURE_REC_MAP: Dict[str, Dict[str, str]] = {
    # ── Tenure ──────────────────────────────────────────────────────────────
    "tenure": {
        "1": "Luncurkan program onboarding intensif bulan 1–3: welcome call, walkthrough produk, dan check-in mingguan dengan customer success.",
        "2": "Berikan insentif loyalty pada milestone: hadiah kecil di bulan 3, 6, dan 12 untuk memperkuat kebiasaan menggunakan layanan.",
        "3": "Kirim survei kepuasan singkat (3 pertanyaan) pada akhir bulan pertama untuk mendeteksi friksi lebih awal.",
    },
    "TotalCharges": {
        "1": "Evaluasi nilai lifetime pelanggan ini — jika CLTV tinggi, eskalasikan ke dedicated account manager segera.",
        "2": "Tawarkan program cicilan atau restrukturisasi tagihan agar total biaya terasa lebih ringan dan mengurangi risiko payment shock.",
        "3": "Tampilkan ringkasan manfaat kumulatif yang sudah diterima pelanggan sesuai total biaya yang dibayarkan.",
    },
    "MonthlyCharges": {
        "1": "Tawarkan paket bundling yang lebih hemat sesuai layanan aktif — potensi penghematan 10–20% per bulan.",
        "2": "Komunikasikan value proposition secara eksplisit: kirim laporan bulanan yang membandingkan biaya vs manfaat yang didapat.",
        "3": "Review apakah ada add-on yang tidak digunakan — tawarkan downgrade paket yang lebih sesuai kebutuhan.",
    },
    # ── Contract ────────────────────────────────────────────────────────────
    "Contract_0": {
        "1": "Berikan penawaran upgrade ke kontrak tahunan dengan diskon 15–20% — sampaikan via personal outreach dari customer success.",
        "2": "Tampilkan simulasi penghematan biaya jika beralih dari month-to-month ke kontrak 1 tahun.",
        "3": "Kirim reminder 7 hari sebelum tanggal renewal dengan tawaran eksklusif untuk penguncian kontrak lebih panjang.",
    },
    "Contract_1": {
        "1": "Dorong upgrade ke kontrak 2 tahun dengan insentif tambahan: bonus kuota atau diskon bulan pertama gratis.",
        "2": "Highlight stabilitas harga jangka panjang — pelanggan kontrak 2 tahun tidak terdampak kenaikan tarif.",
        "3": "Tawarkan early renewal 3 bulan sebelum kontrak habis dengan harga terkunci.",
    },
    # ── Internet Service ────────────────────────────────────────────────────
    "InternetService_0": {
        "1": "Tawarkan upgrade ke Fiber Optic dengan trial gratis 1 bulan — kecepatan dan reliabilitas yang jauh lebih baik.",
        "2": "Bandingkan secara konkret kecepatan layanan saat ini vs Fiber Optic melalui infografis personal.",
        "3": "Kirim penawaran bundling Fiber Optic + TV Streaming dengan harga kompetitif.",
    },
    # ── Online Security ──────────────────────────────────────────────────────
    "OnlineSecurity_0": {
        "1": "Aktifkan trial gratis OnlineSecurity 30 hari — sertakan demo singkat manfaat perlindungan data.",
        "2": "Kirim email edukasi tentang risiko keamanan digital dan bagaimana OnlineSecurity melindungi pelanggan.",
        "3": "Bundling OnlineSecurity dengan layanan yang sudah aktif pada harga diskon 25%.",
    },
    "OnlineSecurity_1": {
        "1": "Pelanggan sudah aktif menggunakan OnlineSecurity — highlight nilai fitur ini dan cross-sell OnlineBackup.",
        "2": "Tawarkan upgrade ke paket keamanan premium yang mencakup DeviceProtection sekaligus.",
        "3": "Kirim laporan bulanan aktivitas keamanan akun untuk meningkatkan perceived value layanan.",
    },
    # ── Online Backup ────────────────────────────────────────────────────────
    "OnlineBackup_1": {
        "1": "Pelanggan sudah menggunakan OnlineBackup — dorong ekspansi ke DeviceProtection untuk perlindungan end-to-end.",
        "2": "Kirim reminder manfaat backup cloud dan statistik pemulihan data sebagai edukasi retention.",
        "3": "Cross-sell TechSupport sebagai pelengkap OnlineBackup untuk pengalaman digital yang lebih aman.",
    },
    # ── Device Protection ──────────────────────────────────────────────────
    "DeviceProtection_0": {
        "1": "Tawarkan trial gratis DeviceProtection 30 hari — sampaikan nilai klaim perangkat rata-rata vs biaya bulanan.",
        "2": "Kirim simulasi kerugian finansial jika perangkat rusak tanpa proteksi — ROI yang jelas.",
        "3": "Bundling DeviceProtection + TechSupport dengan harga paket lebih hemat dari beli satuan.",
    },
    # ── Tech Support ─────────────────────────────────────────────────────────
    "TechSupport_0": {
        "1": "Aktifkan akses TechSupport gratis selama 30 hari — berikan sesi onboarding 1-on-1 dengan teknisi.",
        "2": "Kirim statistik rata-rata waktu resolusi masalah dengan TechSupport vs tanpa dukungan.",
        "3": "Tawarkan paket TechSupport + OnlineSecurity dalam bundling hemat untuk pelanggan segmen ini.",
    },
    "TechSupport_1": {
        "1": "Pelanggan sudah aktif dengan TechSupport — dorong upgrade ke paket premium dengan response time lebih cepat.",
        "2": "Kirim survei kepuasan TechSupport dan gunakan feedback untuk personalisasi layanan berikutnya.",
        "3": "Cross-sell DeviceProtection sebagai natural complement dari TechSupport yang sudah digunakan.",
    },
    # ── Streaming Movies ─────────────────────────────────────────────────────
    "StreamingMovies_1": {
        "1": "Pelanggan aktif streaming — tawarkan upgrade bandwidth atau paket multi-device untuk pengalaman lebih baik.",
        "2": "Kirim rekomendasi konten personal dan notifikasi rilis terbaru untuk meningkatkan engagement.",
        "3": "Bundling StreamingTV + StreamingMovies dalam paket entertainment lengkap dengan harga bundling.",
    },
    # ── Payment Method ───────────────────────────────────────────────────────
    "PaymentMethod_0": {
        "1": "Tawarkan beralih ke pembayaran otomatis (bank transfer/kartu kredit) dengan cashback bulan pertama.",
        "2": "Kirim reminder tagihan lebih awal + kemudahan link pembayaran satu klik untuk mengurangi friction.",
        "3": "Program reward poin untuk pelanggan yang membayar tepat waktu selama 3 bulan berturut-turut.",
    },
    "PaymentMethod_1": {
        "1": "Electronic check memiliki korelasi churn lebih tinggi — insentifkan migrasi ke auto-pay dengan diskon tagihan.",
        "2": "Komunikasikan keamanan dan kemudahan pembayaran otomatis melalui email edukatif.",
        "3": "Tawarkan cashback 5% tagihan bulan depan untuk pelanggan yang beralih ke auto-pay bulan ini.",
    },
    "PaymentMethod_2": {
        "1": "Tawarkan reward loyalty khusus untuk pelanggan yang mempertahankan metode pembayaran otomatis.",
        "2": "Kirim konfirmasi pembayaran yang jelas dan ringkasan manfaat bulan ini untuk memperkuat perceived value.",
        "3": "Aktifkan notifikasi real-time setiap pembayaran berhasil untuk meningkatkan kepercayaan.",
    },
    # ── Paperless Billing ────────────────────────────────────────────────────
    "PaperlessBilling_Yes": {
        "1": "Pelanggan paperless billing lebih mudah dijangkau secara digital — aktifkan push notification untuk penawaran retensi.",
        "2": "Kirim laporan penggunaan bulanan yang interaktif via email untuk meningkatkan engagement.",
        "3": "Tawarkan e-voucher kecil sebagai reward atas preferensi paperless (eco-friendly incentive).",
    },
    # ── Demographics ─────────────────────────────────────────────────────────
    "SeniorCitizen_Yes": {
        "1": "Siapkan dedicated senior support line dengan waktu tunggu prioritas dan agen yang terlatih untuk segmen lansia.",
        "2": "Tawarkan paket yang disederhanakan — lebih sedikit pilihan, lebih mudah dipahami, dengan harga transparan.",
        "3": "Kirim panduan penggunaan layanan dalam format yang mudah dibaca (font lebih besar, langkah lebih sederhana).",
    },
    "Partner_Yes": {
        "1": "Tawarkan paket keluarga atau pasangan — bundle dua layanan dengan diskon gabungan 15%.",
        "2": "Kirim penawaran referral: pelanggan yang mengajak pasangan/keluarga mendapat cashback.",
        "3": "Highlight manfaat berbagi akun streaming atau layanan multi-device untuk segmen ini.",
    },
    "Dependents_Yes": {
        "1": "Tawarkan paket family plan dengan kuota lebih besar dan multi-device untuk mendukung kebutuhan keluarga.",
        "2": "Kirim konten edukasi tentang parental control dan keamanan internet anak — tambah perceived value.",
        "3": "Program loyalty family: poin reward kumulatif untuk seluruh anggota keluarga dalam satu akun.",
    },
}

# Fallback untuk feature yang tidak ada di map
_FALLBACK_REC = {
    "1": "Lakukan outreach personal dari tim customer success untuk memahami kebutuhan spesifik pelanggan ini.",
    "2": "Tawarkan konsultasi produk gratis untuk memastikan pelanggan menggunakan layanan yang paling sesuai.",
    "3": "Kirim survei kepuasan dan tindak lanjuti feedback dalam 48 jam untuk menunjukkan komitmen pelayanan.",
}


def get_feature_recommendations(
    shap_feature_name: str,
    rank: int,
) -> str:
    """
    Given a SHAP feature name and its rank (1/2/3), return the appropriate
    tiered recommendation text.
    rank 1 = most important (highest |SHAP|) → action level "1"
    rank 2 = second                           → action level "2"
    rank 3 = third                            → action level "3"
    """
    level_key = str(rank)
    rec_dict  = FEATURE_REC_MAP.get(shap_feature_name, _FALLBACK_REC)
    return rec_dict.get(level_key, _FALLBACK_REC[level_key])


def get_shap_top3_recommendations(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, str]]:
    """
    Given per-customer SHAP values and feature names,
    return top-3 recommendations in ranked order.

    Returns list of 3 dicts:
        [{rank, feature_name, shap_value, direction, recommendation, level_label}, ...]
    """
    abs_shap = np.abs(shap_values)
    top3_idx = np.argsort(abs_shap)[::-1][:3]

    results = []
    level_labels = {1: "Priority action", 2: "Supporting action", 3: "Additional action"}

    for rank, idx in enumerate(top3_idx, start=1):
        fname = feature_names[idx]
        val   = float(shap_values[idx])
        results.append({
            "rank":          rank,
            "feature_name":  fname,
            "shap_value":    round(val, 4),
            "direction":     "up" if val > 0 else "down",
            "recommendation": get_feature_recommendations(fname, rank),
            "level_label":   level_labels[rank],
        })
    return results


# ─── Single-customer inference ────────────────────────────────────────────────

def transform_row(row: pd.Series, artifacts: Dict[str, Any]) -> np.ndarray:
    """Transform a single customer row through preprocessor → scaler → feature_selection."""
    model        = artifacts["model"]
    preprocessor = model.named_steps["preprocessor"]
    scaler       = model.named_steps["scaler"]
    feat_sel     = model.named_steps["feature_selection"]

    X   = pd.DataFrame([row[FEATURE_COLS]])
    Xp  = preprocessor.transform(X)
    Xs  = scaler.transform(Xp)
    Xf  = feat_sel.transform(Xs)
    return Xf


def predict_single(row: pd.Series, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full inference for a single customer.
    Returns probability, churn_score, segment (Low/Mid/High),
    SHAP values, and per-customer tiered recommendations.
    """
    import shap as _shap

    model         = artifacts["model"]
    threshold     = artifacts["threshold"]
    feature_names = artifacts["selected_feature_names"]
    clf           = model.named_steps["classifier"]
    X_background  = artifacts["X_sel_background"]

    X_sel = transform_row(row, artifacts)
    prob  = float(model.predict_proba(pd.DataFrame([row[FEATURE_COLS]]))[0, 1])
    churn_score = prob_to_churnscore(prob, threshold)
    segment     = get_churn_segment(churn_score)
    prediction  = "Churn" if prob >= threshold else "No Churn"

    # Per-customer SHAP
    explainer  = _shap.LinearExplainer(clf, X_background)
    shap_vals  = explainer.shap_values(X_sel)[0]  # shape (n_features,)

    # Top-3 recs
    recs = get_shap_top3_recommendations(shap_vals, feature_names)

    # All SHAP for waterfall display (top 8 by |SHAP|)
    top8_idx  = np.argsort(np.abs(shap_vals))[::-1][:8]
    shap_display = [
        {
            "name":      feature_names[i],
            "value":     round(float(shap_vals[i]), 4),
            "direction": "up" if shap_vals[i] > 0 else "down",
        }
        for i in top8_idx
    ]

    return {
        "probability":   round(prob, 4),
        "churn_score":   churn_score,
        "segment":       segment,
        "prediction":    prediction,
        "threshold":     round(threshold, 4),
        "shap_display":  shap_display,
        "recommendations": recs,
    }


# ─── Bulk inference ───────────────────────────────────────────────────────────

def predict_bulk(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Run inference on a full DataFrame.
    Adds columns: Churn_Probability, Churn_Score, Churn_Segment, Predicted_Label.
    No per-row SHAP (too slow for bulk) — adds global top-feature context instead.
    """
    model     = artifacts["model"]
    threshold = artifacts["threshold"]

    X     = df[FEATURE_COLS].copy()
    probs = model.predict_proba(X)[:, 1]

    scores   = [prob_to_churnscore(p, threshold) for p in probs]
    segments = [get_churn_segment(s) for s in scores]
    labels   = ["Churn" if p >= threshold else "No Churn" for p in probs]

    result = df.copy()
    result["Churn_Probability"] = np.round(probs, 4)
    result["Churn_Score"]       = scores
    result["Churn_Segment"]     = segments
    result["Predicted_Label"]   = labels
    return result


def validate_upload(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that an uploaded CSV has all required feature columns."""
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        return False, f"Kolom yang hilang: {sorted(missing)}"
    return True, "OK"
