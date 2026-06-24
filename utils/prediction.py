"""
utils/prediction.py
===================
Prediction engine for FINTel Churn Intelligence Dashboard.

Model   : Logistic Regression (CLASS_WEIGHT, best from benchmarking per notebook)
SHAP    : LinearExplainer
Encoding: OHE (drop='first') + TargetEncoder(City) + RobustScaler + SelectKBest(k=20)
Segments: Rendah / Sedang / Tinggi (dari best_threshold, dibagi 3 sama rata di atas threshold)
Recs    : CAMPAIGN_CATALOG — per feature × 3 tiers (high/medium/low)
          mapped from top-3 SHAP features yang mendorong churn (SHAP > 0) per customer
CustomerID: from cell 42 of notebook — customer_ids = df['customerID'].copy()
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Any

# ─── Raw feature columns fed to the model (X, per cell 146 of notebook) ──────
FEATURE_COLS: List[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "City",
]

# ─── Segmentasi Churn ──────────
def prob_to_churnscore(prob: float, threshold: float) -> int:
    """
    Konversi probability ke ChurnScore 0–100 (prob × 100).
    """
    return int(np.clip(round(prob * 100), 0, 100))

def get_churn_segment(score: int, threshold: float) -> str:
    """
    Segmentasi Rendah / Sedang / Tinggi berdasarkan best_threshold dari notebook.
    Range di atas threshold dibagi 3 sama rata.
    """
    prob         = score / 100
    segment_size = (1 - threshold) / 3
    low_limit    = threshold + segment_size
    medium_limit = threshold + (2 * segment_size)

    if prob < threshold:
        return "Rendah"
    elif prob < low_limit:
        return "Rendah"
    elif prob < medium_limit:
        return "Sedang"
    return "Tinggi"

# ─── Campaign Catalog — per feature × 3 tiers (from notebook cell 187) ───────
CAMPAIGN_CATALOG: Dict[str, Dict[str, Dict[str, str]]] = {
    "tenure": {
        "high":   {"nama": "Program Loyalitas Premium",
                   "aksi": "Tawarkan kontrak 2 tahun + cashback 20% tagihan 3 bulan pertama + dedicated account manager."},
        "medium": {"nama": "Reward Tenure Eksklusif",
                   "aksi": "Berikan poin loyalitas yang bisa ditukar diskon tagihan atau upgrade layanan gratis."},
        "low":    {"nama": "Notifikasi Apresiasi",
                   "aksi": "Kirim email/SMS ucapan terima kasih atas kesetiaan + voucher diskon kecil bulan depan."},
    },
    "Contract": {
        "high":   {"nama": "Upgrade Kontrak Berhadiah",
                   "aksi": "Tawarkan upgrade ke kontrak tahunan/2 tahun dengan bonus gadget (router/tablet)."},
        "medium": {"nama": "Diskon Lock-in Kontrak",
                   "aksi": "Diskon 15% tagihan bulanan jika pelanggan pindah ke kontrak 1 tahun, berlaku 6 bulan."},
        "low":    {"nama": "Edukasi Manfaat Kontrak",
                   "aksi": "Kirim materi komparasi keuntungan kontrak panjang vs bulanan melalui email/WhatsApp."},
    },
    "MonthlyCharges": {
        "high":   {"nama": "Repricing Personal",
                   "aksi": "Turunkan tarif bulanan 20-25% selama 6 bulan via negosiasi langsung dengan retention agent."},
        "medium": {"nama": "Bundling Hemat",
                   "aksi": "Tawarkan paket bundling (internet + TV/telepon) dengan harga lebih rendah dari tagihan saat ini."},
        "low":    {"nama": "Transparansi Tagihan",
                   "aksi": "Kirim rincian nilai yang didapat dari layanan + tips mengoptimalkan paket yang sudah ada."},
    },
    "TechSupport": {
        "high":   {"nama": "Premium Tech Support Gratis",
                   "aksi": "Upgrade ke layanan tech support 24/7 premium gratis selama 6 bulan + kunjungan teknisi on-site."},
        "medium": {"nama": "Aktivasi Tech Support Bersubsidi",
                   "aksi": "Aktifkan paket tech support dengan diskon 50% untuk 3 bulan pertama."},
        "low":    {"nama": "Panduan Self-Service",
                   "aksi": "Kirim video tutorial + akses ke portal bantuan mandiri + FAQ troubleshooting umum."},
    },
    "OnlineSecurity": {
        "high":   {"nama": "Paket Keamanan Digital All-in-One",
                   "aksi": "Aktifkan bundling OnlineSecurity + VPN + anti-malware premium gratis 6 bulan."},
        "medium": {"nama": "Trial Security Premium",
                   "aksi": "Aktifkan OnlineSecurity gratis 3 bulan, auto-renew dengan diskon 30%."},
        "low":    {"nama": "Edukasi Keamanan Digital",
                   "aksi": "Kirim tips keamanan online + highlight risiko tanpa proteksi melalui email."},
    },
    "OnlineBackup": {
        "high":   {"nama": "Cloud Backup Premium Gratis",
                   "aksi": "Upgrade ke paket backup cloud 1TB gratis selama 6 bulan + bantuan migrasi data oleh teknisi."},
        "medium": {"nama": "Aktivasi Backup Bersubsidi",
                   "aksi": "Aktifkan OnlineBackup dengan diskon 50% selama 3 bulan pertama."},
        "low":    {"nama": "Awareness Backup Data",
                   "aksi": "Kirim infografis risiko kehilangan data + panduan aktivasi backup mandiri."},
    },
    "DeviceProtection": {
        "high":   {"nama": "Proteksi Perangkat All-Risk",
                   "aksi": "Aktifkan asuransi perangkat all-risk gratis 6 bulan, termasuk pencurian dan kerusakan."},
        "medium": {"nama": "Trial Device Protection",
                   "aksi": "Aktifkan DeviceProtection gratis 3 bulan, perpanjang dengan diskon 40%."},
        "low":    {"nama": "Edukasi Perlindungan Perangkat",
                   "aksi": "Kirim simulasi biaya perbaikan tanpa proteksi + cara aktivasi di aplikasi."},
    },
    "StreamingTV": {
        "high":   {"nama": "Bundling Streaming Premium",
                   "aksi": "Aktifkan StreamingTV + StreamingMovies + akses platform OTT partner gratis 6 bulan."},
        "medium": {"nama": "Upgrade Paket Streaming",
                   "aksi": "Tambah StreamingTV ke paket existing dengan diskon 50% selama 3 bulan."},
        "low":    {"nama": "Preview Konten Eksklusif",
                   "aksi": "Kirim kupon 7 hari akses streaming gratis + highlight konten populer bulan ini."},
    },
    "StreamingMovies": {
        "high":   {"nama": "Paket Hiburan Total",
                   "aksi": "Aktifkan StreamingMovies + StreamingTV + akses bioskop virtual premium gratis 6 bulan."},
        "medium": {"nama": "Aktivasi Streaming Movies Bersubsidi",
                   "aksi": "Tambah StreamingMovies dengan diskon 50% selama 3 bulan pertama."},
        "low":    {"nama": "Trial Film Eksklusif",
                   "aksi": "Kirim akses gratis 5 film pilihan + highlight konten baru bulan ini."},
    },
    "InternetService": {
        "high":   {"nama": "Upgrade Internet Fiber Gratis",
                   "aksi": "Upgrade ke paket fiber optic tercepat gratis 3 bulan + instalasi gratis + router baru."},
        "medium": {"nama": "Speed Boost Bersubsidi",
                   "aksi": "Upgrade kecepatan internet ke tier berikutnya dengan diskon 40% selama 6 bulan."},
        "low":    {"nama": "Komparasi Paket Internet",
                   "aksi": "Kirim perbandingan kecepatan dan harga paket + simulasi penghematan jika upgrade."},
    },
    "PaymentMethod": {
        "high":   {"nama": "Cashback Auto-Payment Premium",
                   "aksi": "Tawarkan cashback 10% setiap bulan jika beralih ke auto-payment kartu kredit/debit."},
        "medium": {"nama": "Insentif Pindah Metode Bayar",
                   "aksi": "Diskon untuk 3 bulan pertama jika beralih ke pembayaran otomatis."},
        "low":    {"nama": "Edukasi Kemudahan Auto-Payment",
                   "aksi": "Kirim panduan cara setup auto-payment + highlight keuntungan tidak perlu ingat tanggal bayar."},
    },
    "PaperlessBilling": {
        "high":   {"nama": "Insentif Beralih ke Digital Billing",
                   "aksi": "Berikan voucher + diskon tagihan bulanan jika beralih ke e-billing."},
        "medium": {"nama": "Aktivasi E-Billing Berhadiah",
                   "aksi": "Aktivasi paperless billing dan dapatkan poin reward."},
        "low":    {"nama": "Awareness E-Billing",
                   "aksi": "Kirim manfaat tagihan digital (lebih cepat, ramah lingkungan) + link aktivasi 1-klik."},
    },
    "PhoneService": {
        "high":   {"nama": "Paket Telepon Unlimited Premium",
                   "aksi": "Upgrade ke paket telepon unlimited gratis 6 bulan + gratis roaming ke 5 negara."},
        "medium": {"nama": "Bonus Pulsa Eksklusif",
                   "aksi": "Tambah kuota telepon 500 menit/bulan gratis selama 3 bulan."},
        "low":    {"nama": "Reminder Manfaat Layanan",
                   "aksi": "Kirim ringkasan penggunaan telepon bulan lalu + tips memaksimalkan paket saat ini."},
    },
    "MultipleLines": {
        "high":   {"nama": "Family Plan Premium",
                   "aksi": "Tawarkan paket keluarga dengan diskon 30% untuk semua nomor + bonus kuota bersama."},
        "medium": {"nama": "Tambah Nomor Bersubsidi",
                   "aksi": "Aktifkan nomor kedua dengan diskon 50% selama 3 bulan pertama."},
        "low":    {"nama": "Penawaran Multi-Line",
                   "aksi": "Kirim informasi keuntungan paket multi-line + simulasi penghematan untuk keluarga."},
    },
    "SeniorCitizen": {
        "high":   {"nama": "Program Senior VIP",
                   "aksi": "Dedicated customer service senior + kunjungan teknisi rumah + diskon tagihan bulanan."},
        "medium": {"nama": "Paket Senior Spesial",
                   "aksi": "Tawarkan paket khusus lansia dengan harga lebih terjangkau + bantuan teknis prioritas."},
        "low":    {"nama": "Panduan Digital untuk Senior",
                   "aksi": "Kirim panduan penggunaan layanan versi cetak/sederhana + nomor hotline khusus senior."},
    },
    "Partner": {
        "high":   {"nama": "Couples/Family Bundle Premium",
                   "aksi": "Tawarkan paket pasangan/keluarga dengan diskon 35% untuk dua nomor + bonus layanan."},
        "medium": {"nama": "Referral Bonus Pasangan",
                   "aksi": "Ajak pasangan bergabung dan dapatkan diskon untuk keduanya."},
        "low":    {"nama": "Penawaran Paket Bersama",
                   "aksi": "Kirim informasi paket shared data + simulasi penghematan jika bergabung dengan pasangan."},
    },
    "Dependents": {
        "high":   {"nama": "Family Protection Plan",
                   "aksi": "Paket keluarga all-in-one: internet + TV + telepon untuk seluruh anggota keluarga, diskon 30%."},
        "medium": {"nama": "Kuota Keluarga Bersubsidi",
                   "aksi": "Tambah kuota data shared untuk anak/tanggungan dengan harga 50% lebih murah."},
        "low":    {"nama": "Edukasi Paket Keluarga",
                   "aksi": "Kirim simulasi penghematan paket keluarga vs paket individual."},
    },
    "TotalCharges": {
        "high":   {"nama": "Restrukturisasi Tagihan Premium",
                   "aksi": "Tawarkan cicilan total tagihan + diskon jika berkomitmen kontrak 2 tahun."},
        "medium": {"nama": "Review Tagihan Personal",
                   "aksi": "Sesi konsultasi 1-on-1 dengan retention agent untuk mereview dan mengoptimalkan tagihan."},
        "low":    {"nama": "Laporan Nilai Layanan",
                   "aksi": "Kirim rincian nilai semua layanan yang sudah dinikmati vs total yang dibayarkan."},
    },
    "gender": {
        "high":   {"nama": "Penawaran Persona Premium",
                   "aksi": "Campaign personalisasi berbasis preferensi demografis + akses program eksklusif."},
        "medium": {"nama": "Konten Relevan Tersegmentasi",
                   "aksi": "Kirim penawaran yang dikurasi berdasarkan preferensi kelompok demografis pelanggan."},
        "low":    {"nama": "Survei Preferensi",
                   "aksi": "Kirim survei singkat preferensi layanan untuk personalisasi komunikasi ke depan."},
    },
    "City": {
        "high":   {"nama": "Penawaran Eksklusif Area Lokal",
                   "aksi": "Kunjungan tim retensi langsung ke wilayah pelanggan + tawaran paket khusus kota tersebut."},
        "medium": {"nama": "Promo Regional",
                   "aksi": "Diskon khusus untuk pengguna di kota/wilayah yang sama dengan jaringan komunitas."},
        "low":    {"nama": "Info Layanan Area",
                   "aksi": "Kirim update jaringan dan layanan terbaru yang tersedia di wilayah pelanggan."},
    },
    "_default": {
        "high":   {"nama": "Retensi Personal Premium",
                   "aksi": "Hubungi langsung oleh retention specialist untuk negosiasi penawaran terbaik."},
        "medium": {"nama": "Tawaran Loyalitas",
                   "aksi": "Kirim penawaran diskon personal berdasarkan histori penggunaan pelanggan."},
        "low":    {"nama": "Engagement Check-in",
                   "aksi": "Kirim survei kepuasan + follow-up dari tim customer success."},
    },
}

TIERS = ["high", "medium", "low"]
TIER_LABELS = {"high": "🔴 Prioritas Tinggi", "medium": "🟡 Prioritas Sedang", "low": "🔵 Prioritas Rendah"}
TIER_LEVEL_LABELS = {"high": "Priority action", "medium": "Supporting action", "low": "Additional action"}


def _match_catalog_key(feature_name: str) -> str:
    """
    Map an encoded feature name back to its CAMPAIGN_CATALOG key.
    e.g. 'Contract_Two year' -> 'Contract'
         'InternetService_Fiber optic' -> 'InternetService'
         'tenure' -> 'tenure'
    """
    for key in CAMPAIGN_CATALOG:
        if key == "_default":
            continue
        if feature_name == key:
            return key
        if feature_name.startswith(key + "_") or feature_name.startswith(key + " "):
            return key
    return "_default"


def get_top3_recommendations(
    shap_values: np.ndarray,
    feature_names: List[str],
    segment: str,
) -> List[Dict[str, str]]:
    """
    Given per-customer SHAP values (1D array) and feature names,
    return 3 personalised campaign recommendations.

    Sesuai notebook: hanya fitur yang mendorong churn (SHAP > 0) yang diambil.
    Fallback ke magnitude tertinggi jika tidak ada fitur positif sama sekali.

    Tier assignment:
        Rank #1 (SHAP positif terbesar) -> tier 'high'   (50% budget)
        Rank #2                         -> tier 'medium'  (35% budget)
        Rank #3                         -> tier 'low'     (15% budget)
    """
    # Ambil hanya fitur yang mendorong churn (SHAP > 0)
    positive_idx = np.where(shap_values > 0)[0]
    sorted_pos   = positive_idx[np.argsort(np.abs(shap_values[positive_idx]))[::-1]]
    top3_idx     = sorted_pos[:3]

    # Fallback kalau tidak ada fitur positif sama sekali
    if len(top3_idx) == 0:
        top3_idx = np.argsort(np.abs(shap_values))[::-1][:3]

    recs = []
    for rank, idx in enumerate(top3_idx, start=1):
        fname    = feature_names[idx]
        val      = float(shap_values[idx])
        tier     = TIERS[rank - 1]
        cat_key  = _match_catalog_key(fname)
        campaign = CAMPAIGN_CATALOG.get(cat_key, CAMPAIGN_CATALOG["_default"])[tier]
        recs.append({
            "rank":           rank,
            "feature_name":   fname,
            "catalog_key":    cat_key,
            "shap_value":     round(val, 4),
            "direction":      "up" if val > 0 else "down",
            "tier":           tier,
            "tier_label":     TIER_LABELS[tier],
            "level_label":    TIER_LEVEL_LABELS[tier],
            "campaign_name":  campaign["nama"],
            "recommendation": campaign["aksi"],
        })
    return recs


# ─── Single-customer inference ────────────────────────────────────────────────

def _transform_row(row: pd.Series, artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess one row through:
    preprocessor (artifacts key) -> scaler -> selector
    Keys 'preprocessor', 'scaler', 'selector' are saved in artifacts
    and map to notebook pipeline steps 'transform', 'scaler', 'feature_select'.
    """
    prep   = artifacts["preprocessor"]
    scaler = artifacts["scaler"]
    sel    = artifacts["selector"]
    X_row  = pd.DataFrame([row[FEATURE_COLS]])
    return sel.transform(scaler.transform(prep.transform(X_row)))


def predict_single(row: pd.Series, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full inference for a single customer row.
    Returns: probability, churn_score, segment, prediction, threshold,
             shap_display (top-8 by abs), recommendations (top-3 SHAP positif).
    """
    model      = artifacts["model"]
    threshold  = artifacts["threshold"]
    feat_names = artifacts["selected_feature_names"]
    clf        = model.named_steps["classifier"]
    X_bg       = artifacts["X_background"]

    X_sel = _transform_row(row, artifacts)
    prob  = float(model.predict_proba(pd.DataFrame([row[FEATURE_COLS]]))[0, 1])
    cs    = prob_to_churnscore(prob, threshold)
    seg   = get_churn_segment(cs, threshold)
    pred  = "Churn" if prob >= threshold else "No Churn"

    # SHAP via LinearExplainer (Logistic Regression)
    explainer = shap.LinearExplainer(clf, X_bg)
    sv_raw    = explainer.shap_values(X_sel)
    sv        = sv_raw[0]   # shape (n_features,)

    # Top-8 untuk display (by abs SHAP value)
    top8_idx = np.argsort(np.abs(sv))[::-1][:8]
    shap_display = [
        {
            "name":      feat_names[i],
            "value":     round(float(sv[i]), 4),
            "direction": "up" if sv[i] > 0 else "down",
        }
        for i in top8_idx
    ]

    # Top-3 rekomendasi — hanya dari SHAP positif (mendorong churn)
    recs = get_top3_recommendations(sv, feat_names, seg)

    return {
        "probability":     round(prob, 4),
        "churn_score":     cs,
        "segment":         seg,
        "prediction":      pred,
        "threshold":       round(threshold, 4),
        "shap_display":    shap_display,
        "recommendations": recs,
    }


# ─── Bulk inference (no per-row SHAP) ────────────────────────────────────────

def predict_bulk(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    model     = artifacts["model"]
    threshold = artifacts["threshold"]

    X      = df[FEATURE_COLS].copy()
    probs  = model.predict_proba(X)[:, 1]
    scores = [prob_to_churnscore(p, threshold) for p in probs]
    segs   = [get_churn_segment(s, threshold) for s in scores]
    labels = ["Churn" if p >= threshold else "No Churn" for p in probs]

    result = df.copy()
    result["Churn_Probability"] = np.round(probs, 4)
    result["Churn_Score"]       = scores
    result["Churn_Segment"]     = segs
    result["Predicted_Label"]   = labels
    return result


def validate_upload(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        return False, f"Kolom yang hilang: {sorted(missing)}"
    if len(df) == 0:
        return False, "File kosong."
    return True, "OK"
