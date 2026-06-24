# FINTel — Sistem Prediksi Churn Pelanggan

> Prediksi churn berbasis Machine Learning & rekomendasi retensi personal berdasarkan SHAP per individu

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintelchurnpredictionapp-01.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

**Final Project** — Purwadhika Digital Technology School · Data Science, ML & AI Bootcamp  
**Anggota:** Akbar Kanugraha · Khaerun Nisa'Tri Safaati · Indira Faisa Afgani

---

## Project Overview

FINTel adalah dashboard prediksi churn pelanggan telekomunikasi menggunakan **Logistic Regression** yang dioptimasi dengan **F2-Score (β=2)**. Setiap prediksi disertai segmentasi risiko (Rendah / Sedang / Tinggi) dan **3 rekomendasi campaign retensi yang dipersonalisasi** berdasarkan Top-3 SHAP features per individu.

---

## Features

### Pelanggan Lama
- Lookup pelanggan berdasarkan Customer ID dari database
- Laporan lengkap: Ringkasan Pelanggan, Informasi Akun, Informasi Pembayaran, Layanan Tambahan (Add-On)
- Churn Probability & segmentasi risiko (Rendah / Sedang / Tinggi)
- Top SHAP features ranked — rank badge merah/kuning/biru untuk top 3
- **3 rekomendasi campaign retensi** dipersonalisasi dari top-3 SHAP per individu

### Pelanggan Baru
- Form input: Informasi Demografi, Layanan, Akun dan Tagihan
- City autocomplete dari database (ketik sebagian nama kota, langsung muncul pilihan)
- Prediksi real-time dengan SHAP individual
- Rekomendasi campaign identik dengan Pelanggan Lama

### Prediksi Secara Keseluruhan (CSV)
- Upload CSV massal
- 4 metric cards: Total, Churn, No Churn, Churn Rate
- 4 visualisasi: Donut Chart, Segment Bar, Probability Histogram, Score Histogram
- Tabel hasil berwarna + Download CSV

---

## Architecture

### Model Pipeline
```
Raw CSV (20 features)
  └── Pipeline:
        ColumnTransformer
          ├── OneHotEncoder (drop='first')  → 16 kolom kategorikal
          └── TargetEncoder                 → City
        → RobustScaler
        → SelectKBest (k=20, ANOVA F-test)
        → LogisticRegression (CLASS_WEIGHT='balanced')
        → Threshold Optimization (F2-Score β=2)
```

### Segmentasi Risiko Churn
Segmentasi berdasarkan `best_threshold` dari model, dibagi 3 sama rata di atas threshold:

| Segmen | Range Probability | Warna |
|--------|-------------------|-------|
| Rendah | prob < threshold | Hijau |
| Sedang | threshold – threshold + ⅓(1–threshold) | Kuning |
| Tinggi | threshold + ⅔(1–threshold) – 1.0 | Merah |

### Recommendation Logic
```
Top-3 SHAP features (by |SHAP value|) per customer
  └── Rank 1 (SHAP terbesar) → Campaign HIGH    → Priority action   (50% budget)
  └── Rank 2                 → Campaign MEDIUM  → Supporting action (35% budget)
  └── Rank 3                 → Campaign LOW     → Additional action (15% budget)
```
Setiap fitur memiliki 3 campaign unik dari `CAMPAIGN_CATALOG`. Rank menentukan tingkat urgensi dan alokasi budget retensi.

---

## Folder Structure

```
fintel-churn-app/
│
├── app.py                   # Main Streamlit app
├── requirements.txt         # Dependencies
├── README.md
├── .gitignore
│
├── models/
│   └── artifacts.pkl        # Trained model + SHAP data + metadata
│
├── utils/
│   ├── __init__.py
│   ├── prediction.py        # Inference, segmentasi, SHAP recs, CAMPAIGN_CATALOG
│   └── visualization.py     # Plotly charts
│
├── data/
│   └── df_clean.csv         # Reference database (dengan customerID & City)
│
└── .streamlit/
    └── config.toml          # Theme config
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/fintel-churn-app.git
cd fintel-churn-app

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "FINTel deploy"
git remote add origin https://github.com/YOUR_USERNAME/fintel-churn-app.git
git push -u origin main

# 2. Buka share.streamlit.io → New App → pilih repo → main file: app.py → Deploy
```

> Pastikan `models/artifacts.pkl` dan `data/df_clean.csv` ikut ter-commit ke repo.

---

## Model Performance

| Metric | Value |
|--------|-------|
| F2 (β=2) | ~0.75 |
| ROC-AUC | ~0.85 |
| Threshold | ~0.37 |

Threshold dioptimasi menggunakan F2-Score (β=2): Recall diprioritaskan 4× di atas Precision, sesuai asumsi bisnis bahwa biaya kehilangan pelanggan jauh lebih tinggi dari biaya intervensi retensi.

---
