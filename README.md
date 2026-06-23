# 🔷 FINTel — Customer Churn Intelligence Dashboard

> ML-powered churn prediction, Low/Mid/High segmentation & SHAP-based personalised retention recommendations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

**Capstone Project** — Purwadhika Digital Technology School · Data Science, ML & AI Bootcamp  
**Tim:** Akbar Kanugraha · Khaerun Nisa'Tri Safaati · Indira Faisa Afgani

---

## 📋 Project Overview

FINTel Churn Intelligence Dashboard memprediksi kemungkinan churn pelanggan telekomunikasi menggunakan **Logistic Regression** yang dioptimasi dengan **F2-Score (β=2)**. Hasil prediksi disajikan dalam tiga segmen churn (Low/Mid/High) beserta rekomendasi retensi yang dipersonalisasi berdasarkan **SHAP per individu**.

---

## ✨ Features

### 🔍 Existing Customer
- Lookup pelanggan by Customer ID dari database
- Laporan lengkap: demografi, layanan, billing, CLTV
- Churn Score (0–100), Churn Segment (Low/Mid/High)
- Top SHAP features ranked (rank badge merah/kuning/biru untuk top 3)
- **Rekomendasi retensi 3 tier** — otomatis sesuai top-3 SHAP per individu

### ➕ New Customer
- Form input 20 fitur lengkap
- Prediksi real-time dengan SHAP individual
- Rekomendasi tiered identik dengan Existing Customer

### 📂 Bulk Prediction
- Upload CSV massal
- 4 metric cards (Total, Churn, No Churn, Churn Rate)
- 4 visualisasi: Donut, Segment Bar, Probability Histogram, Score Histogram
- Tabel hasil berwarna + Download CSV

---

## 🏗️ Architecture

### Model
```
Raw CSV (20 features)
  └── ImbPipeline:
        ColumnTransformer
          ├── OneHotEncoder (drop='first')   → binary categorical
          └── BinaryEncoder                  → multi-category
        → RobustScaler
        → SelectKBest (k=20, ANOVA F-test)
        → SMOTE (resampling)
        → LogisticRegression (C=1, saga, balanced)
        → Threshold (F2-Score β=2 optimized)
        → Piecewise Scaler → ChurnScore 0–100
```

### Churn Segmentation
| Segment | Churn Score | Warna |
|---------|-------------|-------|
| 🟢 Low  | 0 – 33      | Hijau |
| 🟡 Mid  | 34 – 66     | Kuning |
| 🔴 High | 67 – 100    | Merah |

### Recommendation Logic
```
Top-3 SHAP features (by |SHAP value|) per customer
  └── Rank 1 → FEATURE_REC_MAP[feature]["1"]  →  Priority action   🔴
  └── Rank 2 → FEATURE_REC_MAP[feature]["2"]  →  Supporting action 🟡
  └── Rank 3 → FEATURE_REC_MAP[feature]["3"]  →  Additional action 🔵
```
Setiap fitur memiliki 3 rekomendasi unik. Rank menentukan level urgency.

---

## 📁 Folder Structure

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
│   ├── prediction.py        # Inference, segmentation, SHAP recs
│   └── visualization.py     # Plotly charts
│
├── data/
│   └── df_clean.csv         # Reference database (with CustomerID)
│
└── .streamlit/
    └── config.toml          # Theme config
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/fintel-churn-app.git
cd fintel-churn-app

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "FINTel deploy"
git remote add origin https://github.com/YOUR_USERNAME/fintel-churn-app.git
git push -u origin main

# 2. go to share.streamlit.io → New App → select repo → main file: app.py → Deploy
```

> ⚠️ Pastikan `models/artifacts.pkl` dan `data/df_clean.csv` ikut ter-commit ke repo.

---

## 📊 Model Performance

| Metric    | Value  |
|-----------|--------|
| Recall    | ~0.904 |
| Precision | ~0.572 |
| F2 (β=2) | ~0.756 |
| ROC-AUC   | ~0.847 |

Threshold dioptimasi F2-Score: Recall diprioritaskan 4× di atas Precision.

---

## 🎨 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Background | `#EBECEF` | App background |
| Navy | `#0F1D3D` | Text, topbar |
| Primary | `#1A3462` | Sidebar, buttons |
| Secondary | `#476996` | Accents |
| Soft | `#9AADC2` | Muted elements |

---

*Built with ❤️ — Purwadhika 2024*
