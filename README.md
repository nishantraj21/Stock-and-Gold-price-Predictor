# StockSense — ML Stock Price Predictor
### B.Tech CSE Project · RRSDCE Begusarai · 2022–2026

A full-stack machine learning web application that predicts:
- **Tesla (TSLA)** next-day price **direction** (Up/Down) using Logistic Regression, SVM, and Gradient Boosting
- **Microsoft (MSFT)** next-day **close price** using a windowed MLP Regressor (LSTM-equivalent)

---

## 📁 Project Structure

```
stock-price-predictor/
├── app.py                  ← Flask REST API server
├── static/
│   └── index.html          ← Full frontend (single-file SPA)
├── tesla_models.pkl        ← Trained classifiers + StandardScaler
├── msft_model.pkl          ← Trained MLP Regressor
├── tesla_results.json      ← ROC-AUC scores for all models
├── tesla_chart.json        ← Tesla price history for chart
├── msft_chart.json         ← MSFT val/test predictions vs actual
├── requirements.txt        ← Python dependencies
├── train_models.py         ← Re-train models from raw CSVs
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train models from raw data
Place `Tesla.csv` and `MSFT_1986-03-13_2025-02-04.csv` in the project folder, then:
```bash
python train_models.py
```

### 3. Start the server
```bash
python app.py
```

### 4. Open the app
Navigate to **http://localhost:5000** in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/tesla/predict` | Direction prediction (XGBoost/SVM/LR) |
| POST | `/api/msft/predict` | Close price regression |
| GET | `/api/tesla/results` | All model ROC-AUC scores |
| GET | `/api/tesla/chart` | Tesla historical price series |
| GET | `/api/msft/chart` | MSFT val+test prediction data |

### Tesla Predict — Request Body
```json
{
  "open": 264.00,
  "close": 261.50,
  "low": 259.06,
  "high": 265.33,
  "is_quarter_end": 0,
  "model": "XGBoost (GB)"
}
```

### MSFT Predict — Request Body
```json
{
  "price1": 295.40,
  "price2": 294.80,
  "price3": 296.20
}
```

---

## 📊 Models & Performance

### Tesla Classification (1,691 training days · 2010–2017)
| Model | Train AUC | Val AUC |
|-------|-----------|---------|
| Gradient Boosting (XGBoost) | 0.8217 | 0.5928 |
| Logistic Regression | 0.5192 | 0.5437 |
| SVM (Polynomial) | 0.4719 | 0.4465 |

### MSFT Regression (9,800 days · 1986–2025 · windowed 2021–2022)
| Metric | Value |
|--------|-------|
| Val MAE | ~$1.75 |
| Test MAE | ~$3.85 |
| Test MAPE | ~1.22% |

---

## 🏗 Architecture

```
Browser (index.html)
    ↕ REST JSON
Flask API (app.py:5000)
    ↕ pickle.load()
scikit-learn Models (.pkl)
    ↑ trained from
Raw CSV Datasets
```

---

## 👥 Team
| Name | Reg. No. |
|------|----------|
| Nishant Raj | 22105125002 |
| Samar Sudhanshu | 22105125052 |
| Prince Kumar | 22105125019 |
| Anshu Kumar | 22105125047 |
| Ankit Kumar | 22105125039 |

**Supervisor:** Prof. Ravi Kumar, Asst. Prof. CSE  
**Institution:** Rashtrakavi Ramdhari Singh Dinkar College of Engineering, Begusarai  
**University:** Bihar Engineering University
