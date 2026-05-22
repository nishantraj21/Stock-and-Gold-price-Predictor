"""
train_models.py
───────────────
Re-train all models from raw CSV files and save pickles + JSON chart data.
Run this script if you want to update the models with new data.

Usage:
    python train_models.py
    
Required files in same directory:
    Tesla.csv
    MSFT_1986-03-13_2025-02-04.csv  (or any MSFT CSV with Date + Close columns)
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — Tesla Direction Classification
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TRAINING TESLA CLASSIFICATION MODELS")
print("=" * 60)

tesla_path = os.path.join(BASE, 'Tesla.csv')
df = pd.read_csv(tesla_path)
print(f"Loaded {len(df)} rows from Tesla.csv")

# Drop Adj Close (identical to Close for this dataset)
df = df.drop(['Adj Close'], axis=1)

# Date parsing
splitted = df['Date'].str.split('/', expand=True)
df['day']            = splitted[1].astype('int')
df['month']          = splitted[0].astype('int')
df['year']           = splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Feature engineering
df['open-close'] = df['Open']  - df['Close']
df['low-high']   = df['Low']   - df['High']
df['target']     = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

print(f"Class distribution: {df['target'].value_counts().to_dict()}")

# Features + scale
features = df[['open-close', 'low-high', 'is_quarter_end']]
target   = df['target']
scaler   = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features_scaled, target, test_size=0.1, random_state=2022)
print(f"Train: {X_train.shape[0]}  Valid: {X_valid.shape[0]}")

# Train all 3 models
model_configs = [
    ('Logistic Regression', LogisticRegression()),
    ('SVM (Poly)',          SVC(kernel='poly', probability=True)),
    ('XGBoost (GB)',        GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

models_dict, results = {}, {}
for name, model in model_configs:
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    t_auc = float(metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1]))
    v_auc = float(metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:,1]))
    results[name] = {'train_auc': round(t_auc, 4), 'val_auc': round(v_auc, 4)}
    models_dict[name] = model
    print(f"  Train AUC: {t_auc:.4f}   Val AUC: {v_auc:.4f}")

# Save tesla artifacts
with open(os.path.join(BASE, 'tesla_models.pkl'), 'wb') as f:
    pickle.dump({'models': models_dict, 'scaler': scaler, 'results': results}, f)

with open(os.path.join(BASE, 'tesla_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

tesla_chart = {
    'dates':  df['Date'].tolist(),
    'close':  [round(float(v), 2) for v in df['Close'].tolist()],
    'target': df['target'].tolist(),
}
with open(os.path.join(BASE, 'tesla_chart.json'), 'w') as f:
    json.dump(tesla_chart, f)

print("\n✅ Tesla models saved: tesla_models.pkl, tesla_results.json, tesla_chart.json")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — Microsoft LSTM-Equivalent (MLP Windowed Regressor)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING MSFT REGRESSION MODEL (LSTM-equivalent)")
print("=" * 60)

msft_path = os.path.join(BASE, 'MSFT_1986-03-13_2025-02-04.csv')
df2 = pd.read_csv(msft_path)
df2 = df2[['Date', 'Close']].copy()
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index('Date').sort_index()
closes     = df2['Close'].values.astype(np.float64)
dates_all  = list(df2.index)
print(f"Loaded {len(closes)} rows from MSFT CSV  ({dates_all[0].date()} – {dates_all[-1].date()})")

# Build windowed sequences (n=3, same as notebook)
n = 3
X_raw, Y_raw, dates_seq = [], [], []
for i in range(n, len(closes)):
    X_raw.append(closes[i-n:i])
    Y_raw.append(closes[i])
    dates_seq.append(dates_all[i])

X_raw = np.array(X_raw)
Y_raw = np.array(Y_raw)
print(f"Windows created: {len(X_raw)}")

# Per-window ratio normalisation (same as notebook's approach)
X_norm = X_raw / X_raw[:, 0:1]   # divide each window by its first price
Y_norm = Y_raw / X_raw[:, 0]     # target as ratio of window start

sp80 = int(len(X_norm) * 0.8)
sp90 = int(len(X_norm) * 0.9)
print(f"Split — Train: {sp80}  Val: {sp90-sp80}  Test: {len(X_norm)-sp90}")

# Train MLP (64-32-32 architecture matching notebook's LSTM depth)
print("\nTraining MLP Regressor (LSTM-equivalent)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.05,
    n_iter_no_change=20,
)
mlp.fit(X_norm[:sp80], Y_norm[:sp80])
print(f"  Training converged in {mlp.n_iter_} iterations")

def predict_price(model, window_prices):
    w  = np.array(window_prices, dtype=float)
    xn = (w / w[0]).reshape(1, -1)
    return w[0] * model.predict(xn)[0]

val_pred  = [predict_price(mlp, X_raw[i]) for i in range(sp80, sp90)]
val_act   = Y_raw[sp80:sp90]
test_pred = [predict_price(mlp, X_raw[i]) for i in range(sp90, len(X_raw))]
test_act  = Y_raw[sp90:]

mae_val   = np.mean(np.abs(np.array(val_pred)  - val_act))
mae_test  = np.mean(np.abs(np.array(test_pred) - test_act))
mape_test = np.mean(np.abs((np.array(test_pred) - test_act) / test_act)) * 100
print(f"  Val  MAE : ${mae_val:.2f}")
print(f"  Test MAE : ${mae_test:.2f}")
print(f"  Test MAPE: {mape_test:.2f}%")

# Save MSFT artifacts
with open(os.path.join(BASE, 'msft_model.pkl'), 'wb') as f:
    pickle.dump({
        'model':         mlp,
        'last_3_prices': closes[-3:].tolist(),
        'last_close':    float(closes[-1]),
    }, f)

def js(v):
    return [round(float(x), 2) for x in v]

msft_chart = {
    'val_dates':   [str(d)[:10] for d in dates_seq[sp80:sp90]],
    'val_actual':  js(val_act),   'val_pred':   js(val_pred),
    'test_dates':  [str(d)[:10] for d in dates_seq[sp90:]],
    'test_actual': js(test_act),  'test_pred':  js(test_pred),
    'val_mae':     round(float(mae_val),  2),
    'test_mae':    round(float(mae_test), 2),
    'test_mape':   round(float(mape_test), 2),
}
with open(os.path.join(BASE, 'msft_chart.json'), 'w') as f:
    json.dump(msft_chart, f)

print("\n✅ MSFT model saved: msft_model.pkl, msft_chart.json")
print("\n" + "=" * 60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("Run:  python app.py   →   open http://localhost:5000")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# PART 3 — Gold 4H (Random Forest Regressor)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING GOLD 4H RANDOM FOREST MODEL")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor

gold_4h_path = os.path.join(BASE, 'XAU_4h_data.csv')
df_4h = pd.read_csv(gold_4h_path)
print(f"Loaded {len(df_4h)} rows from XAU_4h_data.csv")

X_4h = df_4h.drop(['Date', 'Open'], axis=1)
Y_4h = df_4h['Open']

from sklearn.model_selection import train_test_split
X_tr4h, X_te4h, Y_tr4h, Y_te4h = train_test_split(X_4h, Y_4h, test_size=0.2, random_state=42)
rf_4h = RandomForestRegressor(n_estimators=100, random_state=42)
rf_4h.fit(X_tr4h, Y_tr4h)
pred_4h = rf_4h.predict(X_te4h)
r2_4h  = float(metrics.r2_score(Y_te4h, pred_4h))
mae_4h = float(metrics.mean_absolute_error(Y_te4h, pred_4h))
print(f"  R² Score: {r2_4h:.4f}   MAE: ${mae_4h:.2f}")

last_known = {'High': float(df_4h['High'].iloc[-1]), 'Low': float(df_4h['Low'].iloc[-1]), 'Close': float(df_4h['Close'].iloc[-1])}
with open(os.path.join(BASE, 'gold_4h_model.pkl'), 'wb') as f:
    pickle.dump({'model': rf_4h, 'last_known': last_known, 'r2': round(r2_4h,4), 'mae': round(mae_4h,2)}, f)

gold_4h_chart = {
    'actual':    [round(float(v),2) for v in list(Y_te4h)[-200:]],
    'predicted': [round(float(v),2) for v in list(pred_4h)[-200:]],
    'r2':  round(r2_4h, 4), 'mae': round(mae_4h, 2),
    'last_known': last_known,
}
with open(os.path.join(BASE, 'gold_4h_chart.json'), 'w') as f:
    json.dump(gold_4h_chart, f)

print("✅ Gold 4H model saved: gold_4h_model.pkl, gold_4h_chart.json")

# ──────────────────────────────────────────────────────────────────────────────
# PART 4 — Gold Daily (MLP Windowed Regressor)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING GOLD DAILY MLP MODEL")
print("=" * 60)

from sklearn.neural_network import MLPRegressor

gold_daily_path = os.path.join(BASE, 'Gold1985_25.csv')
df_gd = pd.read_csv(gold_daily_path)
df_gd['DATE'] = pd.to_datetime(df_gd['DATE'], dayfirst=True)
df_gd = df_gd.sort_values('DATE').reset_index(drop=True)
closes_gd = df_gd['GOLD_PRICE'].values.astype(np.float64)
dates_gd  = df_gd['DATE'].tolist()
print(f"Loaded {len(closes_gd)} rows from Gold1985_25.csv")

n = 5
Xg, Yg, dg = [], [], []
for i in range(n, len(closes_gd)):
    Xg.append(closes_gd[i-n:i]); Yg.append(closes_gd[i]); dg.append(dates_gd[i])
Xg, Yg = np.array(Xg), np.array(Yg)
Xn = Xg / Xg[:, 0:1]; Yn = Yg / Xg[:, 0]
sp80 = int(len(Xn)*.8); sp90 = int(len(Xn)*.9)

mlp_gd = MLPRegressor(hidden_layer_sizes=(64,32,32), activation='relu', max_iter=500,
                      random_state=42, learning_rate_init=0.001, early_stopping=True,
                      validation_fraction=0.05, n_iter_no_change=20)
mlp_gd.fit(Xn[:sp80], Yn[:sp80])

def pred_gd(m, win):
    w = np.array(win, dtype=float); return w[0] * m.predict((w/w[0]).reshape(1,-1))[0]

tp = [pred_gd(mlp_gd, Xg[i]) for i in range(sp90, len(Xg))]
ta = Yg[sp90:]
mae_gd  = float(np.mean(np.abs(np.array(tp)-ta)))
mape_gd = float(np.mean(np.abs((np.array(tp)-ta)/ta))*100)
r2_gd   = float(metrics.r2_score(ta, tp))
print(f"  R²: {r2_gd:.4f}   MAE: ${mae_gd:.2f}   MAPE: {mape_gd:.2f}%")

with open(os.path.join(BASE, 'gold_daily_model.pkl'), 'wb') as f:
    pickle.dump({'model': mlp_gd, 'last_5_closes': closes_gd[-5:].tolist(), 'last_close': float(closes_gd[-1])}, f)

gold_daily_chart = {
    'test_dates':  [str(d)[:10] for d in dg[sp90:]],
    'test_actual': [round(float(x),2) for x in ta],
    'test_pred':   [round(float(x),2) for x in tp],
    'mae': round(mae_gd,2), 'mape': round(mape_gd,2), 'r2': round(r2_gd,4),
    'last_5_closes': closes_gd[-5:].tolist(), 'last_close': float(closes_gd[-1]),
}
with open(os.path.join(BASE, 'gold_daily_chart.json'), 'w') as f:
    json.dump(gold_daily_chart, f)

print("✅ Gold Daily model saved: gold_daily_model.pkl, gold_daily_chart.json")
print("\n" + "=" * 60)
print("ALL 4 MODELS TRAINED SUCCESSFULLY")
print("Run:  python app.py   →   open http://localhost:5000")
print("=" * 60)
