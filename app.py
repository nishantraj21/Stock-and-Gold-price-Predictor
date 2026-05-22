import os, json, pickle, warnings
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='')

BASE = os.path.dirname(__file__)

# ── Load Tesla models ─────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'tesla_models.pkl'), 'rb') as f:
    tesla_bundle  = pickle.load(f)
TESLA_MODELS  = tesla_bundle['models']
TESLA_SCALER  = tesla_bundle['scaler']
TESLA_RESULTS = tesla_bundle['results']

# ── Load MSFT model ───────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'msft_model.pkl'), 'rb') as f:
    msft_bundle   = pickle.load(f)
MSFT_MODEL      = msft_bundle['model']
MSFT_LAST3      = msft_bundle['last_3_prices']
MSFT_LAST_CLOSE = msft_bundle['last_close']

# ── Load Gold 4H model (Random Forest Regressor) ──────────────────────────────
with open(os.path.join(BASE, 'gold_4h_model.pkl'), 'rb') as f:
    gold_4h_bundle    = pickle.load(f)
GOLD_4H_MODEL     = gold_4h_bundle['model']
GOLD_4H_LAST      = gold_4h_bundle['last_known']
GOLD_4H_R2        = gold_4h_bundle['r2']
GOLD_4H_MAE       = gold_4h_bundle['mae']

# ── Load Gold Daily model (MLP Windowed Regressor) ────────────────────────────
with open(os.path.join(BASE, 'gold_daily_model.pkl'), 'rb') as f:
    gold_daily_bundle    = pickle.load(f)
GOLD_DAILY_MODEL     = gold_daily_bundle['model']
GOLD_DAILY_LAST5     = gold_daily_bundle['last_5_closes']
GOLD_DAILY_LAST_CLOSE = gold_daily_bundle['last_close']

# ── Load chart JSONs ─────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'msft_chart.json'))       as f: MSFT_CHART       = json.load(f)
with open(os.path.join(BASE, 'tesla_chart.json'))      as f: TESLA_CHART      = json.load(f)
with open(os.path.join(BASE, 'gold_4h_chart.json'))    as f: GOLD_4H_CHART    = json.load(f)
with open(os.path.join(BASE, 'gold_daily_chart.json')) as f: GOLD_DAILY_CHART = json.load(f)

print("✅ All models loaded (Tesla · MSFT · Gold 4H · Gold Daily)")

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp

# ── Health ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(TESLA_MODELS.keys()) + ['MSFT MLP Regressor', 'Gold 4H RF Regressor', 'Gold Daily MLP Regressor']
    })

# ── Tesla ─────────────────────────────────────────────────────────────────────
@app.route('/api/tesla/results')
def tesla_results():  return jsonify(TESLA_RESULTS)

@app.route('/api/tesla/chart')
def tesla_chart():    return jsonify(TESLA_CHART)

@app.route('/api/tesla/predict', methods=['POST','OPTIONS'])
def tesla_predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        d = request.get_json()
        open_close = float(d['open'])  - float(d['close'])
        low_high   = float(d['low'])   - float(d['high'])
        is_qe      = int(d.get('is_quarter_end', 0))
        x = TESLA_SCALER.transform([[open_close, low_high, is_qe]])
        model_name = d.get('model', 'XGBoost (GB)')
        if model_name not in TESLA_MODELS: model_name = 'XGBoost (GB)'
        model     = TESLA_MODELS[model_name]
        direction = int(model.predict(x)[0])
        proba     = float(model.predict_proba(x)[0][1])
        return jsonify({'direction': direction, 'label': 'UP ↑' if direction==1 else 'DOWN ↓',
                        'probability': round(proba*100,1), 'model_used': model_name,
                        'features': {'open_close': round(open_close,4), 'low_high': round(low_high,4), 'is_quarter_end': is_qe},
                        'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# ── MSFT ──────────────────────────────────────────────────────────────────────
@app.route('/api/msft/chart')
def msft_chart():     return jsonify(MSFT_CHART)

@app.route('/api/msft/predict', methods=['POST','OPTIONS'])
def msft_predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        d = request.get_json()
        if 'prices' in d:
            prices = [float(p) for p in d['prices']]
            if len(prices) != 3: return jsonify({'error': 'Need exactly 3 prices', 'status': 'error'}), 400
        else:
            prices = [float(d.get('price1', MSFT_LAST3[0])),
                      float(d.get('price2', MSFT_LAST3[1])),
                      float(d.get('price3', MSFT_LAST3[2]))]
        w      = np.array(prices, dtype=float)
        x_norm = (w / w[0]).reshape(1, -1)
        ratio  = MSFT_MODEL.predict(x_norm)[0]
        predicted   = round(float(w[0] * ratio), 2)
        change      = round(predicted - prices[-1], 2)
        change_pct  = round((predicted / prices[-1] - 1) * 100, 2)
        return jsonify({'predicted_price': predicted, 'last_price': round(prices[-1],2),
                        'change': change, 'change_pct': change_pct,
                        'direction': 'UP ↑' if change > 0 else 'DOWN ↓',
                        'model_used': 'LSTM-equivalent (MLP Regressor)',
                        'input_window': [round(p,2) for p in prices], 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# ── Gold 4H (Random Forest Regressor) ────────────────────────────────────────
@app.route('/api/gold/4h/chart')
def gold_4h_chart():  return jsonify(GOLD_4H_CHART)

@app.route('/api/gold/4h/predict', methods=['POST','OPTIONS'])
def gold_4h_predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        d    = request.get_json()
        high  = float(d.get('high',  GOLD_4H_LAST['High']))
        low   = float(d.get('low',   GOLD_4H_LAST['Low']))
        close = float(d.get('close', GOLD_4H_LAST['Close']))
        x = np.array([[high, low, close]])
        predicted = float(GOLD_4H_MODEL.predict(x)[0])
        change    = round(predicted - close, 2)
        change_pct = round((predicted / close - 1) * 100, 3)
        return jsonify({'predicted_open': round(predicted, 2),
                        'last_close': round(close, 2),
                        'change': change, 'change_pct': change_pct,
                        'direction': 'UP ↑' if change > 0 else 'DOWN ↓',
                        'model_used': 'Random Forest Regressor',
                        'model_r2': GOLD_4H_R2, 'model_mae': GOLD_4H_MAE,
                        'input': {'high': high, 'low': low, 'close': close},
                        'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# ── Gold Daily (MLP Windowed Regressor) ──────────────────────────────────────
@app.route('/api/gold/daily/chart')
def gold_daily_chart():  return jsonify(GOLD_DAILY_CHART)

@app.route('/api/gold/daily/predict', methods=['POST','OPTIONS'])
def gold_daily_predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        d = request.get_json()
        if 'prices' in d:
            prices = [float(p) for p in d['prices']]
            if len(prices) != 5: return jsonify({'error': 'Need exactly 5 daily closing prices', 'status': 'error'}), 400
        else:
            prices = [float(d.get(f'price{i+1}', GOLD_DAILY_LAST5[i])) for i in range(5)]
        w      = np.array(prices, dtype=float)
        x_norm = (w / w[0]).reshape(1, -1)
        ratio  = GOLD_DAILY_MODEL.predict(x_norm)[0]
        predicted   = round(float(w[0] * ratio), 2)
        change      = round(predicted - prices[-1], 2)
        change_pct  = round((predicted / prices[-1] - 1) * 100, 3)
        return jsonify({'predicted_price': predicted,
                        'last_price': round(prices[-1], 2),
                        'change': change, 'change_pct': change_pct,
                        'direction': 'UP ↑' if change > 0 else 'DOWN ↓',
                        'model_used': 'MLP Regressor (windowed n=5)',
                        'input_window': [round(p,2) for p in prices],
                        'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("🚀 StockSense API → http://localhost:5000")
    app.run(debug=True, port=5000)
