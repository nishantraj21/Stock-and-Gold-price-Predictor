
const API = (location.protocol === 'file:' || location.origin === 'null')
  ? 'http://127.0.0.1:5000'
  : location.origin;   // same origin when served by Flask; fallback for local file opens
let teslaResults = null, msftChart = null, teslaChart = null, gold4hChart = null, goldDailyChart = null;
let selectedModel = 'Logistic Regression';
let qeOn = false;
let teslaChartInst = null, msftChartInst = null, aucChartInst = null, gold4hChartInst = null, goldDailyChartInst = null;
let msftChartMode = 'test';

const TESLA_SAMPLES = [
  { open:25.79, close:23.83, low:23.30, high:30.42, qe:1, label:'Jun 2010' },
  { open:24.65, close:24.68, low:24.02, high:24.89, qe:0, label:'Jan 2011' },
  { open:209.00,close:207.29,low:203.50,high:213.61,qe:0, label:'Mar 2014' },
  { open:264.00,close:261.50,low:261.20,high:265.33,qe:0, label:'Mar 2017' },
];
const GOLD_4H_SAMPLES = [
  { high:2089.50, low:2072.10, close:2085.30, label:'Jan 2024' },
  { high:2380.40, low:2340.80, close:2360.15, label:'Apr 2024' },
  { high:2700.50, low:2680.00, close:2695.00, label:'Oct 2024' },
];
const GOLD_DAILY_SAMPLES = [
  { prices:[1820.50,1835.20,1842.10,1855.00,1860.40], label:'Mid 2021' },
  { prices:[1930.20,1918.60,1945.80,1960.10,1955.30], label:'Early 2022' },
  { prices:[2650.00,2620.30,2680.50,2710.80,2695.40], label:'Late 2024' },
];

// Build daily inputs on page load
function buildDailyInputs() {
  const wrap = document.getElementById('gd-inputs');
  if (!wrap) return;
  wrap.innerHTML = [1,2,3,4,5].map(i =>
    \`<div>
      <label style="font-size:.75rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:5px">Day \${i} Close (USD)</label>
      <input id="gd-p\${i}" type="number" step="0.01" value="\${(2000+i*20).toFixed(2)}" style="width:100%;background:var(--bg);border:1px solid var(--border2);border-radius:8px;padding:9px 14px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:.9rem;outline:none"/>
    </div>\`
  ).join('');
}

function loadGold4hSample(i) {
  const s = GOLD_4H_SAMPLES[i];
  document.getElementById('g4h-high').value  = s.high;
  document.getElementById('g4h-low').value   = s.low;
  document.getElementById('g4h-close').value = s.close;
}
function loadGoldDailySample(i) {
  const s = GOLD_DAILY_SAMPLES[i];
  s.prices.forEach((p,j) => { const el=document.getElementById('gd-p'+(j+1)); if(el) el.value=p; });
}

const MSFT_SAMPLES = [
  { p1:238.93, p2:237.13, p3:235.75, label:'Early 2022' },
  { p1:252.10, p2:249.30, p3:247.89, label:'Mid 2022' },
  { p1:430.16, p2:432.53, p3:428.97, label:'Late 2024' },
  { p1:415.06, p2:410.92, p3:414.99, label:'Recent 2025' },
];

// ── page nav ──────────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'compare') renderCompare();
}

// ── API init ──────────────────────────────────────────────────────────────
async function init() {
  try {
    const r = await fetch(API + '/api/health');
    const d = await r.json();
    const el = document.getElementById('api-status');
    el.innerHTML = '<span class="pulse-dot"></span> API online';
    el.style.color = 'var(--green)';
    el.style.borderColor = 'rgba(106,247,168,.3)';
  } catch(e) {
    const el = document.getElementById('api-status');
    el.textContent = '● API offline';
    el.style.color = 'var(--red)';
  }

  // load results + charts in parallel
  try {
    const [rRes, mRes, tRes, g4hRes, gdRes] = await Promise.all([
      fetch(API + '/api/tesla/results').then(r => r.json()),
      fetch(API + '/api/msft/chart').then(r => r.json()),
      fetch(API + '/api/tesla/chart').then(r => r.json()),
      fetch(API + '/api/gold/4h/chart').then(r => r.json()),
      fetch(API + '/api/gold/daily/chart').then(r => r.json()),
    ]);
    teslaResults = rRes; msftChart = mRes; teslaChart = tRes;
    gold4hChart = g4hRes; goldDailyChart = gdRes;
    populateModelAUC();
    renderTeslaChart();
    renderMsftChart('test');
    renderGold4hChart();
    renderGoldDailyChart();
    buildDailyInputs();
  } catch(e) { console.warn('Chart data load failed:', e); }
}

function populateModelAUC() {
  if (!teslaResults) return;
  const map = {
    'Logistic Regression': 'lr',
    'SVM (Poly)': 'svm',
    'XGBoost (GB)': 'xgb',
  };
  for (const [name, key] of Object.entries(map)) {
    const r = teslaResults[name];
    if (r) document.getElementById('auc-' + key).textContent = `Val AUC: ${r.val_auc}`;
  }
}

// ── model select ──────────────────────────────────────────────────────────
function selectModel(name) {
  selectedModel = name;
  document.querySelectorAll('.model-opt').forEach(el => el.classList.remove('active'));
  const map = {'Logistic Regression':'mo-lr','SVM (Poly)':'mo-svm','XGBoost (GB)':'mo-xgb'};
  document.getElementById(map[name]).classList.add('active');
}

function toggleQE() {
  qeOn = !qeOn;
  document.getElementById('qe-toggle').classList.toggle('on', qeOn);
}

// ── quick load ────────────────────────────────────────────────────────────
function loadTeslaSample(i) {
  const s = TESLA_SAMPLES[i];
  document.getElementById('t-open').value  = s.open;
  document.getElementById('t-close').value = s.close;
  document.getElementById('t-low').value   = s.low;
  document.getElementById('t-high').value  = s.high;
  qeOn = s.qe === 1;
  document.getElementById('qe-toggle').classList.toggle('on', qeOn);
}
function loadMsftSample(i) {
  const s = MSFT_SAMPLES[i];
  document.getElementById('m-p1').value = s.p1;
  document.getElementById('m-p2').value = s.p2;
  document.getElementById('m-p3').value = s.p3;
}

// ── Tesla predict ─────────────────────────────────────────────────────────
async function predictTesla() {
  const btn = document.getElementById('tesla-btn');
  const errEl = document.getElementById('tesla-error');
  errEl.style.display = 'none';
  const o = parseFloat(document.getElementById('t-open').value);
  const c = parseFloat(document.getElementById('t-close').value);
  const l = parseFloat(document.getElementById('t-low').value);
  const h = parseFloat(document.getElementById('t-high').value);
  if ([o,c,l,h].some(isNaN)) {
    errEl.textContent = '⚠ Please fill all four price fields.';
    errEl.style.display = 'block'; return;
  }
  btn.disabled = true; btn.classList.add('loading');
  try {
    const res = await fetch(API + '/api/tesla/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ open:o, close:c, low:l, high:h,
                             is_quarter_end: qeOn?1:0, model: selectedModel })
    });
    const d = await res.json();
    if (d.status !== 'success') throw new Error(d.error);
    renderTeslaResult(d);
  } catch(e) {
    errEl.textContent = '⚠ ' + e.message;
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

function renderTeslaResult(d) {
  const isUp = d.direction === 1;
  const cls  = isUp ? 'up' : 'down';
  const rw = document.getElementById('tesla-result-wrap');
  rw.style.display = 'block';
  rw.innerHTML = `
    <div class="result-panel card">
      <div class="result-header ${cls}">
        <div class="result-direction ${cls}">${d.label}</div>
        <div class="result-meta">
          <div class="result-model">${d.model_used}</div>
          <div class="result-prob" style="color:${isUp?'var(--green)':'var(--red)'}">
            ${d.probability}% confidence
          </div>
        </div>
      </div>
      <div class="result-body">
        <div class="result-grid">
          <div class="rstat">
            <div class="rstat-val">${d.features.open_close.toFixed(3)}</div>
            <div class="rstat-key">Open − Close</div>
          </div>
          <div class="rstat">
            <div class="rstat-val">${d.features.low_high.toFixed(3)}</div>
            <div class="rstat-key">Low − High</div>
          </div>
          <div class="rstat">
            <div class="rstat-val">${d.features.is_quarter_end}</div>
            <div class="rstat-key">Quarter End</div>
          </div>
        </div>
      </div>
    </div>`;
}

// ── MSFT predict ──────────────────────────────────────────────────────────
async function predictMsft() {
  const btn = document.getElementById('msft-btn');
  const errEl = document.getElementById('msft-error');
  errEl.style.display = 'none';
  const p1 = parseFloat(document.getElementById('m-p1').value);
  const p2 = parseFloat(document.getElementById('m-p2').value);
  const p3 = parseFloat(document.getElementById('m-p3').value);
  if ([p1,p2,p3].some(isNaN)) {
    errEl.textContent = '⚠ Please enter all three closing prices.';
    errEl.style.display = 'block'; return;
  }
  btn.disabled = true; btn.classList.add('loading');
  try {
    const res = await fetch(API + '/api/msft/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ price1:p1, price2:p2, price3:p3 })
    });
    const d = await res.json();
    if (d.status !== 'success') throw new Error(d.error);
    renderMsftResult(d);
  } catch(e) {
    errEl.textContent = '⚠ ' + e.message;
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

function renderMsftResult(d) {
  const isUp = d.change >= 0;
  const cls  = isUp ? 'up' : 'down';
  const rw = document.getElementById('msft-result-wrap');
  rw.style.display = 'block';
  rw.innerHTML = `
    <div class="result-panel card">
      <div class="result-header ${cls}">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--muted)">PREDICTED NEXT CLOSE</div>
          <div class="result-direction ${cls}" style="font-size:2rem">$${d.predicted_price}</div>
        </div>
        <div class="result-meta" style="text-align:right">
          <div class="result-model">${d.model_used}</div>
          <div class="result-prob" style="color:${isUp?'var(--green)':'var(--red)'}">
            ${d.direction} &nbsp; ${d.change >= 0 ? '+' : ''}${d.change} (${d.change_pct >= 0 ? '+' : ''}${d.change_pct}%)
          </div>
        </div>
      </div>
      <div class="result-body">
        <div class="result-grid">
          <div class="rstat">
            <div class="rstat-val">$${d.input_window[0]}</div>
            <div class="rstat-key">Day T-3</div>
          </div>
          <div class="rstat">
            <div class="rstat-val">$${d.input_window[1]}</div>
            <div class="rstat-key">Day T-2</div>
          </div>
          <div class="rstat">
            <div class="rstat-val">$${d.input_window[2]}</div>
            <div class="rstat-key">Day T-1 (last)</div>
          </div>
        </div>
      </div>
    </div>`;
}

// ── charts ────────────────────────────────────────────────────────────────
const chartDefaults = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#7070a0', font: { family: 'JetBrains Mono', size: 11 }, boxWidth: 12 } }, tooltip: { backgroundColor: '#16161f', borderColor: '#2a2a3e', borderWidth: 1, titleColor: '#e8e8f0', bodyColor: '#7070a0', titleFont: { family: 'JetBrains Mono' }, bodyFont: { family: 'JetBrains Mono', size: 11 } } },
  scales: {
    x: { ticks: { color: '#404060', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,.04)' }, border: { color: '#222232' } },
    y: { ticks: { color: '#7070a0', font: { family: 'JetBrains Mono', size: 11 } }, grid: { color: 'rgba(255,255,255,.04)' }, border: { color: '#222232' } }
  }
};

function renderTeslaChart() {
  if (!teslaChart) return;
  const labels = teslaChart.dates.filter((_,i) => i % 5 === 0);
  const data   = teslaChart.close.filter((_,i) => i % 5 === 0);
  if (teslaChartInst) teslaChartInst.destroy();
  teslaChartInst = new Chart(document.getElementById('tesla-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'TSLA Close',
        data,
        borderColor: '#7c6af7',
        backgroundColor: 'rgba(124,106,247,.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }]
    },
    options: { ...chartDefaults }
  });
}

function switchMsftChart(mode) {
  msftChartMode = mode;
  document.querySelectorAll('.inner-tab').forEach((t,i) => {
    t.classList.toggle('active', (i===0 && mode==='test') || (i===1 && mode==='val'));
  });
  renderMsftChart(mode);
}

function renderMsftChart(mode) {
  if (!msftChart) return;
  const prefix = mode === 'test' ? 'test' : 'val';
  const labels = msftChart[prefix+'_dates'];
  const actual = msftChart[prefix+'_actual'];
  const pred   = msftChart[prefix+'_pred'];
  const thin = labels.length > 100 ? labels.map((_,i) => i%3===0 ? _ : '') : labels;
  if (msftChartInst) msftChartInst.destroy();
  msftChartInst = new Chart(document.getElementById('msft-chart'), {
    type: 'line',
    data: {
      labels: thin,
      datasets: [
        { label: 'Actual', data: actual, borderColor: '#4fc3f7', borderWidth: 1.8, pointRadius: 0, tension: 0.2 },
        { label: 'Predicted', data: pred, borderColor: '#f7c76a', borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, tension: 0.2 },
      ]
    },
    options: { ...chartDefaults }
  });
  // metrics below chart
  const mae = msftChart[prefix === 'test' ? 'test_mae' : 'val_mae'];
  const mape = msftChart.test_mape;
  const mcEl = document.getElementById('msft-chart-metrics');
  mcEl.innerHTML = `
    <div class="rstat" style="flex:1"><div class="rstat-val">$${mae}</div><div class="rstat-key">${mode === 'test'?'Test':'Val'} MAE</div></div>
    ${mode==='test' ? `<div class="rstat" style="flex:1"><div class="rstat-val">${mape}%</div><div class="rstat-key">Test MAPE</div></div>` : ''}
  `;
}

// ── compare page ──────────────────────────────────────────────────────────
function renderCompare() {
  if (!teslaResults) return;

  // table
  const tbody = document.getElementById('compare-tbody');
  const entries = Object.entries(teslaResults).sort((a,b) => b[1].val_auc - a[1].val_auc);
  tbody.innerHTML = entries.map(([name, r], i) => {
    const isBest = i === 0;
    const genap = r.val_auc >= r.train_auc ? '✓ No overfit' : `−${(r.train_auc - r.val_auc).toFixed(4)} gap`;
    return `<tr>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--muted)">${i+1}</td>
      <td><strong>${name}</strong>${isBest ? '<span class="badge-best">BEST</span>' : ''}</td>
      <td style="color:var(--muted);font-size:.8rem">${name === 'XGBoost (GB)' ? 'Gradient Boosting (100 trees)' : name === 'SVM (Poly)' ? 'SVC · polynomial kernel · Platt' : 'Logistic (L2 · lbfgs solver)'}</td>
      <td>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.85rem">${r.train_auc}</span>
        <div class="auc-bar"><div class="auc-fill" style="width:${r.train_auc*100}%"></div></div>
      </td>
      <td>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:${isBest?'var(--green)':'var(--text)'}">${r.val_auc}</span>
        <div class="auc-bar"><div class="auc-fill" style="width:${r.val_auc*100}%"></div></div>
      </td>
      <td style="font-size:.8rem;color:var(--muted)">${genap}</td>
    </tr>`;
  }).join('');

  // bar chart
  const names = entries.map(([n]) => n.replace(' (Poly)','').replace(' (GB)',''));
  const trainA = entries.map(([,r]) => r.train_auc);
  const valA   = entries.map(([,r]) => r.val_auc);
  if (aucChartInst) aucChartInst.destroy();
  aucChartInst = new Chart(document.getElementById('auc-bar-chart'), {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        { label: 'Train AUC', data: trainA, backgroundColor: 'rgba(124,106,247,.6)', borderRadius: 5 },
        { label: 'Val AUC',   data: valA,   backgroundColor: 'rgba(79,195,247,.6)',  borderRadius: 5 },
      ]
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, min: 0.4, max: 1.0 }
      }
    }
  });

  // msft metrics
  if (msftChart) {
    document.getElementById('msft-metrics-grid').innerHTML = `
      <div class="rstat"><div class="rstat-val">$${msftChart.val_mae}</div><div class="rstat-key">Val MAE (USD)</div></div>
      <div class="rstat"><div class="rstat-val">$${msftChart.test_mae}</div><div class="rstat-key">Test MAE (USD)</div></div>
      <div class="rstat"><div class="rstat-val">${msftChart.test_mape ?? '—'}%</div><div class="rstat-key">Test MAPE</div></div>
      <div class="rstat"><div class="rstat-val">n=3</div><div class="rstat-key">Window Size</div></div>
    `;
  }
}

// ── boot ──────────────────────────────────────────────────────────────────
init();
