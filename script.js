/* ===== DOM References ===== */
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const kSlider = document.getElementById('kSlider');
const kValue = document.getElementById('kValue');
const randomBtn = document.getElementById('randomBtn');
const clearQueryBtn = document.getElementById('clearQueryBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const pointCount = document.getElementById('pointCount');
const prediction = document.getElementById('prediction');
const warnings = document.getElementById('warnings');
const variablesContainer = document.getElementById('variablesContainer');
const neighborMatrixContainer = document.getElementById('neighborMatrixContainer');
const confusionMatrixContainer = document.getElementById('confusionMatrixContainer');
const tooltip = document.getElementById('tooltip');
const themeToggle = document.getElementById('themeToggle');
const metricSelect = document.getElementById('metricSelect');
const minkowskiP = document.getElementById('minkowskiP');
const boundaryToggle = document.getElementById('boundaryToggle');
const explainBtn = document.getElementById('explainBtn');
const explainBox = document.getElementById('explainBox');
const modeClassify = document.getElementById('modeClassify');
const modeAddA = document.getElementById('modeAddA');
const modeAddB = document.getElementById('modeAddB');
const leftPanelFooter = document.getElementById('leftPanelFooter');
const rightPanelFooter = document.getElementById('rightPanelFooter');
const patternSelect = document.getElementById('patternSelect');
const accChart = document.getElementById('accChart');
const accCtx = accChart.getContext('2d');
const biasBadge = document.getElementById('biasBadge');
const downloadBtn = document.getElementById('downloadBtn');

/* ===== Constants ===== */
const W = canvas.width;
const H = canvas.height;
const PAD = 28;
const SPARSE_THRESHOLD = Math.hypot(W - 2 * PAD, H - 2 * PAD) * 0.25;
const IMBALANCE_RATIO = 0.65;
const POINT_RADIUS = 6;
const HOVER_RADIUS = 10;
const DRAG_THRESHOLD = 16;
const BOUNDARY_STEP = 6;

/* ===== State ===== */
let classA = [];
let classB = [];
let queryPoint = null;
let isDragging = false;
let hoveredPoint = null;
let addMode = 'classify';
let metric = 'euclidean';
let minkowskiPVal = 3;
let showBoundary = false;
let boundaryCache = null;
let accCache = null;
let explainStep = 0;
let animAlpha = null;
let animRAF = null;
const EXPLAIN_STEPS = [
  null,
  { title: 'Step 1: Compute Distances', text: 'Calculate the distance from the query point to every training point using the chosen distance metric.' },
  { title: 'Step 2: Sort by Distance', text: 'Sort all training points by their distance to the query, closest first.' },
  { title: 'Step 3: Select K Nearest', text: 'Pick the K points with the smallest distances. These are the "K nearest neighbors".' },
  { title: 'Step 4: Majority Vote', text: 'Count which class appears most among the K neighbors. That class is the prediction.' },
];

/* ===== Hardcoded theme colors for canvas ===== */
const COLORS = {
  dark: {
    classA: '#ff6b6b', classABorder: '#e05555',
    classAGlow: 'rgba(255, 107, 107, 0.35)',
    classB: '#4fc3f7', classBBorder: '#3aa8d8',
    classBGlow: 'rgba(79, 195, 247, 0.35)',
    query: '#69f0ae', queryGlow: 'rgba(105, 240, 174, 0.5)',
    textMuted: '#64748b',
    success: '#2dd4a0', danger: '#f87171',
  },
  light: {
    classA: '#e05252', classABorder: '#c04040',
    classAGlow: 'rgba(224, 82, 82, 0.2)',
    classB: '#3498db', classBBorder: '#2980b9',
    classBGlow: 'rgba(52, 152, 219, 0.2)',
    query: '#27ae60', queryGlow: 'rgba(39, 174, 96, 0.3)',
    textMuted: '#8892b0',
    success: '#2db488', danger: '#e05252',
  },
};
let currentTheme = 'dark';

/* ===== Distance Functions ===== */
function dist(p1, p2) {
  const dx = p1.x - p2.x, dy = p1.y - p2.y;
  if (metric === 'euclidean') return Math.hypot(dx, dy);
  if (metric === 'manhattan') return Math.abs(dx) + Math.abs(dy);
  if (metric === 'minkowski') {
    const p = minkowskiPVal;
    return Math.pow(Math.pow(Math.abs(dx), p) + Math.pow(Math.abs(dy), p), 1 / p);
  }
  return Math.hypot(dx, dy);
}

function rand(min, max) { return Math.random() * (max - min) + min; }

/* ===== Data Generation ===== */
function generateRandomData(countA, countB) {
  classA = []; classB = [];
  for (let i = 0; i < countA; i++) classA.push({ x: rand(PAD, W - PAD), y: rand(PAD, H - PAD) });
  for (let i = 0; i < countB; i++) classB.push({ x: rand(PAD, W - PAD), y: rand(PAD, H - PAD) });
  boundaryCache = null;
}

/* ===== Pattern Generation ===== */
function noise() { return rand(-12, 12); }

function generateLinear(countA, countB) {
  classA = []; classB = [];
  const mid = (W - PAD * 2) / 2 + PAD;
  for (let i = 0; i < countA; i++) classA.push({ x: rand(PAD, mid - 15) + noise(), y: rand(PAD, H - PAD) + noise() });
  for (let i = 0; i < countB; i++) classB.push({ x: rand(mid + 15, W - PAD) + noise(), y: rand(PAD, H - PAD) + noise() });
  boundaryCache = null;
}

function generateCircles(countA, countB) {
  classA = []; classB = [];
  const cx = W / 2, cy = H / 2;
  const rInner = 55, rOuter = 110, rOuterMax = 155;
  for (let i = 0; i < countA; i++) {
    const a = rand(0, Math.PI * 2), r = rand(10, rInner) + noise() * 0.5;
    classA.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
  }
  for (let i = 0; i < countB; i++) {
    const a = rand(0, Math.PI * 2), r = rand(rOuter, rOuterMax) + noise() * 0.5;
    classB.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
  }
  boundaryCache = null;
}

function generateXOR(countA, countB) {
  classA = []; classB = [];
  const cx = W / 2, cy = H / 2;
  for (let i = 0; i < Math.ceil(countA / 2); i++) {
    classA.push({ x: rand(PAD, cx - 10) + noise(), y: rand(PAD, cy - 10) + noise() });
    classA.push({ x: rand(cx + 10, W - PAD) + noise(), y: rand(cy + 10, H - PAD) + noise() });
  }
  for (let i = 0; i < Math.ceil(countB / 2); i++) {
    classB.push({ x: rand(cx + 10, W - PAD) + noise(), y: rand(PAD, cy - 10) + noise() });
    classB.push({ x: rand(PAD, cx - 10) + noise(), y: rand(cy + 10, H - PAD) + noise() });
  }
  boundaryCache = null;
}

function generateMoons(countA, countB) {
  classA = []; classB = [];
  const cx = W / 2, cy = H / 2, r = 70;
  for (let i = 0; i < countA; i++) {
    const t = rand(0, Math.PI);
    classA.push({ x: cx + r * Math.cos(t) + noise() * 0.6, y: cy + r * Math.sin(t) + noise() * 0.6 });
  }
  for (let i = 0; i < countB; i++) {
    const t = rand(Math.PI, Math.PI * 2);
    classB.push({ x: cx + 15 + r * Math.cos(t) + noise() * 0.6, y: cy + 20 + r * Math.sin(t) + noise() * 0.6 });
  }
  boundaryCache = null;
}

function generatePattern(pattern, countA, countB) {
  switch (pattern) {
    case 'linear': generateLinear(countA, countB); break;
    case 'circles': generateCircles(countA, countB); break;
    case 'xor': generateXOR(countA, countB); break;
    case 'moons': generateMoons(countA, countB); break;
    default: generateRandomData(countA, countB);
  }
  accCache = null;
}

/* ===== KNN Algorithm ===== */
function knnPredict(q, k) {
  const neighbors = [];
  const totalPoints = classA.length + classB.length;
  const warn = [];

  if (totalPoints === 0) {
    return { predicted: 'no_data', nearest: [], countA: 0, countB: 0, k: 0, warnings: [{ type: 'error', msg: 'No training data available.' }] };
  }

  const effectiveK = Math.min(k, totalPoints);
  if (k > totalPoints) warn.push({ type: 'warning', msg: `K (${k}) exceeds total points (${totalPoints}). Capped at ${effectiveK}.` });

  for (const p of classA) neighbors.push({ point: p, dist: dist(q, p), cls: 'A' });
  for (const p of classB) neighbors.push({ point: p, dist: dist(q, p), cls: 'B' });
  neighbors.sort((a, b) => a.dist - b.dist);
  const nearest = neighbors.slice(0, effectiveK);

  const avgDist = nearest.reduce((s, n) => s + n.dist, 0) / nearest.length;
  if (avgDist > SPARSE_THRESHOLD && totalPoints > 0) warn.push({ type: 'warning', msg: `Low-density (avg dist: ${avgDist.toFixed(0)}px). Unreliable.` });

  const aPct = (classA.length / totalPoints) * 100;
  const bPct = (classB.length / totalPoints) * 100;
  if (aPct > IMBALANCE_RATIO * 100 || bPct > IMBALANCE_RATIO * 100) warn.push({ type: 'info', msg: `Imbalanced (A: ${aPct.toFixed(0)}%, B: ${bPct.toFixed(0)}%).` });

  let countA = 0, countB = 0;
  for (const n of nearest) { if (n.cls === 'A') countA++; else countB++; }

  let predicted;
  if (countA > countB) predicted = 'A';
  else if (countB > countA) predicted = 'B';
  else { predicted = 'tie'; warn.push({ type: 'warning', msg: `Tie (${countA}-${countB}) for K=${effectiveK}. Try odd K.` }); }

  return { predicted, nearest, countA, countB, k: effectiveK, warnings: warn };
}

/* ===== Decision Boundary ===== */
function computeBoundary() {
  const k = Math.min(parseInt(kSlider.value), classA.length + classB.length);
  const cols = Math.ceil(W / BOUNDARY_STEP);
  const rows = Math.ceil(H / BOUNDARY_STEP);
  const grid = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const x = c * BOUNDARY_STEP + BOUNDARY_STEP / 2;
      const y = r * BOUNDARY_STEP + BOUNDARY_STEP / 2;
      const result = knnPredict({ x, y }, k);
      row.push(result.predicted);
    }
    grid.push(row);
  }
  return grid;
}

function drawBoundary() {
  if (!showBoundary) return;
  if (!boundaryCache || boundaryCache.stale) {
    boundaryCache = { grid: computeBoundary(), stale: false };
  }
  const { grid } = boundaryCache;
  const C = COLORS[currentTheme];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) {
      ctx.fillStyle = grid[r][c] === 'A' ? C.classA : grid[r][c] === 'B' ? C.classB : '#888';
      ctx.globalAlpha = 0.12;
      ctx.fillRect(c * BOUNDARY_STEP, r * BOUNDARY_STEP, BOUNDARY_STEP, BOUNDARY_STEP);
    }
  }
  ctx.globalAlpha = 1;
}

/* ===== Confusion Matrix ===== */
function computeConfusionMatrix(k) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < classA.length; i++) {
    const q = classA[i];
    const nbrs = [];
    for (let j = 0; j < classA.length; j++) if (j !== i) nbrs.push({ point: classA[j], dist: dist(q, classA[j]), cls: 'A' });
    for (const p of classB) nbrs.push({ point: p, dist: dist(q, p), cls: 'B' });
    nbrs.sort((a, b) => a.dist - b.dist);
    const nearest = nbrs.slice(0, Math.min(k, nbrs.length));
    let cA = 0, cB = 0;
    for (const n of nearest) { if (n.cls === 'A') cA++; else cB++; }
    if (cA >= cB) tp++; else fn++;
  }
  for (let i = 0; i < classB.length; i++) {
    const q = classB[i];
    const nbrs = [];
    for (const p of classA) nbrs.push({ point: p, dist: dist(q, p), cls: 'A' });
    for (let j = 0; j < classB.length; j++) if (j !== i) nbrs.push({ point: classB[j], dist: dist(q, classB[j]), cls: 'B' });
    nbrs.sort((a, b) => a.dist - b.dist);
    const nearest = nbrs.slice(0, Math.min(k, nbrs.length));
    let cA = 0, cB = 0;
    for (const n of nearest) { if (n.cls === 'A') cA++; else cB++; }
    if (cB >= cA) tn++; else fp++;
  }
  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
  const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const fpr = (tn + fp) > 0 ? fp / (tn + fp) : 0;
  return { tp, tn, fp, fn, accuracy, precision, recall, f1, tpr, fpr, total };
}

/* ===== Canvas Drawing Helpers ===== */
function drawArrow(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 5;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
  ctx.strokeStyle = COLORS[currentTheme].textMuted;
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

function drawWarningsBox(warnings) {
  warnings.innerHTML = '';
  for (const w of warnings) {
    const el = document.createElement('div');
    el.className = `warning-item ${w.type}`;
    const icons = { error: '\u26A0', warning: '\u26A0', info: '\u2139' };
    el.innerHTML = `${icons[w.type] || ''} ${w.msg}`;
    warnings.appendChild(el);
  }
}

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function findNearestTrainingPoint(px, py, threshold) {
  let best = null, bestDist = threshold;
  for (const p of classA) { const d = dist({ x: px, y: py }, p); if (d < bestDist) { bestDist = d; best = { point: p, cls: 'A' }; } }
  for (const p of classB) { const d = dist({ x: px, y: py }, p); if (d < bestDist) { bestDist = d; best = { point: p, cls: 'B' }; } }
  return best;
}

/* ===== Tooltip ===== */
function showTooltip(text, x, y) {
  tooltip.textContent = text;
  tooltip.style.left = (x + 14) + 'px';
  tooltip.style.top = (y - 8) + 'px';
  tooltip.classList.add('visible');
}
function hideTooltip() { tooltip.classList.remove('visible'); }

/* ===== Variables Panel ===== */
const VARIABLE_DEFS = [
  { id: 'k', name: 'K', def: 'Number of nearest neighbors considered for the majority vote' },
  { id: 'effectiveK', name: 'Effective K', def: 'Actual K used, capped to total points if exceeded' },
  { id: 'totalPoints', name: 'Total Points', def: 'Total labeled training samples in the dataset' },
  { id: 'classACount', name: 'Class A Count', def: 'Training samples belonging to Class A (red)' },
  { id: 'classBCount', name: 'Class B Count', def: 'Training samples belonging to Class B (blue)' },
  { id: 'imbalance', name: 'Class Imbalance', def: 'Proportion difference between classes; higher = more skew' },
  { id: 'queryPos', name: 'Query Position', def: '(x, y) coordinates of the query point' },
  { id: 'predicted', name: 'Predicted Class', def: 'Majority vote result among K nearest neighbors' },
  { id: 'votes', name: 'Vote Distribution', def: 'Count of K neighbors voting A : B' },
  { id: 'decisionMargin', name: 'Decision Margin', def: 'Vote decisiveness (|A-B|/K). 1=unanimous, 0=tie' },
  { id: 'avgDist', name: 'Avg Neighbor Dist', def: 'Mean distance from query to its K nearest neighbors' },
  { id: 'minDist', name: 'Min Neighbor Dist', def: 'Distance to the closest training point' },
  { id: 'distMetric', name: 'Distance Metric', def: 'Metric used for distance computation' },
  { id: 'weighting', name: 'Weighting', def: 'Vote weighting scheme \u2014 uniform' },
];

function renderVariables(result) {
  variablesContainer.innerHTML = '';
  const isActive = queryPoint && classA.length + classB.length > 0;
  const total = classA.length + classB.length;
  const k = parseInt(kSlider.value);
  const effectiveK = isActive ? result.k : Math.min(k, total || 1);
  const aPct = total > 0 ? (classA.length / total) * 100 : 0;
  const bPct = total > 0 ? (classB.length / total) * 100 : 0;
  const imbalance = total > 0 ? Math.abs(aPct - bPct) : 0;

  let predClass = '\u2014', predValClass = '';
  if (isActive) {
    if (result.predicted === 'A') { predClass = 'A'; predValClass = 'class-a-val'; }
    else if (result.predicted === 'B') { predClass = 'B'; predValClass = 'class-b-val'; }
    else { predClass = 'Tie'; predValClass = 'tie-val'; }
  }

  let voteStr = isActive ? `${result.countA} : ${result.countB}` : '\u2014';
  let margin = isActive ? Math.abs(result.countA - result.countB) / result.k : 0;
  let marginStr = isActive ? (margin === 0 ? '0 (tie)' : margin.toFixed(2)) : '\u2014';
  let marginClass = '';
  if (isActive) { if (margin >= 0.8) marginClass = 'conf-high'; else if (margin > 0) marginClass = 'conf-low'; }

  let avgD = isActive ? result.nearest.reduce((s, n) => s + n.dist, 0) / result.nearest.length : 0;
  let minD = isActive ? result.nearest[0].dist : 0;
  const metricNames = { euclidean: 'Euclidean (L2)', manhattan: 'Manhattan (L1)', minkowski: `Minkowski (p=${minkowskiPVal})` };

  const values = {
    k: { v: k, cls: '' },
    effectiveK: { v: effectiveK, cls: effectiveK !== k ? 'conf-low' : '' },
    totalPoints: { v: total, cls: '' },
    classACount: { v: classA.length, cls: 'class-a-val' },
    classBCount: { v: classB.length, cls: 'class-b-val' },
    imbalance: { v: total > 0 ? `${imbalance.toFixed(0)}%` : '\u2014', cls: imbalance > 50 ? 'conf-low' : '' },
    queryPos: { v: queryPoint ? `(${queryPoint.x.toFixed(0)}, ${queryPoint.y.toFixed(0)})` : '\u2014', cls: '' },
    predicted: { v: predClass, cls: predValClass },
    votes: { v: voteStr, cls: '' },
    decisionMargin: { v: marginStr, cls: marginClass },
    avgDist: { v: isActive ? avgD.toFixed(1) : '\u2014', cls: '' },
    minDist: { v: isActive ? minD.toFixed(1) : '\u2014', cls: '' },
    distMetric: { v: metricNames[metric] || 'Euclidean (L2)', cls: '' },
    weighting: { v: 'Uniform', cls: '' },
  };

  for (const vd of VARIABLE_DEFS) {
    const row = document.createElement('div');
    row.className = 'var-row';
    const left = document.createElement('div');
    left.className = 'var-left';
    left.innerHTML = `<div class="var-name">${vd.name}</div><div class="var-def">${vd.def}</div>`;
    const val = document.createElement('div');
    const info = values[vd.id];
    val.className = `var-value ${info.cls}`;
    val.textContent = info.v;
    row.appendChild(left);
    row.appendChild(val);
    variablesContainer.appendChild(row);
  }
}

/* ===== Neighbor Matrix ===== */
function renderNeighborMatrix(result) {
  neighborMatrixContainer.innerHTML = '';
  const total = classA.length + classB.length;
  if (total === 0) { neighborMatrixContainer.innerHTML = '<div class="empty-msg">No training data</div>'; return; }
  if (!queryPoint) { neighborMatrixContainer.innerHTML = '<div class="empty-msg">Click canvas to place a query</div>'; return; }

  const table = document.createElement('table');
  table.className = 'matrix-table';
  table.innerHTML = '<thead><tr><th class="rank-col">#</th><th class="cls-col">Cls</th><th>Distance</th><th class="vote-col">Vote</th></tr></thead>';
  const tbody = document.createElement('tbody');
  const winningClass = result.predicted;

  for (let i = 0; i < result.nearest.length; i++) {
    const n = result.nearest[i];
    const tr = document.createElement('tr');
    const rankCell = document.createElement('td'); rankCell.className = 'rank-col'; rankCell.textContent = i + 1;
    const clsCell = document.createElement('td'); clsCell.className = `cls-col ${n.cls === 'A' ? 'cls-a' : 'cls-b'}`; clsCell.textContent = n.cls;
    const distCell = document.createElement('td'); distCell.className = 'dist-col'; distCell.textContent = n.dist.toFixed(1);
    const voteCell = document.createElement('td'); voteCell.className = 'vote-col';
    if (winningClass === 'tie') { voteCell.textContent = '\u2014'; voteCell.style.color = 'var(--text-muted)'; }
    else if (n.cls === winningClass) { voteCell.textContent = '\u2713'; voteCell.style.color = 'var(--success)'; }
    else { voteCell.textContent = '\u2014'; voteCell.style.color = 'var(--text-muted)'; }
    tr.appendChild(rankCell); tr.appendChild(clsCell); tr.appendChild(distCell); tr.appendChild(voteCell);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  neighborMatrixContainer.appendChild(table);
}

/* ===== Confusion Matrix Panel ===== */
function renderConfusionMatrix() {
  confusionMatrixContainer.innerHTML = '';
  const total = classA.length + classB.length;
  if (total === 0) { confusionMatrixContainer.innerHTML = '<div class="empty-msg">No training data</div>'; return; }

  const k = Math.min(parseInt(kSlider.value), total);
  const cm = computeConfusionMatrix(k);
  const C = COLORS[currentTheme];

  const grid = document.createElement('div');
  grid.className = 'cm-grid';
  grid.innerHTML = `
    <div class="cm-label-row"></div>
    <div class="cm-label-col">Pred<br><strong>A</strong></div>
    <div class="cm-label-col">Pred<br><strong>B</strong></div>
    <div class="cm-label-row"></div>
    <div class="cm-label-row actual"><strong>A</strong><br><span>Actual</span></div>
    <div class="cm-cell tp">${cm.tp}<span class="cm-sublabel">TP</span></div>
    <div class="cm-cell fn">${cm.fn}<span class="cm-sublabel">FN</span></div>
    <div class="cm-label-row" style="font-size:9px;color:${C.success};font-weight:600;">TPR ${(cm.tpr * 100).toFixed(0)}%</div>
    <div class="cm-label-row actual"><strong>B</strong><br><span>Actual</span></div>
    <div class="cm-cell fp">${cm.fp}<span class="cm-sublabel">FP</span></div>
    <div class="cm-cell tn">${cm.tn}<span class="cm-sublabel">TN</span></div>
    <div class="cm-label-row" style="font-size:9px;color:${C.danger};font-weight:600;">FPR ${(cm.fpr * 100).toFixed(0)}%</div>
  `;
  confusionMatrixContainer.appendChild(grid);

  const stats = document.createElement('div');
  stats.className = 'cm-stats';
  for (const m of [
    { label: 'Accuracy', value: `${(cm.accuracy * 100).toFixed(1)}%` },
    { label: 'Precision', value: `${(cm.precision * 100).toFixed(1)}%` },
    { label: 'Recall (TPR)', value: `${(cm.recall * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(cm.f1 * 100).toFixed(1)}%` },
  ]) {
    const stat = document.createElement('div');
    stat.className = 'cm-stat';
    stat.innerHTML = `<div class="stat-label">${m.label}</div><div class="stat-value">${m.value}</div>`;
    stats.appendChild(stat);
  }
  confusionMatrixContainer.appendChild(stats);
}

/* ===== K vs Accuracy Chart ===== */
function computeAccuracyCurve() {
  const total = classA.length + classB.length;
  if (total === 0) { accCache = []; return []; }
  const maxK = Math.min(20, total);
  if (accCache && accCache.length === maxK) return accCache;

  const curve = [];
  for (let k = 1; k <= maxK; k++) {
    const cm = computeConfusionMatrix(k);
    curve.push({ k, accuracy: cm.accuracy });
  }
  accCache = curve;
  return curve;
}

function drawAccuracyChart() {
  const CW = accChart.width, CH = accChart.height;
  const pad = { t: 10, r: 8, b: 18, l: 24 };
  const plotW = CW - pad.l - pad.r;
  const plotH = CH - pad.t - pad.b;

  accCtx.clearRect(0, 0, CW, CH);

  const total = classA.length + classB.length;
  if (total === 0) {
    accCtx.fillStyle = COLORS[currentTheme].textMuted;
    accCtx.font = '9px Inter, sans-serif';
    accCtx.textAlign = 'center';
    accCtx.textBaseline = 'middle';
    accCtx.fillText('No data', CW / 2, CH / 2);
    return;
  }

  const curve = computeAccuracyCurve();
  const C = COLORS[currentTheme];
  const currentK = parseInt(kSlider.value);

  function xForK(k) { return pad.l + ((k - 1) / (curve.length - 1 || 1)) * plotW; }
  function yForAcc(a) { return pad.t + (1 - a) * plotH; }

  accCtx.strokeStyle = C.textMuted;
  accCtx.globalAlpha = 0.15;
  accCtx.lineWidth = 0.5;
  accCtx.setLineDash([2, 3]);
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = yForAcc(pct / 100);
    accCtx.beginPath(); accCtx.moveTo(pad.l, y); accCtx.lineTo(CW - pad.r, y); accCtx.stroke();
  }
  accCtx.setLineDash([]);
  accCtx.globalAlpha = 1;

  accCtx.fillStyle = C.textMuted;
  accCtx.font = '7px Inter, sans-serif';
  accCtx.textAlign = 'right';
  accCtx.textBaseline = 'middle';
  for (let pct = 0; pct <= 100; pct += 25) {
    accCtx.fillText(pct + '%', pad.l - 3, yForAcc(pct / 100));
  }

  accCtx.textAlign = 'center';
  accCtx.textBaseline = 'top';
  const xStep = Math.max(1, Math.floor(curve.length / 6));
  for (let i = 0; i < curve.length; i += xStep) {
    accCtx.fillText('K=' + curve[i].k, xForK(curve[i].k), CH - pad.b + 3);
  }
  accCtx.fillText('K=' + curve[curve.length - 1].k, xForK(curve[curve.length - 1].k), CH - pad.b + 3);

  accCtx.beginPath();
  accCtx.strokeStyle = C.classB;
  accCtx.lineWidth = 1.8;
  accCtx.globalAlpha = 0.7;
  for (let i = 0; i < curve.length; i++) {
    const x = xForK(curve[i].k), y = yForAcc(curve[i].accuracy);
    i === 0 ? accCtx.moveTo(x, y) : accCtx.lineTo(x, y);
  }
  accCtx.stroke();
  accCtx.globalAlpha = 1;

  const idx = Math.min(currentK, curve.length) - 1;
  if (idx >= 0) {
    const mx = xForK(curve[idx].k), my = yForAcc(curve[idx].accuracy);
    accCtx.beginPath();
    accCtx.arc(mx, my, 4, 0, Math.PI * 2);
    accCtx.fillStyle = C.classA;
    accCtx.fill();
    accCtx.strokeStyle = 'rgba(255,255,255,0.7)';
    accCtx.lineWidth = 1.5;
    accCtx.stroke();

    accCtx.fillStyle = C.classA;
    accCtx.font = 'bold 8px Inter, sans-serif';
    accCtx.textAlign = 'left';
    accCtx.textBaseline = 'bottom';
    accCtx.fillText((curve[idx].accuracy * 100).toFixed(0) + '%', mx + 6, my - 1);
  }
}

/* ===== Explain Mode ===== */
function updateExplain(result) {
  if (explainStep === 0) {
    explainBox.style.display = 'none';
    explainBtn.textContent = 'Explain Step 0/4';
    return;
  }
  const step = EXPLAIN_STEPS[explainStep];
  explainBox.style.display = 'block';
  explainBtn.textContent = `Explain Step ${explainStep}/4`;
  explainBox.innerHTML = `<div class="step-title">${step.title}</div><div>${step.text}</div>`;
}

function drawExplainOverlay(result) {
  if (explainStep === 0 || !queryPoint) return;
  const C = COLORS[currentTheme];

  if (explainStep >= 1) {
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const allPoints = [...classA.map(p => ({ point: p, cls: 'A' })), ...classB.map(p => ({ point: p, cls: 'B' }))];
    for (const { point: p, cls } of allPoints) {
      const d = dist(queryPoint, p);
      ctx.fillStyle = cls === 'A' ? C.classA : C.classB;
      ctx.globalAlpha = 0.7;
      ctx.fillText(d.toFixed(1), p.x, p.y - POINT_RADIUS - 3);
    }
    ctx.globalAlpha = 1;
  }

  if (explainStep >= 3 && result) {
    for (const n of result.nearest) {
      ctx.beginPath();
      ctx.arc(n.point.x, n.point.y, 13, 0, Math.PI * 2);
      ctx.strokeStyle = n.cls === 'A' ? C.classA : C.classB;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}

/* ===== Bias-Variance Risk Badge ===== */
function updateBiasBadge() {
  const total = classA.length + classB.length;
  const k = parseInt(kSlider.value);
  if (total === 0) { biasBadge.className = ''; biasBadge.textContent = ''; return; }

  const lowKThreshold = Math.max(2, Math.ceil(total * 0.05));
  const highKThreshold = Math.floor(total * 0.7);

  if (k <= lowKThreshold) {
    biasBadge.className = 'visible warning';
    biasBadge.textContent = '\u2191 Variance (overfit)';
  } else if (k >= highKThreshold) {
    biasBadge.className = 'visible warning';
    biasBadge.textContent = '\u2191 Bias (underfit)';
  } else {
    biasBadge.className = '';
    biasBadge.textContent = '';
  }
}

/* ===== Animated Neighbor Lines ===== */
function startNeighborAnim() {
  if (animRAF) { cancelAnimationFrame(animRAF); animRAF = null; }
  animAlpha = 0;
  const startTime = performance.now();
  const duration = 200;

  function tick(now) {
    const t = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 2);
    animAlpha = ease;
    draw(ease);
    if (t < 1) { animRAF = requestAnimationFrame(tick); }
    else { animAlpha = null; animRAF = null; draw(1); }
  }
  animRAF = requestAnimationFrame(tick);
}

/* ===== Keyboard Shortcuts ===== */
document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

  const tip = document.getElementById('firstTip');
  if (tip && !tip.classList.contains('fadeout')) tip.classList.add('fadeout');

  switch (e.key) {
    case '+': case '=': case 'ArrowUp':
      e.preventDefault();
      kSlider.value = Math.min(parseInt(kSlider.value) + 1, parseInt(kSlider.max));
      kSlider.dispatchEvent(new Event('input'));
      break;
    case '-': case '_': case 'ArrowDown':
      e.preventDefault();
      kSlider.value = Math.max(parseInt(kSlider.value) - 1, parseInt(kSlider.min));
      kSlider.dispatchEvent(new Event('input'));
      break;
    case 'n': case 'N':
      e.preventDefault(); randomBtn.click(); break;
    case 'e': case 'E':
      e.preventDefault(); explainBtn.click(); break;
    case 't': case 'T':
      e.preventDefault(); themeToggle.click(); break;
    case 'c': case 'C':
      e.preventDefault(); clearQueryBtn.click(); break;
  }
});

/* ===== Export Snapshot ===== */
function downloadSnapshot() {
  const total = classA.length + classB.length;
  const k = parseInt(kSlider.value);
  const effectiveK = Math.min(k, total || 1);
  const pattern = patternSelect.options[patternSelect.selectedIndex].text;
  let predText = 'No query point';
  let accText = '';
  if (queryPoint && total > 0) {
    const result = knnPredict(queryPoint, k);
    if (result.predicted === 'A') predText = 'Class A';
    else if (result.predicted === 'B') predText = 'Class B';
    else predText = 'Tie';
    predText += ` (${result.countA}:${result.countB} votes)`;
  }
  if (total > 0) {
    const cm = computeConfusionMatrix(effectiveK);
    accText = `Accuracy: ${(cm.accuracy * 100).toFixed(1)}%`;
  }

  const pad = 20, barH = 100;
  const offscreen = document.createElement('canvas');
  offscreen.width = W + pad * 2;
  offscreen.height = H + pad * 2 + barH;
  const offCtx = offscreen.getContext('2d');

  offCtx.fillStyle = '#0f1625';
  offCtx.fillRect(0, 0, offscreen.width, offscreen.height);
  offCtx.drawImage(canvas, pad, pad);

  offCtx.fillStyle = '#e2e8f0';
  offCtx.font = '13px Inter, sans-serif';
  offCtx.textAlign = 'left';
  offCtx.textBaseline = 'top';

  const y = H + pad * 2 + 10;
  const leftX = pad;
  const rightX = offscreen.width / 2 + 10;

  offCtx.font = 'bold 14px Inter, sans-serif';
  offCtx.fillText('KNN Snapshot', leftX, y);
  offCtx.font = '12px Inter, sans-serif';
  offCtx.fillStyle = '#94a3b8';
  offCtx.fillText(`Pattern: ${pattern}`, leftX, y + 22);
  offCtx.fillText(`K: ${effectiveK}${effectiveK !== k ? ` (capped from ${k})` : ''}`, leftX, y + 40);
  offCtx.fillText(`Prediction: ${predText}`, leftX, y + 58);
  offCtx.fillText(`Points: ${total}`, rightX, y + 22);
  offCtx.fillText(accText, rightX, y + 40);
  offCtx.fillText(`Distance: ${metric}`, rightX, y + 58);

  const link = document.createElement('a');
  link.download = 'knn-snapshot.png';
  link.href = offscreen.toDataURL('image/png');
  link.click();
}

/* ===== Main Draw ===== */
function draw(animT) {
  const C = COLORS[currentTheme];
  const neighborAlpha = animT !== undefined ? animT : 1;

  ctx.clearRect(0, 0, W, H);

  const totalPoints = classA.length + classB.length;

  /* Empty state */
  if (totalPoints === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No training data', W / 2, H / 2 - 8);
    ctx.fillText('Click "New Data" to begin', W / 2, H / 2 + 12);

    pointCount.textContent = 'A: 0 \u2022 B: 0';
    prediction.textContent = 'No data to classify';
    prediction.className = '';
    if (queryPoint) drawWarningsBox([{ type: 'error', msg: 'No training data available.' }]);
    else warnings.innerHTML = '';
    renderVariables(null);
    renderNeighborMatrix(null);
    renderConfusionMatrix();
    updateExplain(null);
    drawAccuracyChart();
    updateBiasBadge();
    return;
  }

  drawBoundary();

  /* Draw Class A */
  for (const p of classA) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, POINT_RADIUS, 0, Math.PI * 2);
    if (hoveredPoint && hoveredPoint.point === p && hoveredPoint.cls === 'A') {
      ctx.shadowColor = C.classAGlow;
      ctx.shadowBlur = 14;
    }
    ctx.fillStyle = C.classA;
    ctx.fill();
    ctx.strokeStyle = C.classABorder;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  /* Draw Class B */
  for (const p of classB) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, POINT_RADIUS, 0, Math.PI * 2);
    if (hoveredPoint && hoveredPoint.point === p && hoveredPoint.cls === 'B') {
      ctx.shadowColor = C.classBGlow;
      ctx.shadowBlur = 14;
    }
    ctx.fillStyle = C.classB;
    ctx.fill();
    ctx.strokeStyle = C.classBBorder;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  renderConfusionMatrix();

  /* Query point */
  if (queryPoint) {
    const k = parseInt(kSlider.value);
    const result = knnPredict(queryPoint, k);

    if (result.predicted === 'no_data') {
      drawWarningsBox(result.warnings);
      pointCount.textContent = 'A: 0 \u2022 B: 0';
      prediction.textContent = 'No data to classify';
      prediction.className = '';
      renderVariables(null);
      renderNeighborMatrix(null);
      updateExplain(null);
      drawAccuracyChart();
      updateBiasBadge();
      return;
    }

    drawWarningsBox(result.warnings);
    renderVariables(result);
    renderNeighborMatrix(result);
    updateExplain(result);
    drawExplainOverlay(result);

    /* Sparsity ring */
    if (result.warnings.some(w => w.msg.includes('Low-density'))) {
      ctx.beginPath();
      ctx.arc(queryPoint.x, queryPoint.y, SPARSE_THRESHOLD, 0, Math.PI * 2);
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.04;
      ctx.fillStyle = '#fbbf24';
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    /* Neighbor lines */
    ctx.globalAlpha = neighborAlpha;
    for (const n of result.nearest) {
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.moveTo(queryPoint.x, queryPoint.y);
      ctx.lineTo(n.point.x, n.point.y);
      ctx.strokeStyle = n.cls === 'A' ? C.classA : C.classB;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      drawArrow(queryPoint.x, queryPoint.y, n.point.x, n.point.y);
    }

    /* Neighbor highlight rings */
    ctx.globalAlpha = neighborAlpha * 0.35;
    for (const n of result.nearest) {
      ctx.beginPath();
      ctx.arc(n.point.x, n.point.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = n.cls === 'A' ? C.classA : C.classB;
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    /* Query diamond */
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(queryPoint.x, queryPoint.y - 12);
    ctx.lineTo(queryPoint.x + 8, queryPoint.y);
    ctx.lineTo(queryPoint.x, queryPoint.y + 12);
    ctx.lineTo(queryPoint.x - 8, queryPoint.y);
    ctx.closePath();
    ctx.shadowColor = C.queryGlow;
    ctx.shadowBlur = 20;

    if (result.predicted === 'tie') {
      ctx.clip();
      ctx.fillStyle = C.classA;
      ctx.fillRect(queryPoint.x - 10, queryPoint.y - 14, 20, 14);
      ctx.fillStyle = C.classB;
      ctx.fillRect(queryPoint.x - 10, queryPoint.y, 20, 14);
      ctx.restore();
      ctx.save();
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.moveTo(queryPoint.x, queryPoint.y - 12);
      ctx.lineTo(queryPoint.x + 8, queryPoint.y);
      ctx.lineTo(queryPoint.x, queryPoint.y + 12);
      ctx.lineTo(queryPoint.x - 8, queryPoint.y);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('?', queryPoint.x, queryPoint.y - 4);
      ctx.restore();
    } else {
      ctx.fillStyle = C.query;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = 'white';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = result.predicted === 'A' ? 'A' : result.predicted === 'B' ? 'B' : '?';
      ctx.fillText(label, queryPoint.x, queryPoint.y);
    }

    pointCount.textContent = `A: ${classA.length}  \u2022  B: ${classB.length}  \u2022  Total: ${totalPoints}`;

    let predText;
    prediction.className = '';
    if (result.predicted === 'A') { predText = `Pred: Class A (${result.countA}/${result.k})`; prediction.classList.add('class-a-pred'); }
    else if (result.predicted === 'B') { predText = `Pred: Class B (${result.countB}/${result.k})`; prediction.classList.add('class-b-pred'); }
    else { predText = `Tie (${result.countA}-${result.countB})`; prediction.classList.add('tie'); }
    prediction.textContent = predText;
  } else {
    pointCount.textContent = `A: ${classA.length}  \u2022  B: ${classB.length}  \u2022  Total: ${totalPoints}`;
    prediction.textContent = 'Click canvas to place query';
    prediction.className = '';
    warnings.innerHTML = '';
    renderVariables(null);
    renderNeighborMatrix(null);
    updateExplain(null);
  }
  drawAccuracyChart();
  updateBiasBadge();
}

/* ===== Event Handlers ===== */

function handleCanvasMove(e) {
  if (isDragging) return;
  const coords = getCanvasCoords(e);
  const hit = findNearestTrainingPoint(coords.x, coords.y, HOVER_RADIUS);
  if (hit && queryPoint && dist(hit.point, queryPoint) < DRAG_THRESHOLD) {
    hideTooltip(); hoveredPoint = null; return;
  }
  if (hit) {
    if (!hoveredPoint || hoveredPoint.point !== hit.point) { hoveredPoint = hit; draw(); }
    showTooltip(`Class ${hit.cls}  (${hit.point.x.toFixed(0)}, ${hit.point.y.toFixed(0)})`, e.clientX, e.clientY);
  } else {
    if (hoveredPoint) { hoveredPoint = null; draw(); }
    hideTooltip();
  }
}

function handleCanvasDown(e) {
  const coords = getCanvasCoords(e);

  /* Add mode */
  if (addMode === 'addA' || addMode === 'addB') {
    const cls = addMode === 'addA' ? classA : classB;
    cls.push({ x: coords.x, y: coords.y });
    boundaryCache = null; accCache = null;
    if (explainStep !== 0) { explainStep = 0; }
    draw();
    return;
  }

  /* Drag existing query */
  if (queryPoint && dist(coords, queryPoint) < DRAG_THRESHOLD) {
    isDragging = true;
    canvas.style.cursor = 'grabbing';
    return;
  }

  /* Place new query point */
  queryPoint = { x: coords.x, y: coords.y };
  hoveredPoint = null;
  hideTooltip();
  draw();
}

function handleCanvasDrag(e) {
  if (!isDragging) return;
  const coords = getCanvasCoords(e);
  queryPoint = { x: coords.x, y: coords.y };
  hoveredPoint = null;
  hideTooltip();
  draw();
}

function handleCanvasUp() {
  if (isDragging) { isDragging = false; canvas.style.cursor = 'crosshair'; draw(); }
}

function handleCanvasLeave() {
  hideTooltip();
  if (hoveredPoint) { hoveredPoint = null; draw(); }
  if (isDragging) { isDragging = false; canvas.style.cursor = 'crosshair'; }
}

/* ===== Controls ===== */

kSlider.addEventListener('input', () => {
  kValue.classList.remove('pulse');
  void kValue.offsetWidth;
  kValue.classList.add('pulse');
  kValue.textContent = kSlider.value;
  boundaryCache = null;
  if (queryPoint && classA.length + classB.length > 0) {
    startNeighborAnim();
  } else {
    renderConfusionMatrix();
    if (classA.length + classB.length > 0) renderVariables(null);
    drawAccuracyChart();
    updateBiasBadge();
  }
});

randomBtn.addEventListener('click', () => {
  generatePattern(patternSelect.value, 12, 12);
  queryPoint = null; hoveredPoint = null; hideTooltip();
  if (explainStep !== 0) explainStep = 0;
  draw();
});

patternSelect.addEventListener('change', () => {
  generatePattern(patternSelect.value, 12, 12);
  queryPoint = null; hoveredPoint = null; hideTooltip();
  if (explainStep !== 0) explainStep = 0;
  draw();
});

clearQueryBtn.addEventListener('click', () => {
  queryPoint = null; hoveredPoint = null; hideTooltip(); draw();
});

clearAllBtn.addEventListener('click', () => {
  classA = []; classB = []; queryPoint = null; hoveredPoint = null; boundaryCache = null; accCache = null; hideTooltip();
  if (explainStep !== 0) explainStep = 0;
  draw();
});

/* ===== Distance Metric ===== */
metricSelect.addEventListener('change', () => {
  metric = metricSelect.value;
  const showP = metric === 'minkowski';
  minkowskiP.style.display = showP ? 'inline-block' : 'none';
  boundaryCache = null; accCache = null;
  draw();
});

minkowskiP.addEventListener('change', () => {
  minkowskiPVal = parseFloat(minkowskiP.value) || 3;
  boundaryCache = null; accCache = null;
  draw();
});

/* ===== Decision Boundary Toggle ===== */
boundaryToggle.addEventListener('change', () => {
  showBoundary = boundaryToggle.checked;
  boundaryCache = null;
  draw();
});

/* ===== Explain Mode ===== */
explainBtn.addEventListener('click', () => {
  if (classA.length + classB.length === 0) return;
  if (!queryPoint) {
    explainStep = 0;
    explainBox.style.display = 'none';
    explainBtn.textContent = 'Explain Step 0/4';
    return;
  }
  explainStep = (explainStep % 4) + 1;
  draw();
});

downloadBtn.addEventListener('click', downloadSnapshot);

/* ===== Add Mode Buttons ===== */
function setAddMode(mode) {
  addMode = mode;
  [modeClassify, modeAddA, modeAddB].forEach(b => b.classList.remove('active'));
  if (mode === 'classify') modeClassify.classList.add('active');
  else if (mode === 'addA') modeAddA.classList.add('active');
  else if (mode === 'addB') modeAddB.classList.add('active');
  canvas.title = mode === 'classify' ? 'Click to place query point' : `Click to add Class ${mode === 'addA' ? 'A' : 'B'} point`;
}

modeClassify.addEventListener('click', () => setAddMode('classify'));
modeAddA.addEventListener('click', () => setAddMode('addA'));
modeAddB.addEventListener('click', () => setAddMode('addB'));

/* ===== Canvas Events ===== */
canvas.addEventListener('mousedown', handleCanvasDown);
canvas.addEventListener('mousemove', handleCanvasMove);
canvas.addEventListener('mousemove', handleCanvasDrag);
canvas.addEventListener('mouseup', handleCanvasUp);
canvas.addEventListener('mouseleave', handleCanvasLeave);

/* ===== Theme Toggle ===== */
function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('knn-theme', theme);
  themeToggle.textContent = theme === 'dark' ? '\u2600' : '\u263E';
  draw();
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

/* ===== Panel footers ===== */
leftPanelFooter.textContent = 'KNN algorithm state variables';
rightPanelFooter.textContent = 'Leave-one-out cross-validation on training data';

/* ===== Init ===== */
const savedTheme = localStorage.getItem('knn-theme') || 'dark';
applyTheme(savedTheme);
generatePattern('random', 12, 12);
draw();

/* First-launch tip */
const tip = document.getElementById('firstTip');
if (!localStorage.getItem('knn-tip-seen')) {
  localStorage.setItem('knn-tip-seen', '1');
  setTimeout(() => { if (tip) tip.classList.add('fadeout'); }, 5000);
  document.addEventListener('keydown', () => {
    if (tip && !tip.classList.contains('fadeout')) tip.classList.add('fadeout');
  }, { once: true });
  document.addEventListener('click', () => {
    if (tip && !tip.classList.contains('fadeout')) tip.classList.add('fadeout');
  }, { once: true });
} else {
  if (tip) { tip.classList.add('fadeout'); tip.style.display = 'none'; }
}
