/* ===== Neural Network Visualizer — JavaScript ===== */

/* ===== DOM References ===== */
const canvas = document.getElementById('nnCanvas');
const ctx = canvas.getContext('2d');
const weightPopup = document.getElementById('nnWeightPopup');
const themeToggle = document.getElementById('nnThemeToggle');

const nnConfig = {
  inputInc: document.getElementById('nnInputInc'), inputDec: document.getElementById('nnInputDec'), inputVal: document.getElementById('nnInputVal'),
  hidden1Inc: document.getElementById('nnH1Inc'), hidden1Dec: document.getElementById('nnH1Dec'), hidden1Val: document.getElementById('nnH1Val'),
  hidden2Inc: document.getElementById('nnH2Inc'), hidden2Dec: document.getElementById('nnH2Dec'), hidden2Val: document.getElementById('nnH2Val'),
  outputInc: document.getElementById('nnOutInc'), outputDec: document.getElementById('nnOutDec'), outputVal: document.getElementById('nnOutVal'),
};

const actSelect = document.getElementById('nnActSelect');
const actCurveCanvas = document.getElementById('nnActCurve');
const actCurveCtx = actCurveCanvas.getContext('2d');

const inputSliders = document.querySelectorAll('.nn-input-row input[type=range]');
const inputLabels = document.querySelectorAll('.nn-input-row .iv');

const runBtn = document.getElementById('nnRunBtn');
const randomBtn = document.getElementById('nnRandomBtn');
const resetBtn = document.getElementById('nnResetBtn');

const compPanel = document.getElementById('nnCompPanel');
const legend = document.getElementById('nnLegend');

/* ===== Constants ===== */
const W = canvas.width, H = canvas.height;
const NEURON_R = 18;
const PAD_X = 40, PAD_Y = 30;

/* ===== State ===== */
let arch = [3, 4, 0, 2]; // input, hidden1, hidden2, output
let activation = 'sigmoid';
let network = null;
let animState = 'idle'; // idle | running | done
let animPhase = -1; // -1=idle, 0=input, 1..n = layers processing
let animProgress = 0; // 0-1
let animLayerStart = 0;
let animId = null;
let hoveredConnection = null;
let selectedConnection = null;
let inputValues = [0.5, 0.3, 0.8];
let currentTheme = 'dark';

/* ===== Activation Functions ===== */
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function relu(z) { return Math.max(0, z); }
function tanh(z) { return Math.tanh(z); }

function activate(z, fn) {
  if (fn === 'sigmoid') return sigmoid(z);
  if (fn === 'relu') return relu(z);
  if (fn === 'tanh') return tanh(z);
  return z;
}

function activateDeriv(z, fn) {
  if (fn === 'sigmoid') { const s = sigmoid(z); return s * (1 - s); }
  if (fn === 'relu') return z > 0 ? 1 : 0;
  if (fn === 'tanh') { const t = Math.tanh(z); return 1 - t * t; }
  return 1;
}

/* ===== Network Construction ===== */
function buildNetwork(arch_) {
  const layers = [];
  for (let li = 0; li < arch_.length; li++) {
    const n = arch_[li];
    if (n === 0) continue;
    const neurons = [];
    for (let ni = 0; ni < n; ni++) {
      neurons.push({
        value: 0, bias: (Math.random() - 0.5) * 0.4,
        z: 0, activation: 0,
        active: false, animated: false,
      });
    }
    layers.push({ neurons, label: li === 0 ? 'Input' : li === arch_.length - 1 ? 'Output' : `Hidden ${li}` });
  }
  const weights = [];
  for (let li = 0; li < layers.length - 1; li++) {
    const conn = [];
    for (let fi = 0; fi < layers[li].neurons.length; fi++) {
      for (let ti = 0; ti < layers[li + 1].neurons.length; ti++) {
        conn.push({ fromLayer: li, from: fi, toLayer: li + 1, to: ti, w: (Math.random() - 0.5) * 1.2 });
      }
    }
    weights.push(conn);
  }
  return { layers, weights };
}

function randomizeNetwork() {
  network = buildNetwork(arch);
  animState = 'idle'; animPhase = -1;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  clearNeuronActive();
}

function resetNetwork() {
  network = buildNetwork(arch);
  for (const wg of network.weights) {
    for (const c of wg) c.w = (Math.random() - 0.5) * 0.6;
  }
  for (const layer of network.layers) {
    for (const n of layer.neurons) n.bias = (Math.random() - 0.5) * 0.3;
  }
  animState = 'idle'; animPhase = -1;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  clearNeuronActive();
}

function clearNeuronActive() {
  for (const layer of network.layers)
    for (const n of layer.neurons) { n.active = false; n.animated = false; n.z = 0; n.activation = 0; n.value = 0; }
}

/* ===== Forward Pass ===== */
function computeForward(inputs, net) {
  const { layers, weights } = net;
  // Set input values
  for (let i = 0; i < Math.min(inputs.length, layers[0].neurons.length); i++) {
    layers[0].neurons[i].value = inputs[i];
    layers[0].neurons[i].activation = inputs[i];
    layers[0].neurons[i].z = inputs[i];
  }
  // Compute hidden/output layers
  for (let li = 1; li < layers.length; li++) {
    const prev = layers[li - 1].neurons;
    const curr = layers[li].neurons;
    const wg = weights[li - 1];
    for (let ni = 0; ni < curr.length; ni++) {
      let z = curr[ni].bias || 0;
      for (let pi = 0; pi < prev.length; pi++) {
        const connIdx = pi * curr.length + ni;
        if (connIdx < wg.length) z += prev[pi].activation * wg[connIdx].w;
      }
      curr[ni].z = z;
      curr[ni].activation = activate(z, activation);
    }
  }
  updateCompPanel();
}

/* ===== Layout Positions ===== */
function getNeuronPos(layerIdx, neuronIdx, layers) {
  const layerCount = layers.length;
  const nCount = layers[layerIdx].neurons.length;
  const spacingX = (W - PAD_X * 2) / Math.max(layerCount - 1, 1);
  const spacingY = (H - PAD_Y * 2) / Math.max(nCount - 1, 1);
  const x = PAD_X + layerIdx * spacingX;
  const y = nCount === 1 ? H / 2 : PAD_Y + neuronIdx * spacingY;
  return { x, y };
}

function getWeightCenter(c, layers) {
  const from = getNeuronPos(c.fromLayer, c.from, layers);
  const to = getNeuronPos(c.toLayer, c.to, layers);
  return { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
}

/* ===== Drawing ===== */
function getColors() {
  const t = document.documentElement.getAttribute('data-theme') || 'dark';
  if (t === 'light') return {
    bg: '#fafbff', border: '#d0d6e0', text: '#161b2e', text2: '#4a5570', text3: '#8892b0',
    accent: '#6c4ce0', surface2: '#f4f6fb', surface3: '#e8ecf4',
    pos: '#3498db', neg: '#e05252', glow: 'rgba(108,76,224,0.3)',
  };
  return {
    bg: '#0f1625', border: '#283048', text: '#e2e8f0', text2: '#94a3b8', text3: '#64748b',
    accent: '#7c5cfc', surface2: '#1a2032', surface3: '#222a40',
    pos: '#4fc3f7', neg: '#ff6b6b', glow: 'rgba(124,92,252,0.35)',
  };
}

function drawNetwork(animPhase_, animProgress_, highlightLayer) {
  if (!network) return;
  const C = getColors();
  ctx.clearRect(0, 0, W, H);
  const { layers, weights } = network;

  // Determine which weights to highlight during animation
  const highlightWeights = new Set();
  if (highlightLayer >= 0 && highlightLayer < weights.length) {
    for (let i = 0; i < weights[highlightLayer].length; i++) highlightWeights.add(i);
  }

  // Draw connections
  for (let li = 0; li < weights.length; li++) {
    for (let ci = 0; ci < weights[li].length; ci++) {
      const c = weights[li][ci];
      const from = getNeuronPos(c.fromLayer, c.from, layers);
      const to = getNeuronPos(c.toLayer, c.to, layers);
      const absW = Math.abs(c.w);
      const thickness = 1 + absW * 3;
      const isHighlighted = highlightLayer === li;
      let alpha = 0.25 + absW * 0.5;
      let color = c.w >= 0 ? C.pos : C.neg;

      if (isHighlighted && animPhase_ >= 0) {
        alpha = 0.3 + animProgress_ * 0.6;
        // Traveling dot animation
        const dotProgress = animProgress_;
        const dx = to.x - from.x, dy = to.y - from.y;
        const dotX = from.x + dx * dotProgress;
        const dotY = from.y + dy * dotProgress;

        ctx.beginPath();
        ctx.arc(dotX, dotY, 4 + absW * 2, 0, Math.PI * 2);
        ctx.fillStyle = c.w >= 0 ? C.pos : C.neg;
        ctx.globalAlpha = 0.7 + animProgress_ * 0.3;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.globalAlpha = Math.min(alpha, 0.7);
      ctx.lineWidth = thickness;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Hover effect
      if (hoveredConnection && hoveredConnection.fromLayer === li && hoveredConnection.from === c.from && hoveredConnection.to === c.to) {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = C.accent;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = thickness + 3;
        ctx.setLineDash([3, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }
  }

  // Draw neurons
  for (let li = 0; li < layers.length; li++) {
    for (let ni = 0; ni < layers[li].neurons.length; ni++) {
      const n = layers[li].neurons[ni];
      const pos = getNeuronPos(li, ni, layers);
      const isActive = n.active || (animState === 'done');

      // Glow for active
      if (isActive) {
        const grd = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, NEURON_R * 2.5);
        grd.addColorStop(0, C.glow);
        grd.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, NEURON_R * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();
      }

      // Neuron circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, NEURON_R, 0, Math.PI * 2);
      const activationVal = isActive ? n.activation : 0;
      const intensity = Math.abs(activationVal);
      ctx.fillStyle = isActive
        ? `rgba(124, 92, 252, ${0.2 + intensity * 0.5})`
        : C.surface2;
      ctx.fill();
      ctx.strokeStyle = isActive ? C.accent : C.border;
      ctx.lineWidth = isActive ? 2 : 1.2;
      ctx.stroke();

      // Neuron label (index or type)
      ctx.fillStyle = C.text3;
      ctx.font = '7px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      let label = li === 0 ? `x${ni + 1}` : li === layers.length - 1 ? `y${ni + 1}` : `h${li}${ni + 1}`;
      ctx.fillText(label, pos.x, pos.y - NEURON_R - 4);

      // Value display
      if (isActive) {
        ctx.fillStyle = C.text;
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.activation.toFixed(2), pos.x, pos.y + 1);
      }

      // Bias label
      if (isActive) {
        ctx.fillStyle = C.text3;
        ctx.font = '7px JetBrains Mono, monospace';
        ctx.textBaseline = 'top';
        ctx.fillText(`b=${n.bias.toFixed(2)}`, pos.x, pos.y + NEURON_R + 3);
      }
    }
  }

  // Layer labels
  ctx.fillStyle = C.text2;
  ctx.font = '8px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let li = 0; li < layers.length; li++) {
    const mid = layers[li].neurons.length;
    if (mid === 0) continue;
    const first = getNeuronPos(li, 0, layers);
    const last = getNeuronPos(li, layers[li].neurons.length - 1, layers);
    const labelY = Math.max(first.y, last.y) + NEURON_R + 16;
    const x = (first.x + last.x) / 2;
    ctx.fillText(layers[li].label, x, labelY);
  }
}

/* ===== Animation System ===== */
function startForwardAnim() {
  if (!network) return;
  if (animState === 'running') return;
  // Set input values
  const layers = network.layers;
  for (let i = 0; i < Math.min(inputValues.length, layers[0].neurons.length); i++) {
    layers[0].neurons[i].value = inputValues[i];
    layers[0].neurons[i].activation = inputValues[i];
    layers[0].neurons[i].z = inputValues[i];
    layers[0].neurons[i].active = true;
    layers[0].neurons[i].animated = true;
  }
  animState = 'running';
  animPhase = 0; // Starting with input shown, now animate to layer 1
  animProgress = 0;
  animLayerStart = performance.now();
  runBtn.disabled = true;
  updateCompPanel();
  if (animId) cancelAnimationFrame(animId);
  animId = requestAnimationFrame(tickAnim);
}

function tickAnim(now) {
  const layers = network.layers;
  const weights = network.weights;
  const elapsed = now - animLayerStart;
  // Duration for each connection layer animation: 800ms
  const phaseDuration = 800;
  animProgress = Math.min(elapsed / phaseDuration, 1);

  // Current phase animates connections from animPhase (layer index) to animPhase+1
  const srcLayerIdx = animPhase;
  const dstLayerIdx = animPhase + 1;

  // Ease for smoother animation
  const ease = 1 - Math.pow(1 - animProgress, 2);

  // Draw with current animation state
  drawNetwork(srcLayerIdx, ease, srcLayerIdx);

  // If phase complete, activate destination layer neurons
  if (animProgress >= 1) {
    const prev = layers[srcLayerIdx].neurons;
    const curr = layers[dstLayerIdx].neurons;
    const wg = weights[srcLayerIdx];

    // Compute activation for destination layer
    for (let ni = 0; ni < curr.length; ni++) {
      let z = curr[ni].bias || 0;
      for (let pi = 0; pi < prev.length; pi++) {
        const connIdx = pi * curr.length + ni;
        if (connIdx < wg.length) z += prev[pi].activation * wg[connIdx].w;
      }
      curr[ni].z = z;
      curr[ni].activation = activate(z, activation);
      curr[ni].active = true;
      curr[ni].animated = true;
    }

    // Move to next phase
    animPhase++;
    if (animPhase >= layers.length - 1) {
      // Done
      animState = 'done';
      animPhase = -1;
      runBtn.disabled = false;
      updateCompPanel();
      drawNetwork(-1, 1, -1);
      return;
    }
    animLayerStart = now;
    animProgress = 0;
    updateCompPanel();
    animId = requestAnimationFrame(tickAnim);
  } else {
    animId = requestAnimationFrame(tickAnim);
  }
}

/* ===== Computation Panel ===== */
function updateCompPanel() {
  if (!network) return;
  const layers = network.layers;
  let html = '';
  for (let li = 0; li < layers.length; li++) {
    const isActive = li === 0 || layers[li].neurons[0].active;
    html += `<div class="nn-layer-group">`;
    html += `<div class="nn-layer-label">${layers[li].label} Layer</div>`;
    for (let ni = 0; ni < layers[li].neurons.length; ni++) {
      const n = layers[li].neurons[ni];
      const active = n.active || animState === 'done';
      const label = li === 0 ? `x${ni + 1}` : li === layers.length - 1 ? `y${ni + 1}` : `h${li}${ni + 1}`;
      html += `<div class="nn-neuron-detail${active ? ' active' : ''}">`;
      html += `<span class="nd-name">${label}</span> `;
      if (active) {
        html += `<span class="nd-val">${n.activation.toFixed(4)}</span>`;
        if (li > 0) {
          html += `<div class="nd-formula">z = ${n.z.toFixed(4)} &rarr; ${activation}(${n.z.toFixed(4)}) = ${n.activation.toFixed(4)}</div>`;
          html += `<div class="nd-formula">bias = ${(n.bias || 0).toFixed(4)}</div>`;
          // Show weighted sum breakdown
          const prev = layers[li - 1].neurons;
          const wg = network.weights[li - 1];
          let formula = '';
          for (let pi = 0; pi < prev.length; pi++) {
            const connIdx = pi * layers[li].neurons.length + ni;
            if (connIdx < wg.length) {
              const w = wg[connIdx].w;
              formula += `${w >= 0 ? '+' : ''}${w.toFixed(2)}&times;${prev[pi].activation.toFixed(2)}`;
            }
          }
          if (formula) html += `<div class="nd-formula">${formula}</div>`;
        }
      } else {
        html += `<span style="color:var(--text3)">&#8212;</span>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
  }
  compPanel.innerHTML = html;
}

/* ===== Weight Popup ===== */
function handleCanvasClick(e) {
  if (!network || animState === 'running') return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const my = (e.clientY - rect.top) * (canvas.height / rect.height);

  // Find closest connection
  let bestDist = 15, bestConn = null;
  for (const wg of network.weights) {
    for (const c of wg) {
      const from = getNeuronPos(c.fromLayer, c.from, network.layers);
      const to = getNeuronPos(c.toLayer, c.to, network.layers);
      const cx = (from.x + to.x) / 2;
      const cy = (from.y + to.y) / 2;
      const d = Math.hypot(mx - cx, my - cy);
      if (d < bestDist) { bestDist = d; bestConn = c; }
    }
  }

  if (bestConn) {
    selectedConnection = bestConn;
    const fromN = network.layers[bestConn.fromLayer].neurons[bestConn.from];
    const toN = network.layers[bestConn.toLayer].neurons[bestConn.to];
    const pos = getWeightCenter(bestConn, network.layers);
    const screenPos = canvas.getBoundingClientRect();
    weightPopup.style.display = 'block';
    weightPopup.style.left = (screenPos.left + (pos.x / canvas.width) * screenPos.width + 10) + 'px';
    weightPopup.style.top = (screenPos.top + (pos.y / canvas.height) * screenPos.height - 40) + 'px';
    const fromLabel = bestConn.fromLayer === 0 ? `x${bestConn.from + 1}` : `h${bestConn.fromLayer}${bestConn.from + 1}`;
    const toLabel = bestConn.toLayer === network.layers.length - 1 ? `y${bestConn.to + 1}` : `h${bestConn.toLayer}${bestConn.to + 1}`;
    weightPopup.innerHTML = `
      <div class="wp-label">Weight: ${fromLabel} &rarr; ${toLabel}</div>
      <input type="number" step="0.05" id="nnWeightInput" value="${bestConn.w.toFixed(3)}">
    `;
    const inp = document.getElementById('nnWeightInput');
    inp.focus(); inp.select();
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (!isNaN(v) && isFinite(v)) { bestConn.w = v; draw(); }
      weightPopup.style.display = 'none';
    });
    inp.addEventListener('blur', () => { setTimeout(() => { weightPopup.style.display = 'none'; }, 200); });
  } else {
    weightPopup.style.display = 'none';
    selectedConnection = null;
  }
}

/* ===== Activation Curve Drawing ===== */
function drawActCurve() {
  const CW = actCurveCanvas.width, CH = actCurveCanvas.height;
  const pad = { t: 8, r: 6, b: 14, l: 14 };
  const pw = CW - pad.l - pad.r, ph = CH - pad.t - pad.b;
  actCurveCtx.clearRect(0, 0, CW, CH);
  const C = getColors();

  // Grid
  actCurveCtx.strokeStyle = C.border;
  actCurveCtx.globalAlpha = 0.15;
  actCurveCtx.lineWidth = 0.5;
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = pad.t + (1 - pct / 100) * ph;
    actCurveCtx.beginPath(); actCurveCtx.moveTo(pad.l, y); actCurveCtx.lineTo(CW - pad.r, y); actCurveCtx.stroke();
  }
  actCurveCtx.globalAlpha = 1;

  // Axis labels
  actCurveCtx.fillStyle = C.text3;
  actCurveCtx.font = '6px Inter,sans-serif';
  actCurveCtx.textAlign = 'center'; actCurveCtx.textBaseline = 'top';
  actCurveCtx.fillText('z', CW / 2, CH - pad.b + 2);

  // Curve
  const steps = 80;
  actCurveCtx.beginPath();
  actCurveCtx.strokeStyle = C.accent;
  actCurveCtx.lineWidth = 1.8;
  for (let i = 0; i <= steps; i++) {
    const z = -5 + (i / steps) * 10;
    const a = activate(z, activation);
    const x = pad.l + ((z + 5) / 10) * pw;
    const y = pad.t + (1 - a) * ph;
    i === 0 ? actCurveCtx.moveTo(x, y) : actCurveCtx.lineTo(x, y);
  }
  actCurveCtx.stroke();

  actCurveCtx.fillStyle = C.text2;
  actCurveCtx.font = 'bold 7px Inter,sans-serif';
  actCurveCtx.textAlign = 'left'; actCurveCtx.textBaseline = 'top';
  const names = { sigmoid: '\u03C3(z)', relu: 'ReLU(z)', tanh: 'tanh(z)' };
  actCurveCtx.fillText(names[activation] || 'f(z)', pad.l, pad.t + 2);
}

/* ===== Drawing Wrapper ===== */
function draw() {
  if (animState === 'running') return; // animation loop handles drawing
  drawNetwork(-1, 1, -1);
  updateCompPanel();
  drawActCurve();
}

/* ===== Configuration Change ===== */
function rebuildFromConfig() {
  const newArch = [nnConfig.inputVal.textContent, nnConfig.hidden1Val.textContent, nnConfig.hidden2Val.textContent, nnConfig.outputVal.textContent].map(Number);
  arch = newArch;
  network = buildNetwork(arch);
  animState = 'idle'; animPhase = -1;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  runBtn.disabled = false;
  // Sync input sliders
  const nInput = arch[0];
  document.querySelectorAll('.nn-input-row').forEach((el, i) => { el.style.display = i < nInput ? '' : 'none'; });
  draw();
}

function setupCfgButtons() {
  const cfg = [
    { inc: nnConfig.inputInc, dec: nnConfig.inputDec, val: nnConfig.inputVal, min: 2, max: 4 },
    { inc: nnConfig.hidden1Inc, dec: nnConfig.hidden1Dec, val: nnConfig.hidden1Val, min: 0, max: 6 },
    { inc: nnConfig.hidden2Inc, dec: nnConfig.hidden2Dec, val: nnConfig.hidden2Val, min: 0, max: 6 },
    { inc: nnConfig.outputInc, dec: nnConfig.outputDec, val: nnConfig.outputVal, min: 1, max: 3 },
  ];
  for (const c of cfg) {
    c.inc.addEventListener('click', () => {
      const v = Math.min(parseInt(c.val.textContent) + 1, c.max);
      c.val.textContent = v;
      rebuildFromConfig();
    });
    c.dec.addEventListener('click', () => {
      const v = Math.max(parseInt(c.val.textContent) - 1, c.min);
      c.val.textContent = v;
      if (c.min === 0 && v === 0 && cfg.indexOf(c) >= 1 && cfg.indexOf(c) <= 2) {
        // If setting a hidden layer to 0, force the next one to 0 too
        const idx = cfg.indexOf(c);
        if (idx === 1) {
          cfg[2].val.textContent = '0';
          document.getElementById('nnH2Val').textContent = '0';
        }
      }
      rebuildFromConfig();
    });
  }
}

/* ===== Input Sliders ===== */
function setupInputSliders() {
  inputSliders.forEach((slider, i) => {
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      inputValues[i] = v;
      inputLabels[i].textContent = v.toFixed(2);
      if (network && network.layers[0] && network.layers[0].neurons[i]) {
        network.layers[0].neurons[i].value = v;
        network.layers[0].neurons[i].activation = v;
        if (animState === 'idle' || animState === 'done') {
          // If forward pass was done, reset all for fresh run
          if (animState === 'done') {
            clearNeuronActive();
            animState = 'idle';
            runBtn.disabled = false;
          }
          draw();
        }
      }
    });
  });
}

/* ===== Activation Change ===== */
actSelect.addEventListener('change', () => {
  activation = actSelect.value;
  if (animState === 'done') {
    // Recompute with new activation
    const layers = network.layers;
    for (let li = 1; li < layers.length; li++) {
      for (const n of layers[li].neurons) {
        n.activation = activate(n.z, activation);
      }
    }
  }
  drawActCurve();
  if (animState === 'done') updateCompPanel();
  if (animState === 'idle') draw();
});

/* ===== Event Handlers ===== */
runBtn.addEventListener('click', startForwardAnim);

randomBtn.addEventListener('click', () => {
  randomizeNetwork();
  draw();
  weightPopup.style.display = 'none';
});

resetBtn.addEventListener('click', () => {
  resetNetwork();
  draw();
  weightPopup.style.display = 'none';
});

canvas.addEventListener('click', handleCanvasClick);

canvas.addEventListener('mousemove', (e) => {
  if (!network) return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const my = (e.clientY - rect.top) * (canvas.height / rect.height);
  let found = null, bestDist = 12;
  for (const wg of network.weights) {
    for (const c of wg) {
      const from = getNeuronPos(c.fromLayer, c.from, network.layers);
      const to = getNeuronPos(c.toLayer, c.to, network.layers);
      const cx = (from.x + to.x) / 2;
      const cy = (from.y + to.y) / 2;
      const d = Math.hypot(mx - cx, my - cy);
      if (d < bestDist) { bestDist = d; found = c; }
    }
  }
  if (found !== hoveredConnection) {
    hoveredConnection = found;
    canvas.style.cursor = found ? 'pointer' : 'default';
    if (animState !== 'running') drawNetwork(-1, 1, -1);
  }
});

canvas.addEventListener('mouseleave', () => {
  hoveredConnection = null;
  canvas.style.cursor = 'default';
  if (animState !== 'running') drawNetwork(-1, 1, -1);
});

/* ===== Theme ===== */
function applyTheme(t) {
  currentTheme = t;
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('nn-theme', t);
  themeToggle.textContent = t === 'dark' ? '\u2600' : '\u263E';
  drawActCurve();
  if (animState !== 'running') draw();
}

themeToggle.addEventListener('click', () => {
  applyTheme((document.documentElement.getAttribute('data-theme') || 'dark') === 'dark' ? 'light' : 'dark');
});

/* ===== Resize Canvas ===== */
function resizeCanvas() {
  const wrapper = canvas.parentElement;
  if (!wrapper) return;
  const maxW = wrapper.clientWidth - 2;
  if (maxW < 420) {
    canvas.style.width = maxW + 'px';
    canvas.style.height = ((maxW / 500) * 400) + 'px';
  } else {
    canvas.style.width = '';
    canvas.style.height = '';
  }
}

window.addEventListener('resize', resizeCanvas);

/* ===== Init ===== */
function init() {
  const saved = localStorage.getItem('nn-theme') || 'dark';
  setupCfgButtons();
  setupInputSliders();
  arch = [3, 4, 0, 2];
  network = buildNetwork(arch);
  applyTheme(saved);
  // Show correct input sliders
  document.querySelectorAll('.nn-input-row').forEach((el, i) => { el.style.display = i < 3 ? '' : 'none'; });
  draw();
  resizeCanvas();
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') weightPopup.style.display = 'none';
  });
  // Click outside popup
  document.addEventListener('mousedown', (e) => {
    if (weightPopup.style.display === 'block' && !weightPopup.contains(e.target) && e.target !== canvas) {
      weightPopup.style.display = 'none';
    }
  });
}

document.addEventListener('DOMContentLoaded', init);
