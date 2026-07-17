const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const kSlider = document.getElementById('kSlider');
const kValue = document.getElementById('kValue');
const randomBtn = document.getElementById('randomBtn');
const clearQueryBtn = document.getElementById('clearQueryBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const pointCount = document.getElementById('pointCount');
const prediction = document.getElementById('prediction');

const W = canvas.width;
const H = canvas.height;
const PAD = 40;

let classA = [];
let classB = [];
let queryPoint = null;

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function dist(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function generateRandomData(countA, countB) {
  classA = [];
  classB = [];
  for (let i = 0; i < countA; i++) {
    classA.push({ x: rand(PAD, W - PAD), y: rand(PAD, H - PAD) });
  }
  for (let i = 0; i < countB; i++) {
    classB.push({ x: rand(PAD, W - PAD), y: rand(PAD, H - PAD) });
  }
}

function knnPredict(q, k) {
  const neighbors = [];
  for (const p of classA) {
    neighbors.push({ point: p, dist: dist(q, p), cls: 'A' });
  }
  for (const p of classB) {
    neighbors.push({ point: p, dist: dist(q, p), cls: 'B' });
  }
  neighbors.sort((a, b) => a.dist - b.dist);
  const nearest = neighbors.slice(0, k);
  let countA = 0, countB = 0;
  for (const n of nearest) {
    if (n.cls === 'A') countA++;
    else countB++;
  }
  let predicted;
  if (countA > countB) predicted = 'A';
  else if (countB > countA) predicted = 'B';
  else predicted = 'tie';
  return { predicted, nearest, countA, countB };
}

function drawArrow(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 6;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
  ctx.strokeStyle = '#999';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, W, H);

  for (const p of classA) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#e74c3c';
    ctx.fill();
    ctx.strokeStyle = '#c0392b';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  for (const p of classB) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#3498db';
    ctx.fill();
    ctx.strokeStyle = '#2980b9';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (queryPoint) {
    const k = parseInt(kSlider.value);
    const result = knnPredict(queryPoint, k);

    for (const n of result.nearest) {
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      ctx.moveTo(queryPoint.x, queryPoint.y);
      ctx.lineTo(n.point.x, n.point.y);
      const color = n.cls === 'A' ? '#e74c3c' : '#3498db';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.setLineDash([]);

      const midX = (queryPoint.x + n.point.x) / 2;
      const midY = (queryPoint.y + n.point.y) / 2;
      drawArrow(queryPoint.x, queryPoint.y, n.point.x, n.point.y);
    }

    for (const n of result.nearest) {
      ctx.beginPath();
      ctx.arc(n.point.x, n.point.y, 11, 0, Math.PI * 2);
      const color = n.cls === 'A' ? '#e74c3c' : '#3498db';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.moveTo(queryPoint.x, queryPoint.y - 14);
    ctx.lineTo(queryPoint.x + 10, queryPoint.y);
    ctx.lineTo(queryPoint.x, queryPoint.y + 14);
    ctx.lineTo(queryPoint.x - 10, queryPoint.y);
    ctx.closePath();
    ctx.fillStyle = '#2ecc71';
    ctx.fill();
    ctx.strokeStyle = '#27ae60';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.fillStyle = 'white';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    let label;
    if (result.predicted === 'A') label = 'A';
    else if (result.predicted === 'B') label = 'B';
    else label = '?';
    ctx.fillText(label, queryPoint.x, queryPoint.y);

    const total = classA.length + classB.length;
    pointCount.textContent = `Class A: ${classA.length}  |  Class B: ${classB.length}  |  Total: ${total}`;

    let predText;
    prediction.className = '';
    if (result.predicted === 'A') {
      predText = `Prediction: Class A (${result.countA}/${k} neighbors)`;
      prediction.classList.add('class-a-pred');
    } else if (result.predicted === 'B') {
      predText = `Prediction: Class B (${result.countB}/${k} neighbors)`;
      prediction.classList.add('class-b-pred');
    } else {
      predText = `Tie (${result.countA}-${result.countB})`;
      prediction.classList.add('tie');
    }
    prediction.textContent = predText;
  } else {
    const total = classA.length + classB.length;
    pointCount.textContent = `Class A: ${classA.length}  |  Class B: ${classB.length}  |  Total: ${total}`;
    prediction.textContent = 'Click on the canvas to place a query point';
    prediction.className = '';
  }
}

function handleCanvasClick(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  queryPoint = { x, y };
  draw();
}

kSlider.addEventListener('input', () => {
  kValue.textContent = kSlider.value;
  if (queryPoint) draw();
});

randomBtn.addEventListener('click', () => {
  generateRandomData(10, 10);
  queryPoint = null;
  draw();
});

clearQueryBtn.addEventListener('click', () => {
  queryPoint = null;
  draw();
});

clearAllBtn.addEventListener('click', () => {
  classA = [];
  classB = [];
  queryPoint = null;
  draw();
});

canvas.addEventListener('click', handleCanvasClick);

generateRandomData(12, 12);
draw();
