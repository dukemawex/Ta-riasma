async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return await res.json();
}

function kpi(label, value) {
  const d = document.createElement('div');
  d.className = 'kpi';
  d.innerHTML = `<span class="label">${label}</span><b>${value}</b>`;
  return d;
}

function drawLineChart(canvasId, labels, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !labels.length || !datasets.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 20, right: 16, bottom: 32, left: 42 };

  const allValues = datasets.flatMap((ds) => ds.values).filter((v) => Number.isFinite(v));
  if (!allValues.length) return;
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ((h - pad.top - pad.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  const xFor = (i) => pad.left + ((w - pad.left - pad.right) * i) / Math.max(labels.length - 1, 1);
  const yFor = (v) => h - pad.bottom - ((v - min) / range) * (h - pad.top - pad.bottom);

  datasets.forEach((ds) => {
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ds.values.forEach((v, i) => {
      const x = xFor(i);
      const y = yFor(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.fillStyle = '#334155';
  ctx.font = '12px Inter, sans-serif';
  labels.forEach((label, i) => {
    const x = xFor(i);
    ctx.fillText(String(label), x - 18, h - 10);
  });
}

function drawBarChart(canvasId, items) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !items.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 20, right: 16, bottom: 40, left: 40 };

  const max = Math.max(...items.map((i) => i.value), 1);
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;
  const barW = chartW / items.length * 0.64;
  const gap = chartW / items.length * 0.36;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  items.forEach((item, idx) => {
    const x = pad.left + idx * (barW + gap) + gap / 2;
    const barH = (item.value / max) * chartH;
    const y = pad.top + chartH - barH;
    ctx.fillStyle = item.color || '#2563eb';
    ctx.fillRect(x, y, barW, barH);
    ctx.fillStyle = '#334155';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText(item.label, x, h - 14);
  });
}

(async function main() {
  const multiStatus = document.getElementById('multi-status');
  const dupStatus = document.getElementById('dup-status');
  const summaryStatus = document.getElementById('summary-status');
  const summary = document.getElementById('summary-kpis');
  let multiData = null;
  let dupData = null;

  try {
    const multi = await loadJson('../results/multilingual_results.json');
    multiData = multi;
    const m = multi.metrics || {};
    const box = document.getElementById('multi-metrics');
    box.appendChild(kpi('Positive mean', (m.positive_mean ?? 0).toFixed(4)));
    box.appendChild(kpi('Negative mean', (m.negative_mean ?? 0).toFixed(4)));
    box.appendChild(kpi('Separation gap', (m.separation_gap ?? 0).toFixed(4)));
    box.appendChild(kpi('Positives < 0.85', String(m.positive_failures_below_0_85 ?? 0)));
    box.appendChild(kpi('Negatives >= 0.85', String(m.negative_false_positives_above_0_85 ?? 0)));
    drawLineChart(
      'multi-chart',
      ['Positive', 'Negative', 'Gap'],
      [{
        color: '#2563eb',
        values: [
          Number(m.positive_mean ?? 0),
          Number(m.negative_mean ?? 0),
          Number(m.separation_gap ?? 0)
        ]
      }]
    );
    const health = Number(m.separation_gap ?? 0) >= 0.2 ? 'good' : 'risk';
    document.getElementById('multi-insight').innerHTML = `Multilingual stability signal is <b class="${health}">${health === 'good' ? 'strong' : 'weak'}</b> based on current separation gap.`;
    multiStatus.textContent = 'Loaded multilingual results successfully.';
  } catch (e) {
    multiStatus.textContent = `Multilingual results not found yet (${e.message}). Run eval_multilingual.py first.`;
  }

  try {
    const dup = await loadJson('../results/duplicate_results.json');
    dupData = dup;
    const dist = dup.score_distribution || {};
    const rec = dup.recommended_threshold || {};

    document.getElementById('dup-threshold').textContent =
      `Recommended threshold: DUPLICATE_THRESHOLD=${(rec.threshold ?? 0).toFixed(2)} (Combined F1 ${(rec.combined_f1 ?? 0).toFixed(4)})`;

    const dbox = document.getElementById('dup-distribution');
    for (const [name, stats] of Object.entries(dist)) {
      dbox.appendChild(kpi(`${name} mean`, (stats.mean ?? 0).toFixed(4)));
      dbox.appendChild(kpi(`${name} std`, (stats.std ?? 0).toFixed(4)));
    }

    const tbody = document.querySelector('#threshold-table tbody');
    const thresholdRows = dup.threshold_metrics || [];
    thresholdRows.forEach((row) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${(row.threshold ?? 0).toFixed(2)}</td>
        <td>${(row.rule_f1 ?? 0).toFixed(4)}</td>
        <td>${(row.claude_f1 ?? 0).toFixed(4)}</td>
        <td>${(row.combined_f1 ?? 0).toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });
    drawLineChart(
      'threshold-chart',
      thresholdRows.map((r) => (r.threshold ?? 0).toFixed(2)),
      [
        { color: '#16a34a', values: thresholdRows.map((r) => Number(r.rule_f1 ?? 0)) },
        { color: '#9333ea', values: thresholdRows.map((r) => Number(r.claude_f1 ?? 0)) },
        { color: '#2563eb', values: thresholdRows.map((r) => Number(r.combined_f1 ?? 0)) }
      ]
    );

    drawBarChart(
      'distribution-chart',
      Object.entries(dist).map(([name, stats], idx) => ({
        label: name,
        value: Number(stats.mean ?? 0),
        color: ['#2563eb', '#16a34a', '#9333ea', '#ea580c'][idx % 4]
      }))
    );
    const qualityClass = Number(rec.combined_f1 ?? 0) >= 0.8 ? 'good' : 'risk';
    document.getElementById('dup-insight').innerHTML = `Duplicate stack fit is <b class="${qualityClass}">${qualityClass === 'good' ? 'production-ready' : 'needs tuning'}</b> at current threshold.`;

    dupStatus.textContent = 'Loaded duplicate evaluation results successfully.';
  } catch (e) {
    dupStatus.textContent = `Duplicate results not found yet (${e.message}). Run eval_duplicates.py first.`;
  }

  const m = (multiData && multiData.metrics) || {};
  const r = (dupData && dupData.recommended_threshold) || {};
  summary.appendChild(kpi('Product Trust Score', ((Number(m.separation_gap ?? 0) * 100).toFixed(1)) + '%'));
  summary.appendChild(kpi('Duplicate Ops F1', (Number(r.combined_f1 ?? 0)).toFixed(4)));
  summary.appendChild(kpi('Recommended Threshold', (Number(r.threshold ?? 0)).toFixed(2)));
  summary.appendChild(kpi('Pipeline Health', multiData && dupData ? 'All signals loaded' : 'Partial signals'));
  summaryStatus.textContent = multiData && dupData
    ? 'Executive summary generated from multilingual + duplicate evaluations.'
    : 'Summary loaded with available data only; run both evaluations for complete analytics.';
})();
