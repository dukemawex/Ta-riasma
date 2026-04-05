async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return await res.json();
}

function kpi(label, value) {
  const d = document.createElement('div');
  d.className = 'kpi';
  d.innerHTML = `${label}<b>${value}</b>`;
  return d;
}

(async function main() {
  const multiStatus = document.getElementById('multi-status');
  const dupStatus = document.getElementById('dup-status');

  try {
    const multi = await loadJson('../results/multilingual_results.json');
    const m = multi.metrics || {};
    const box = document.getElementById('multi-metrics');
    box.appendChild(kpi('Positive mean', (m.positive_mean ?? 0).toFixed(4)));
    box.appendChild(kpi('Negative mean', (m.negative_mean ?? 0).toFixed(4)));
    box.appendChild(kpi('Separation gap', (m.separation_gap ?? 0).toFixed(4)));
    box.appendChild(kpi('Positives < 0.85', String(m.positive_failures_below_0_85 ?? 0)));
    box.appendChild(kpi('Negatives >= 0.85', String(m.negative_false_positives_above_0_85 ?? 0)));
    multiStatus.textContent = 'Loaded multilingual results successfully.';
  } catch (e) {
    multiStatus.textContent = `Multilingual results not found yet (${e.message}). Run eval_multilingual.py first.`;
  }

  try {
    const dup = await loadJson('../results/duplicate_results.json');
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
    (dup.threshold_metrics || []).forEach((row) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${(row.threshold ?? 0).toFixed(2)}</td>
        <td>${(row.rule_f1 ?? 0).toFixed(4)}</td>
        <td>${(row.claude_f1 ?? 0).toFixed(4)}</td>
        <td>${(row.combined_f1 ?? 0).toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });

    dupStatus.textContent = 'Loaded duplicate evaluation results successfully.';
  } catch (e) {
    dupStatus.textContent = `Duplicate results not found yet (${e.message}). Run eval_duplicates.py first.`;
  }
})();
