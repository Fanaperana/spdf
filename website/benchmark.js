// Render the accuracy and spatial benchmark tables. Source data is
// copied into /website at build time by .github/workflows/pages.yml.

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const ms = (x) => `${x} ms`;

function mean(rows, pick, key) {
  const vals = rows
    .map((r) => pick(r))
    .filter(Boolean)
    .map((e) => e[key]);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

async function loadJSON(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function renderAccuracy(host, rows) {
  const tableRows = rows.flatMap((r) => {
    const engines = [["spdf", r.spdf]];
    if (r.lite) engines.push(["liteparse", r.lite]);
    const winner = engines.slice().sort((a, b) => b[1].f1 - a[1].f1)[0][0];
    return engines.map(([name, e]) => {
      const cls = name === winner ? "winner" : "";
      return `
        <tr class="${cls}">
          <td>${r.fixture}</td>
          <td class="engine">${name}</td>
          <td class="num">${ms(e.ms)}</td>
          <td class="num">${e.tokens}</td>
          <td class="num">${pct(e.recall)}</td>
          <td class="num">${pct(e.precision)}</td>
          <td class="num">${pct(e.f1)}</td>
        </tr>
      `;
    });
  });

  const spdfF1 = mean(rows, (r) => r.spdf, "f1");
  const spdfMs = mean(rows, (r) => r.spdf, "ms");
  const liteF1 = mean(rows, (r) => r.lite, "f1");
  const liteMs = mean(rows, (r) => r.lite, "ms");

  const liteSummary = rows.some((r) => r.lite)
    ? `
      <div class="stat">
        <div class="label">liteparse mean F1</div>
        <div class="value">${pct(liteF1)}</div>
      </div>
      <div class="stat">
        <div class="label">liteparse mean wall-clock</div>
        <div class="value">${liteMs.toFixed(0)} ms</div>
      </div>
    `
    : "";

  host.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>fixture</th>
          <th>engine</th>
          <th class="num">wall-clock</th>
          <th class="num">tokens</th>
          <th class="num">recall</th>
          <th class="num">precision</th>
          <th class="num">F1</th>
        </tr>
      </thead>
      <tbody>${tableRows.join("")}</tbody>
    </table>
    <div class="summary-card">
      <div class="stat">
        <div class="label">spdf mean F1</div>
        <div class="value">${pct(spdfF1)}</div>
      </div>
      <div class="stat">
        <div class="label">spdf mean wall-clock</div>
        <div class="value">${spdfMs.toFixed(0)} ms</div>
      </div>
      ${liteSummary}
    </div>
  `;
  host.removeAttribute("aria-busy");
}

function renderSpatial(host, rows) {
  const tableRows = rows.flatMap((r) => {
    const engines = [["spdf", r.spdf]];
    if (r.lite) engines.push(["liteparse", r.lite]);
    const winner = engines
      .slice()
      .sort((a, b) => b[1].mean_iou - a[1].mean_iou)[0][0];
    return engines.map(([name, s]) => {
      const cls = name === winner ? "winner" : "";
      return `
        <tr class="${cls}">
          <td>${r.fixture}</td>
          <td class="engine">${name}</td>
          <td class="num">${s.matches}</td>
          <td class="num">${s.mean_iou.toFixed(3)}</td>
          <td class="num">${pct(s.iou_ge_threshold_rate)}</td>
          <td class="num">${s.mean_centroid_err_pt.toFixed(2)} pt</td>
        </tr>
      `;
    });
  });

  const spdfIou = mean(rows, (r) => r.spdf, "mean_iou");
  const spdfGe = mean(rows, (r) => r.spdf, "iou_ge_threshold_rate");
  const spdfErr = mean(rows, (r) => r.spdf, "mean_centroid_err_pt");
  const liteIou = mean(rows, (r) => r.lite, "mean_iou");
  const liteGe = mean(rows, (r) => r.lite, "iou_ge_threshold_rate");
  const liteErr = mean(rows, (r) => r.lite, "mean_centroid_err_pt");

  const liteSummary = rows.some((r) => r.lite)
    ? `
      <div class="stat">
        <div class="label">liteparse mean IoU</div>
        <div class="value">${liteIou.toFixed(3)}</div>
      </div>
      <div class="stat">
        <div class="label">liteparse IoU ≥ 0.5</div>
        <div class="value">${pct(liteGe)}</div>
      </div>
      <div class="stat">
        <div class="label">liteparse centroid err</div>
        <div class="value">${liteErr.toFixed(2)} pt</div>
      </div>
    `
    : "";

  host.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>fixture</th>
          <th>engine</th>
          <th class="num">matched</th>
          <th class="num">mean IoU</th>
          <th class="num">IoU ≥ 0.5</th>
          <th class="num">centroid err</th>
        </tr>
      </thead>
      <tbody>${tableRows.join("")}</tbody>
    </table>
    <div class="summary-card">
      <div class="stat">
        <div class="label">spdf mean IoU</div>
        <div class="value">${spdfIou.toFixed(3)}</div>
      </div>
      <div class="stat">
        <div class="label">spdf IoU ≥ 0.5</div>
        <div class="value">${pct(spdfGe)}</div>
      </div>
      <div class="stat">
        <div class="label">spdf centroid err</div>
        <div class="value">${spdfErr.toFixed(2)} pt</div>
      </div>
      ${liteSummary}
    </div>
  `;
  host.removeAttribute("aria-busy");
}

(async () => {
  const accuracyHost = document.getElementById("benchmark-block");
  const spatialHost = document.getElementById("spatial-block");

  if (accuracyHost) {
    try {
      const data = await loadJSON("./benchmark.json");
      if ((data.rows || []).length === 0) throw new Error("no rows");
      renderAccuracy(accuracyHost, data.rows);
    } catch (err) {
      accuracyHost.innerHTML = `<p class="muted">Accuracy results unavailable: ${err.message}.</p>`;
      accuracyHost.removeAttribute("aria-busy");
    }
  }

  if (spatialHost) {
    try {
      const data = await loadJSON("./spatial.json");
      if ((data.rows || []).length === 0) throw new Error("no rows");
      renderSpatial(spatialHost, data.rows);
    } catch (err) {
      spatialHost.innerHTML = `<p class="muted">Spatial results unavailable: ${err.message}.</p>`;
      spatialHost.removeAttribute("aria-busy");
    }
  }
})();
