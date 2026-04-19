// Render the benchmark table from ./benchmark.json (copied into /website at
// build time by .github/workflows/pages.yml).
(async () => {
  const host = document.getElementById("benchmark-block");
  if (!host) return;

  let data;
  try {
    const resp = await fetch("./benchmark.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    host.innerHTML = `<p class="muted">Benchmark results unavailable: ${err.message}. Run <code>./benchmark/run.sh</code> locally to reproduce.</p>`;
    host.removeAttribute("aria-busy");
    return;
  }

  const rows = data.rows || [];
  if (rows.length === 0) {
    host.innerHTML = `<p class="muted">No benchmark rows available.</p>`;
    host.removeAttribute("aria-busy");
    return;
  }

  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const ms = (x) => `${x} ms`;

  // Per-fixture table. We emit two rows per fixture (spdf, liteparse) and
  // class the winning row so CSS can highlight it.
  const tableRows = rows.flatMap((r) => {
    const engines = [["spdf", r.spdf]];
    if (r.lite) engines.push(["liteparse", r.lite]);
    const winner = engines
      .slice()
      .sort((a, b) => b[1].f1 - a[1].f1)[0][0];
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

  const mean = (key, engine) => {
    const vals = rows
      .map((r) => (engine === "spdf" ? r.spdf : r.lite))
      .filter(Boolean)
      .map((e) => e[key]);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };

  const spdfF1 = mean("f1", "spdf");
  const spdfMs = mean("ms", "spdf");
  const liteF1 = mean("f1", "lite");
  const liteMs = mean("ms", "lite");

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
      ${
        rows.some((r) => r.lite)
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
          : ""
      }
    </div>
  `;
  host.removeAttribute("aria-busy");
})();
