// spdf project page. Vanilla JS, no dependencies.
// Ports of Magic UI components from 21st.dev (MIT): Number Ticker (spring
// count-up), Terminal (typed lines), Animated Grid Pattern, Magic Card
// (pointer spotlight). All numbers come from ./results.json, which is
// generated from bench/results/latest.json by website/build_data.py.
(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // ---- theme toggle --------------------------------------------------
  var root = document.documentElement;
  $("#theme-toggle").addEventListener("click", function () {
    var cur = root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    var next = cur === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("spdf-theme", next); } catch (e) {}
  });

  // ---- copy buttons --------------------------------------------------
  $$(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var text = b.parentNode.querySelector("pre").innerText;
      navigator.clipboard && navigator.clipboard.writeText(text).then(function () {
        b.textContent = "Copied"; b.classList.add("done");
        setTimeout(function () { b.textContent = "Copy"; b.classList.remove("done"); }, 1400);
      });
    });
  });

  // ---- Animated Grid Pattern (Magic UI) ------------------------------
  (function grid() {
    var svg = $("#agp"); if (!svg) return;
    var NS = "http://www.w3.org/2000/svg", W = 40, H = 40, N = 34;
    var r = svg.getBoundingClientRect();
    svg.innerHTML = '<defs><pattern id="agp-p" width="' + W + '" height="' + H + '" patternUnits="userSpaceOnUse" x="-1" y="-1">' +
      '<path d="M.5 ' + H + 'V.5H' + W + '" fill="none"/></pattern></defs><rect width="100%" height="100%" fill="url(#agp-p)" stroke-width="0"/>';
    var cols = Math.max(1, Math.floor(r.width / W)), rows = Math.max(1, Math.floor(r.height / H));
    function place(el) {
      el.setAttribute("x", Math.floor(Math.random() * cols) * W);
      el.setAttribute("y", Math.floor(Math.random() * rows) * H);
    }
    for (var i = 0; i < N; i++) {
      var sq = document.createElementNS(NS, "rect");
      sq.setAttribute("class", "sq"); sq.setAttribute("width", W - 1); sq.setAttribute("height", H - 1);
      sq.style.setProperty("--delay", (i * 0.1).toFixed(1) + "s");
      sq.style.setProperty("--d", (3 + Math.random() * 2).toFixed(2) + "s");
      place(sq);
      (function (el) {
        var n = 0;
        el.addEventListener("animationiteration", function () { if (++n % 2 === 0) place(el); });
      })(sq);
      svg.appendChild(sq);
    }
  })();

  // ---- Terminal (Magic UI): type commands, then reveal output --------
  (function terminal() {
    var term = $(".term"), body = $("#term"); if (!term || !body || reduce) return;
    var lines = $$(".t-line", body);
    var texts = lines.map(function (l) { return l.innerHTML; });
    term.classList.add("typing");
    var caret = document.createElement("span"); caret.className = "caret";
    var i = 0;
    function next() {
      if (i >= lines.length) { lines[lines.length - 1].appendChild(caret); return; }
      var el = lines[i], isCmd = !el.classList.contains("t-out");
      el.classList.add("shown");
      if (!isCmd) { i++; setTimeout(next, 260); return; }
      var full = el.textContent.replace(/^\$ /, ""), k = 0;
      el.innerHTML = '<span class="p">$</span> ';
      var span = document.createElement("span"); el.appendChild(span); el.appendChild(caret);
      (function type() {
        if (k <= full.length) { span.textContent = full.slice(0, k++); setTimeout(type, 22 + Math.random() * 26); }
        else { el.innerHTML = texts[i]; i++; setTimeout(next, 380); }
      })();
    }
    setTimeout(next, 500);
  })();

  // ---- Number Ticker (Magic UI): spring from 0 to value when in view -
  function tick(el, value, dec) {
    var fmt = function (v) { return Intl.NumberFormat("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }).format(Number(v.toFixed(dec))); };
    if (reduce || !("IntersectionObserver" in window)) { el.textContent = fmt(value); return; }
    el.textContent = fmt(0);
    var io = new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting) return; io.disconnect();
      // critically damped-ish spring, same constants as Magic UI (stiffness 100, damping 60, mass 1)
      var x = 0, v = 0, k = 100, c = 60, last = performance.now();
      (function step(now) {
        var dt = Math.min(0.032, (now - last) / 1000); last = now;
        for (var s = 0; s < 4; s++) { var a = -k * (x - value) - c * v; v += a * dt / 4; x += v * dt / 4; }
        el.textContent = fmt(x);
        if (Math.abs(x - value) > Math.pow(10, -dec - 1) * 5 || Math.abs(v) > 0.001) requestAnimationFrame(step);
        else el.textContent = fmt(value);
      })(last);
    });
    io.observe(el);
  }

  // ---- Magic Card spotlight ------------------------------------------
  function magic(card) {
    card.addEventListener("pointermove", function (e) {
      var r = card.getBoundingClientRect();
      card.style.setProperty("--mx", e.clientX - r.left + "px");
      card.style.setProperty("--my", e.clientY - r.top + "px");
    });
    card.addEventListener("pointerleave", function () {
      card.style.setProperty("--mx", "-300px"); card.style.setProperty("--my", "-300px");
    });
  }

  // ---- data ----------------------------------------------------------
  var f2 = function (v, d) { return Number(v).toFixed(d); };
  var secs = function (v) { return v < 1 ? f2(v, 3) + " s" : f2(v, 2) + " s"; };
  var mb = function (v) { return f2(v, 1) + " MB"; };
  var set = function (k, txt) { $$('[data-k="' + k + '"]').forEach(function (e) { e.textContent = txt; }); };

  var CARDS = [
    { title: "Accuracy", unit: "score", note: "higher is better", metrics: [
      ["odl.overall", "Overall score, 200 real PDFs"], ["odl.nid", "Reading order (NID)"],
      ["odl.teds", "Tables (TEDS)"], ["odl.mhs", "Heading hierarchy (MHS)"], ["edge.f1_mean", "Edge cases, token F1"]] },
    { title: "Speed", unit: "s", note: "lower is better", metrics: [
      ["odl.sec_per_doc", "Mean time per PDF"], ["odl.sec_p95", "p95 time per PDF"],
      ["edge.total_sec", "All edge cases, total"], ["edge.huge_sec", "400-page PDF"]] },
    { title: "Memory", unit: "MB", note: "peak RSS, lower is better", metrics: [
      ["odl.peak_rss_mb", "Peak, 200 real PDFs"], ["odl.rss_mean_mb", "Mean peak per PDF"],
      ["edge.max_rss_mb", "Peak, edge cases"], ["edge.huge_rss_mb", "400-page PDF"]] },
    { title: "OCR", unit: "mixed", note: "scanned edge cases", metrics: [
      ["edge.ocr_nid_mean", "Text accuracy on scans (NID)"], ["edge.ocr_sec_mean", "Mean time per scan"],
      ["edge.ocr_sec_max", "Slowest scan"], ["edge.ocr_rss_max_mb", "Peak memory on scans"]] }
  ];

  function fmtVal(name, v) {
    if (v === null || v === undefined) return "–";
    if (/rss|_mb$/.test(name)) return mb(v);
    if (/sec/.test(name)) return secs(v);
    if (name === "odl.failures") return String(v);
    return f2(v, 3);
  }
  function deltaText(r) {
    if (r.status === "tie") return "tie";
    var s = r.spdf, l = r.liteparse;
    if (r.better === "higher") return (s - l >= 0 ? "+" : "") + f2(s - l, 3);
    if (s < l) return (l / s >= 1.5 ? f2(l / s, 1) + "× " + (/sec/.test(r.metric) ? "faster" : "less") : Math.round((1 - s / l) * 100) + "% " + (/sec/.test(r.metric) ? "faster" : "less"));
    return "+" + Math.round((s / l - 1) * 100) + "% " + (/sec/.test(r.metric) ? "slower" : "more");
  }

  function render(d) {
    var rows = {}; d.rows.forEach(function (r) { rows[r.metric] = r; });
    var S = d.engines.spdf, L = d.engines.liteparse, T = d.tally;

    // tally + hero
    set("tally-pill", T.wins + " wins · " + T.ties + " ties · " + T.losses + (T.losses === 1 ? " loss" : " losses") + " vs LiteParse " + d.versions.liteparse.replace(/^lit\s*/, ""));
    ["total", "total2", "total3"].forEach(function (k) { set(k, T.total); });
    set("cases", S.edge.case_count); set("cases2", S.edge.case_count); set("repeats", d.repeats);
    var bar = $(".tally-bar");
    bar.setAttribute("aria-label", T.wins + " wins, " + T.ties + " ties, " + T.losses + " losses out of " + T.total + " metrics");
    bar.innerHTML = '<span class="seg win" style="flex:' + T.wins + '"></span><span class="seg tie" style="flex:' + T.ties + '"></span>' +
      (T.losses ? '<span class="seg loss" style="flex:' + T.losses + '"></span>' : "");
    var tick1 = function (k, v, dec) { $$('[data-k="' + k + '"]').forEach(function (e) { tick(e, v, dec); }); };
    tick1("wins", T.wins, 0); tick1("ties", T.ties, 0); tick1("losses", T.losses, 0);
    tick1("speedup", L.odl.sec_per_doc / S.odl.sec_per_doc, 1);
    tick1("memsave", (1 - S.odl.peak_rss_mb / L.odl.peak_rss_mb) * 100, 0);
    tick1("ocrnid", S.edge.ocr_nid_mean, 3);
    set("sec-pair", f2(S.odl.sec_per_doc, 3) + " s vs " + f2(L.odl.sec_per_doc, 3) + " s");
    set("rss-pair", mb(S.odl.peak_rss_mb) + " vs " + mb(L.odl.peak_rss_mb));
    set("ocr-pair", "vs " + f2(L.edge.ocr_nid_mean, 3) + " (NID)");
    set("ocr-nid-s", f2(S.edge.ocr_nid_mean, 3)); set("ocr-nid-l", f2(L.edge.ocr_nid_mean, 3));
    set("ocr-sec-s", secs(S.edge.ocr_sec_mean)); set("ocr-sec-l", secs(L.edge.ocr_sec_mean));
    set("ocr-rss-s", Math.round(S.edge.ocr_rss_mean_mb) + " MB"); set("ocr-rss-l", Math.round(L.edge.ocr_rss_mean_mb) + " MB");
    var date = d.timestamp.slice(0, 10);
    set("run-meta", "Accepted run " + date + " · " + d.versions.spdf + " (main) · " + d.versions.liteparse + " · median of " + d.repeats + " · one Windows machine");

    // loss card (first loss, if any)
    var losses = d.rows.filter(function (r) { return r.status === "loss"; });
    var lc = $(".loss-card");
    if (!losses.length) { lc.hidden = true; }
    else if (losses[0].metric === "edge.case.scanned_chinese.sec") {
      var lr = losses[0], acc = rows["edge.case.scanned_chinese"], mem = rows["edge.case.scanned_chinese.rss_mb"];
      set("loss-sec", f2(lr.spdf, 3) + " s vs " + f2(lr.liteparse, 3) + " s");
      set("loss-delta", Math.round((lr.spdf - lr.liteparse) * 1000) + " ms (" + Math.round((lr.spdf / lr.liteparse - 1) * 100) + "%)");
      set("loss-acc-s", f2(acc.spdf, 2)); set("loss-acc-l", f2(acc.liteparse, 2));
      set("loss-mem", mb(mem.spdf) + " vs " + mb(mem.liteparse));
      if (losses.length > 1) lc.querySelector("p").insertAdjacentHTML("beforeend", " <strong>" + (losses.length - 1) + " more loss(es) are listed in the table below.</strong>");
    } else {
      lc.querySelector(".badge").textContent = losses.length === 1 ? "The one loss" : losses.length + " losses";
      lc.querySelector("h3").textContent = losses.map(function (r) { return r.metric + ": " + fmtVal(r.metric, r.spdf) + " vs " + fmtVal(r.metric, r.liteparse); }).join("; ");
      lc.querySelector("p").textContent = "Every metric where spdf is currently behind LiteParse. See the full table below.";
    }

    // metric cards
    var host = $("#metric-cards");
    host.innerHTML = CARDS.map(function (c) {
      var ms = c.metrics.filter(function (m) { return rows[m[0]]; });
      var w = ms.filter(function (m) { return rows[m[0]].status === "win"; }).length;
      return '<div class="magic-card"><div class="magic-card-inner"><div class="card-head"><h3>' + c.title + '</h3><span class="count">' +
        w + "/" + ms.length + " won · " + c.note + "</span></div>" +
        '<div class="legend"><span class="s"><i></i>spdf</span><span class="l"><i></i>LiteParse 2.19.0</span></div>' +
        ms.map(function (m) {
          var r = rows[m[0]], max = r.better === "higher" ? 1 : Math.max(r.spdf, r.liteparse) * 1.04;
          return '<div class="mrow"><div class="mrow-top"><span class="name">' + m[1] + '</span><span class="delta ' + r.status + '">' + deltaText(r) + "</span></div>" +
            '<div class="bars"><div class="bar s"><span data-w="' + (r.spdf / max * 100).toFixed(1) + '"></span></div><span class="v">' + fmtVal(r.metric, r.spdf) + "</span>" +
            '<div class="bar l"><span data-w="' + (r.liteparse / max * 100).toFixed(1) + '"></span></div><span class="v">' + fmtVal(r.metric, r.liteparse) + "</span></div></div>";
        }).join("") + "</div></div>";
    }).join("");
    $$(".magic-card").forEach(magic);

    // OCR case bars
    var CASES = [["scanned_chinese", "Chinese scan"], ["scanned_rotated_90", "Scan rotated 90°"], ["scanned_skewed", "Skewed scan"], ["scanned_multilang", "Accented Latin scan"]];
    $("#case-bars").innerHTML = CASES.filter(function (c) { return rows["edge.case." + c[0]]; }).map(function (c) {
      var r = rows["edge.case." + c[0]];
      return '<div class="case-row"><span class="cname">' + c[1] + '</span><div class="bars"><div class="bar s"><span data-w="' + (r.spdf * 100).toFixed(1) + '"></span></div><span class="v">' +
        f2(r.spdf, 2) + '</span><div class="bar l"><span data-w="' + (r.liteparse * 100).toFixed(1) + '"></span></div><span class="v">' + f2(r.liteparse, 2) + "</span></div></div>";
    }).join("");

    // grow bars when visible
    var bars = $$(".bar > span[data-w]");
    var grow = function (b) { b.style.width = b.dataset.w + "%"; };
    if (reduce || !("IntersectionObserver" in window)) bars.forEach(grow);
    else {
      var bio = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { grow(e.target); bio.unobserve(e.target); } }); });
      bars.forEach(function (b) { bio.observe(b); });
    }

    // full table
    var tbody = $("#metric-table tbody");
    tbody.innerHTML = d.rows.map(function (r) {
      return '<tr class="' + r.status + '" data-s="' + r.status + '"><td>' + r.metric + "</td><td>" + r.better + '</td><td class="num">' + fmtVal(r.metric, r.spdf) +
        '</td><td class="num">' + fmtVal(r.metric, r.liteparse) + '</td><td><span class="badge ' + r.status + '">' + r.status + "</span></td></tr>";
    }).join("");
    var filt = "all", q = $("#metric-q");
    function apply() {
      var s = q.value.trim().toLowerCase();
      $$("tr", tbody).forEach(function (tr) {
        tr.hidden = !((filt === "all" || tr.dataset.s === filt) && (!s || tr.firstChild.textContent.toLowerCase().indexOf(s) >= 0));
      });
    }
    $$(".chip").forEach(function (b) {
      b.addEventListener("click", function () {
        $$(".chip").forEach(function (x) { x.classList.remove("on"); x.setAttribute("aria-pressed", "false"); });
        b.classList.add("on"); b.setAttribute("aria-pressed", "true"); filt = b.dataset.f; apply();
      });
    });
    q.addEventListener("input", apply);
  }

  fetch("./results.json", { cache: "no-cache" }).then(function (r) { return r.json(); }).then(render).catch(function () {
    $("#metric-cards").innerHTML = '<p class="muted">Could not load results.json. See <a href="https://github.com/Fanaperana/spdf/blob/main/bench/RESULTS.md">bench/RESULTS.md</a>.</p>';
  });
})();
