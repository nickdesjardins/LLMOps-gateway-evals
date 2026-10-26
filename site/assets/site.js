// The project page. Everything it shows is read from files in recorded/, which scripts in the
// repository write from the real code (nothing is typed in by hand):
//   recorded/runs.json             scripts/record_runs.py: requests through the real app
//   recorded/cache-threshold.json  scripts/measure_cache.py: the threshold trade-off
//   recorded/replay.json           python -m evals.replay: the cache on a replayed workload
//   recorded/routing.json          scripts/route_examples.py: the router on the golden set
//   recorded/retrieval.json        python -m evals.retrieval_check: retrieval metrics
//   recorded/thresholds.json       a copy of evals/thresholds.json (a test keeps them equal)
//   recorded/deepeval.json, ragas.json: the judged suite, once it has run with a key
// It also wakes the live demo, which sleeps on its free host.
(function () {
  "use strict";

  var root = document.documentElement;
  var $ = function (id) { return document.getElementById(id); };
  var SVG = "http://www.w3.org/2000/svg";
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function svg(tag, attrs, text) {
    var node = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function load(path) {
    return fetch(path, { headers: { "Accept": "application/json" } }).then(function (r) {
      if (!r.ok) throw new Error(path + " " + r.status);
      return r.json();
    });
  }

  // ------------------------------------------------------------------ theme ---
  var toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    });
  }

  // ---------------------------------------------------------------- helpers ---
  function ms(value) {
    if (value === null || value === undefined) return "n/a";
    if (value >= 1000) return (value / 1000).toFixed(1) + " s";
    if (value >= 10) return Math.round(value) + " ms";
    return value.toFixed(2) + " ms";
  }
  function pct(value, digits) { return value === null || value === undefined ? "n/a" : (value * 100).toFixed(digits === undefined ? 1 : digits) + "%"; }
  function money(value) {
    var n = Number(value);
    if (value === null || value === undefined || isNaN(n)) return "unknown";
    if (n === 0) return "$0";
    if (n < 0.01) return "$" + n.toFixed(6);
    return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function fact(list, term, value) {
    if (value === null || value === undefined || value === "") return;
    list.appendChild(el("dt", null, term));
    var dd = el("dd");
    if (Array.isArray(value)) {
      var ul = el("ul");
      value.forEach(function (item) { ul.appendChild(el("li", null, item)); });
      dd.appendChild(ul);
    } else {
      dd.textContent = String(value);
    }
    list.appendChild(dd);
  }
  function tabs(box, labels, onSelect, selected) {
    clear(box);
    var buttons = labels.map(function (label, index) {
      var b = el("button", "tab", label);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.addEventListener("click", function () {
        buttons.forEach(function (x) { x.setAttribute("aria-selected", "false"); });
        b.setAttribute("aria-selected", "true");
        onSelect(index);
      });
      box.appendChild(b);
      return b;
    });
    var start = selected || 0;
    if (buttons[start]) { buttons[start].setAttribute("aria-selected", "true"); onSelect(start); }
  }

  // ------------------------------------------------------------ recorded runs ---
  var runsData = null;

  function gatewayView(run) {
    var r = run.response || {};
    if (r.object === "chat.completion") {
      var g = r.x_gateway || {};
      return {
        answer: r.choices && r.choices[0] ? r.choices[0].message.content : "",
        refused: false,
        cache: g.cache || {},
        route: g.route,
        tier: g.tier,
        model: r.model,
        attempts: g.attempts || [],
        usage: { input_tokens: r.usage.prompt_tokens, output_tokens: r.usage.completion_tokens },
        cost: g.cost || {},
        latency_ms: g.latency_ms,
        ttft_ms: g.ttft_ms,
        injection: g.injection || {}
      };
    }
    return r;
  }

  function cacheWords(cache) {
    if (cache.outcome === "hit") {
      return cache.match === "exact" ? "exact hit: one key lookup, no embedding, no model" :
        "semantic hit: similarity " + cache.similarity + " to the cached “" + cache.nearest_question + "”";
    }
    if (cache.outcome === "miss") {
      var words = cache.nearest_question ? "miss: nearest cached question at " + cache.similarity + " (threshold " + cache.threshold + ")" : "miss: nothing cached yet";
      (cache.blocked || []).forEach(function (b) {
        words += "; the contrast guard refused “" + b.question + "” at " + b.similarity + " (" + b.reason + ")";
      });
      return words;
    }
    return cache.outcome || "n/a";
  }

  function spanDetail(span) {
    var a = span.attributes || {};
    switch (span.name) {
      case "guard.defang": return a.detected ? "flagged: " + a.signals.join(", ") : "nothing flagged";
      case "cache.lookup": return a.outcome + (a.match ? " (" + a.match + ")" : "") + (a.similarity !== null && a.similarity !== undefined ? ", similarity " + a.similarity : "") + ", embedding " + ms(a.embed_ms);
      case "retrieval.hybrid": return (a.chunks || []).map(function (c) { return c.chunk_id; }).join(", ");
      case "router.complexity": return a.tier + ", score " + a.score;
      case "cache.write": return a.stored ? "stored for the next asker" : "refused by the store";
      default:
        if (span.kind === "generation") {
          var u = span.usage || {};
          return (span.model || "") + ": " + (span.status === "ok" ? "answered" : (a.failure || "failed") + (a.detail ? " (" + a.detail + ")" : "")) +
            (u.input_tokens !== undefined ? ", " + u.input_tokens + " in / " + u.output_tokens + " out" : "");
        }
        return "";
    }
  }

  function waterfall(trace) {
    var list = el("ol", "waterfall");
    var total = Math.max(trace.latency_ms || 1, 0.001);
    trace.spans.forEach(function (span) {
      var li = el("li");
      li.appendChild(el("span", "wf__name", span.name));
      var track = el("span", "wf__track");
      var bar = el("span", "wf__bar");
      bar.dataset.kind = span.kind;
      bar.dataset.status = span.status;
      bar.style.left = (span.start_ms / total) * 100 + "%";
      bar.style.width = Math.max(0.5, ((span.duration_ms || 0) / total) * 100) + "%";
      track.appendChild(bar);
      li.appendChild(track);
      li.appendChild(el("span", "wf__ms", ms(span.duration_ms)));
      var detail = spanDetail(span);
      if (detail) li.appendChild(el("span", "wf__detail", detail));
      list.appendChild(li);
    });
    return list;
  }

  function showRun(run) {
    var card = $("run");
    clear(card);
    var view = gatewayView(run);
    var cache = view.cache || {};
    var head = el("div", "run__head");
    var badge = el("span", "badge", "cache " + (cache.outcome || "n/a") + (cache.match ? " (" + cache.match + ")" : ""));
    badge.dataset.tone = cache.outcome === "hit" ? "ok" : (cache.blocked && cache.blocked.length ? "warn" : "muted");
    head.appendChild(badge);
    if (view.tier) { var t = el("span", "badge", view.tier + " tier"); t.dataset.tone = "accent"; head.appendChild(t); }
    if (view.injection && view.injection.detected) { var f = el("span", "badge", "injection flagged"); f.dataset.tone = "bad"; head.appendChild(f); }
    head.appendChild(el("span", "muted small", run.endpoint));
    card.appendChild(head);
    card.appendChild(el("p", "run__why", run.why));
    if (runsData.stack.provider === "offline") {
      card.appendChild(el("p", "provider-note", "Recorded with the offline provider: no model was called, and the answer quotes the best retrieved passage. The cache, the router, the trace and the list-price costs are the real code's."));
    }
    var question = run.request.question || (run.request.messages ? run.request.messages[run.request.messages.length - 1].content : "");
    card.appendChild(el("h3", null, "Request"));
    card.appendChild(el("p", "run__q", question));
    card.appendChild(el("h3", null, "Answer"));
    var answer = el("p", "run__answer", view.answer);
    answer.dataset.refused = view.refused ? "true" : "false";
    card.appendChild(answer);
    card.appendChild(el("h3", null, "What the gateway did"));
    var dl = el("dl", "facts");
    fact(dl, "Cache", cacheWords(cache) + (cache.total_ms !== undefined && cache.total_ms !== null ? " (" + ms(cache.total_ms) + ")" : ""));
    if (view.route) {
      fact(dl, "Router", view.route.tier + " tier, score " + view.route.score + " against " + view.route.threshold + ": base " + view.route.base +
        (view.route.signals.length ? ", " + view.route.signals.map(function (s) { return s.name + " " + (s.weight > 0 ? "+" : "") + s.weight; }).join(", ") : ""));
    } else if (cache.outcome === "hit") {
      fact(dl, "Router", "not consulted: the cache answered");
    }
    fact(dl, "Model", view.attempts && view.attempts.length
      ? view.attempts.map(function (a) { return a.model + " (" + a.tier + "): " + (a.outcome === "ok" ? "answered" : a.outcome + (a.detail ? ", " + a.detail : "")) + ", " + ms(a.latency_ms); })
      : "none called; the stored answer came from " + view.model);
    if (view.usage) fact(dl, "Tokens", view.usage.input_tokens + " in, " + view.usage.output_tokens + " out" + (view.usage.estimated ? " (estimated)" : ""));
    var cost = view.cost || {};
    fact(dl, "Cost", cache.outcome === "hit" ? "$0; avoided " + money(cost.avoided_usd) + " at list price" : money(cost.actual_usd) + " actual, " + money(cost.reference_usd) + " at list price");
    fact(dl, "Latency", ms(view.latency_ms) + (view.ttft_ms !== null && view.ttft_ms !== undefined ? ", first text after " + ms(view.ttft_ms) : ""));
    if (run.events) fact(dl, "Events", run.events.map(function (e) { return e.event; }).reduce(function (acc, name) {
      var last = acc[acc.length - 1];
      if (last && last.name === name) last.count++; else acc.push({ name: name, count: 1 });
      return acc;
    }, []).map(function (x) { return x.count > 1 ? x.name + " x" + x.count : x.name; }).join(", "));
    card.appendChild(dl);
    if (run.trace) {
      card.appendChild(el("h3", null, "Trace (" + ms(run.trace.latency_ms) + ")"));
      card.appendChild(waterfall(run.trace));
    }
  }

  load("recorded/runs.json").then(function (data) {
    runsData = data;
    var provider = data.stack.provider === "offline" ? "the offline provider (no model called)" : data.stack.provider + " (" + data.stack.chains.light.concat(data.stack.chains.heavy).join(", ") + ")";
    $("runs-meta").textContent = data.runs.length + " requests recorded " + data.recorded_at.slice(0, 10) + " with " + provider + ".";
    tabs($("run-tabs"), data.runs.map(function (r) { return r.title.split(":")[0]; }), function (i) { showRun(data.runs[i]); });
    $("foot-recorded").textContent = "Requests recorded " + data.recorded_at.slice(0, 10) + ".";
    var misses = data.runs.map(gatewayView).filter(function (v) { return v.cache && v.cache.outcome === "miss" && v.usage && v.usage.input_tokens; });
    if (misses.length) {
      var avg = misses.reduce(function (s, v) { return s + v.usage.input_tokens; }, 0) / misses.length;
      $("c-input").value = Math.round(avg / 50) * 50;
      calculate();
    }
  }).catch(function () { /* the fallback sentence in the page stays */ });

  // -------------------------------------------------------------- the threshold ---
  var thresholdData = null;
  var CHART = { left: 44, right: 620, top: 18, bottom: 220 };

  function x(t) { return CHART.left + ((t - 0.80) / (0.99 - 0.80)) * (CHART.right - CHART.left); }
  function y(rate) { return CHART.bottom - rate * (CHART.bottom - CHART.top); }

  function drawChart(data) {
    var chart = $("threshold-chart");
    clear(chart);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (r) {
      chart.appendChild(svg("line", { class: "grid", x1: CHART.left, x2: CHART.right, y1: y(r), y2: y(r) }));
      chart.appendChild(svg("text", { class: "tick", x: CHART.left - 6, y: y(r) + 4, "text-anchor": "end" }, (r * 100) + "%"));
    });
    [0.80, 0.84, 0.88, 0.92, 0.96, 0.99].forEach(function (t) {
      chart.appendChild(svg("text", { class: "tick", x: x(t), y: CHART.bottom + 16, "text-anchor": "middle" }, t.toFixed(2)));
    });
    chart.appendChild(svg("line", { class: "axis", x1: CHART.left, x2: CHART.right, y1: CHART.bottom, y2: CHART.bottom }));
    function path(key, cls) {
      var d = data.sweep.map(function (s, i) { return (i ? "L" : "M") + x(s.threshold).toFixed(1) + " " + y(s[key]).toFixed(1); }).join("");
      chart.appendChild(svg("path", { class: "series " + cls, d: d }));
    }
    path("true_hit_rate", "series--true");
    path("false_hit_rate_unguarded", "series--raw");
    path("false_hit_rate", "series--false");
    chart.appendChild(svg("text", { class: "legend legend--true", x: CHART.left + 8, y: CHART.top + 10 }, "rephrasings served from the cache"));
    chart.appendChild(svg("text", { class: "legend legend--false", x: CHART.left + 8, y: CHART.top + 26 }, "look-alikes served a wrong answer (dashed: without the guard)"));
    chart.appendChild(svg("line", { class: "marker", id: "threshold-marker", x1: x(0.96), x2: x(0.96), y1: CHART.top, y2: CHART.bottom }));
    chart.appendChild(svg("text", { class: "tick", x: CHART.right, y: CHART.bottom + 34, "text-anchor": "end" }, "similarity threshold"));
  }

  function showThreshold(t) {
    var data = thresholdData;
    var row = data.sweep.reduce(function (best, s) { return Math.abs(s.threshold - t) < Math.abs(best.threshold - t) ? s : best; }, data.sweep[0]);
    $("threshold-value").textContent = row.threshold.toFixed(2);
    var marker = document.getElementById("threshold-marker");
    if (marker) { marker.setAttribute("x1", x(row.threshold)); marker.setAttribute("x2", x(row.threshold)); }
    $("t-true").textContent = pct(row.true_hit_rate);
    $("t-false").textContent = pct(row.false_hit_rate);
    $("t-false-raw").textContent = pct(row.false_hit_rate_unguarded);
    var body = $("near-table").tBodies[0];
    clear(body);
    data.pairs.filter(function (p) { return p.kind === "near_miss"; }).slice(0, 8).forEach(function (p) {
      var tr = el("tr");
      var above = p.similarity >= row.threshold;
      var sim = el("td", "num " + (above && !p.guard ? "bad-text" : ""), p.similarity.toFixed(4));
      tr.appendChild(sim);
      tr.appendChild(el("td", null, p.a));
      tr.appendChild(el("td", null, p.b));
      tr.appendChild(el("td", p.guard ? "ok-text" : "", p.guard ? "refuses: " + p.guard : (above ? "cannot tell: a wrong answer would be served" : "not needed: below the threshold")));
      body.appendChild(tr);
    });
  }

  load("recorded/cache-threshold.json").then(function (data) {
    thresholdData = data;
    $("threshold-fallback").hidden = true;
    $("threshold-body").hidden = false;
    $("threshold-meta").textContent = data.same_pairs + " rephrasings and " + data.near_miss_pairs + " look-alikes, " + data.embedder.replace("fastembed:", "") + ", measured " + data.measured_at.slice(0, 10) + ".";
    $("threshold-note").textContent = "Rephrasings of one question have a median similarity of " + data.same_similarity.median +
      "; the closest look-alike with a different answer reaches " + data.near_miss_similarity.max +
      ". The two populations overlap, so no threshold separates them: the contrast guard refuses a hit when the two questions contradict each other on a number, a code or an opposite term, and blocked " + data.guard_blocks_same +
      " of the " + data.same_pairs + " true rephrasings. Embedding one question takes " + data.query_embedding_ms.p50 + " ms (median) on " + data.query_embedding_ms.machine + ".";
    drawChart(data);
    var slider = $("threshold-slider");
    slider.value = String(data.default_threshold);
    slider.addEventListener("input", function () { showThreshold(Number(slider.value)); });
    showThreshold(data.default_threshold);
  }).catch(function () { /* the fallback sentence stays */ });

  // ------------------------------------------------------------------ replay ---
  var redisReplay = load("recorded/replay-redis.json").catch(function () { return null; });
  load("recorded/replay.json").then(function (data) {
    var w = data.workload;
    $("replay-meta").textContent = w.requests + " requests, " + w.distinct_questions + " distinct questions, " + w.intents + " intents, measured " + data.measured_at.slice(0, 10) + ".";
    tabs($("replay-tabs"), data.runs.map(function (r) { return "threshold " + r.threshold; }), function (i) {
      var r = data.runs[i];
      $("r-hit").textContent = pct(r.hit_rate);
      $("r-split").textContent = r.exact_hits + " / " + r.semantic_hits;
      $("r-false").textContent = String(r.false_hits);
      $("r-cost").textContent = pct(r.cost_reduction);
      $("r-exact").textContent = ms(r.exact_lookup_ms.p50);
      $("r-semantic").textContent = ms(r.semantic_lookup_ms.p50) + " / " + ms(r.semantic_lookup_ms.p99);
      $("replay-note").textContent = "At " + r.threshold + ": " + r.misses + " misses would have cost " + money(r.reference_cost_usd) + " at list price; the " + r.hits +
        " hits avoided " + money(r.avoided_cost_usd) + ". The contrast guard refused " + r.blocked_by_guard + " look-alike neighbours. Most hits are exact repeats: a help desk gets the same question many times, and the semantic layer adds the rephrasings on top. Recorded in recorded/replay.json by python -m evals.replay.";
      redisReplay.then(function (redis) {
        var same = redis && redis.runs.filter(function (x) { return x.threshold === r.threshold; })[0];
        if (!same) return;
        $("replay-note").textContent += " The same replay against Redis 8 (a local container, " + redis.measured_at.slice(0, 10) + ") served the same " + same.hits +
          " hits; an exact lookup took " + ms(same.exact_lookup_ms.p50) + " at the median and a semantic one " + ms(same.semantic_lookup_ms.p50) + " (p99 " + ms(same.semantic_lookup_ms.p99) + "), network round trips and embedding included.";
      });
    });
    $("c-hit").value = Math.round(data.runs[0].hit_rate * 100);
    calculate();
  }).catch(function () { /* the tiles keep their dashes */ });

  // ----------------------------------------------------------------- routing ---
  load("recorded/routing.json").then(function (data) {
    $("routing-meta").textContent = data.rows.length + " golden questions: " + data.tiers.light + " light, " + data.tiers.heavy + " heavy (threshold " + data.threshold + ").";
    $("c-heavy").value = Math.round(data.tiers.heavy / data.rows.length * 100);
    calculate();
    var body = $("routing-table").tBodies[0];
    function show(filter) {
      clear(body);
      data.rows.filter(function (r) { return filter === "all" || r.tier === filter; }).forEach(function (r) {
        var tr = el("tr");
        tr.appendChild(el("td", null, r.question));
        var tier = el("td", "tier", r.tier);
        tier.dataset.tier = r.tier;
        tr.appendChild(tier);
        tr.appendChild(el("td", "num", r.score.toFixed(2)));
        tr.appendChild(el("td", "small", r.signals.length ? r.signals.map(function (s) { return s.name + " " + (s.weight > 0 ? "+" : "") + s.weight; }).join(", ") : "none (base " + r.base + ")"));
        body.appendChild(tr);
      });
    }
    var filters = ["heavy", "light", "all"];
    tabs($("routing-tabs"), ["Heavy", "Light", "All"], function (i) { show(filters[i]); });
  }).catch(function () { /* the empty table stays */ });

  // ------------------------------------------------------------- economics ---
  // List prices per million tokens (app/pricing.py; tests/test_site.py keeps them equal).
  var PRICES = {
    light: { model: "claude-haiku-5-5", input: 0.10, output: 0.50 },
    heavy: { model: "claude-opus-5-5", input: 4.00, output: 20.00 }
  };

  function calculate() {
    var n = Number($("c-requests").value) || 0;
    var tin = Number($("c-input").value) || 0;
    var tout = Number($("c-output").value) || 0;
    var heavy = Math.min(100, Math.max(0, Number($("c-heavy").value) || 0)) / 100;
    var hit = Math.min(100, Math.max(0, Number($("c-hit").value) || 0)) / 100;
    function per(tier) { return (tin * PRICES[tier].input + tout * PRICES[tier].output) / 1e6; }
    var frontier = n * per("heavy");
    var routed = n * ((1 - heavy) * per("light") + heavy * per("heavy"));
    var rows = [
      ["Every request to " + PRICES.heavy.model, frontier],
      ["Cache only", frontier * (1 - hit)],
      ["Routing only (" + PRICES.light.model + " for light work)", routed],
      ["Cache and routing", routed * (1 - hit)]
    ];
    var body = $("calc-table").tBodies[0];
    clear(body);
    rows.forEach(function (row) {
      var tr = el("tr");
      tr.appendChild(el("td", null, row[0]));
      tr.appendChild(el("td", "num", money(row[1])));
      tr.appendChild(el("td", "num", frontier ? (row[1] === frontier ? "100%" : pct(row[1] / frontier)) : "n/a"));
      body.appendChild(tr);
    });
    $("calc-note").textContent = "Per request at list price: " + money(per("light")) + " on " + PRICES.light.model + " and " + money(per("heavy")) + " on " + PRICES.heavy.model +
      ". Anthropic's prices per million tokens: " + PRICES.light.model + " $" + PRICES.light.input.toFixed(2) + " in, $" + PRICES.light.output.toFixed(2) + " out; " +
      PRICES.heavy.model + " $" + PRICES.heavy.input.toFixed(2) + " in, $" + PRICES.heavy.output.toFixed(2) + " out (published 6 October 2026; check before quoting). Prompt caching would lower the input side further.";
  }
  ["c-requests", "c-input", "c-output", "c-heavy", "c-hit"].forEach(function (id) { $(id).addEventListener("input", calculate); });
  calculate();

  // --------------------------------------------------------------- the gate ---
  Promise.all([
    load("recorded/thresholds.json"),
    load("recorded/retrieval.json").catch(function () { return null; }),
    load("recorded/deepeval.json").catch(function () { return null; }),
    load("recorded/ragas.json").catch(function () { return null; })
  ]).then(function (all) {
    var t = all[0], retrieval = all[1], deepeval = all[2], ragas = all[3];
    var body = $("gate-table").tBodies[0];
    function measured(report, key, asPct) {
      if (!report) return "not run yet";
      var v = report.summary[key];
      if (v === null || v === undefined) return "not measured";
      return (asPct ? pct(v) : v.toFixed(3)) + " (" + (report.measured_at || report.scored_at || "").slice(0, 10) + ")";
    }
    [
      ["Retrieval hit rate@4", ">= " + t.retrieval.hit_rate_min, measured(retrieval, "hit_rate")],
      ["Retrieval context precision@4", ">= " + t.retrieval.context_precision_min, measured(retrieval, "context_precision")],
      ["Hallucination rate (DeepEval)", "<= " + pct(t.deepeval.hallucination_rate_max, 0), measured(deepeval, "hallucination_rate", true)],
      ["Faithfulness (DeepEval)", ">= " + t.deepeval.faithfulness_min, measured(deepeval, "faithfulness")],
      ["Injection resistance", ">= " + t.deepeval.injection_resistance_min, measured(deepeval, "injection_resistance")],
      ["Refusal accuracy", ">= " + t.deepeval.refusal_accuracy_min, measured(deepeval, "refusal_accuracy")],
      ["Faithfulness (Ragas)", "no drop over " + t.regression.tolerance + " from the baseline", measured(ragas, "faithfulness")],
      ["Context precision (Ragas)", "no drop over " + t.regression.tolerance + " from the baseline", measured(ragas, "context_precision")]
    ].forEach(function (row) {
      var tr = el("tr");
      row.forEach(function (cell, i) { tr.appendChild(el("td", i === 0 ? null : "num", cell)); });
      body.appendChild(tr);
    });
  }).catch(function () { /* the empty table stays */ });

  // --------------------------------------------------------------- live demo ---
  // The demo sleeps on a free host after 15 minutes without a request; the next request wakes
  // it in about a minute. It lives at another address, so this page may not read its answers.
  // What it can do is load the demo's icon as a picture: "load" fires only when the service
  // itself sends the icon, never for the host's own "waking up" page.
  var link = $("demo-link");
  var words = $("demo-words");
  var chip = $("demo-chip");
  if (!link || !words) return;

  var address;
  try { address = new URL(link.href); } catch (e) { address = null; }
  if (!address || address.protocol !== "https:" || /\.invalid$/.test(address.hostname)) {
    root.dataset.demo = "unset";
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    words.textContent = "The live demo is not online yet. This page shows recorded requests instead.";
    return;
  }
  if (chip) { chip.href = address.href; chip.hidden = false; }

  var ICON = new URL("/playground/assets/favicon.svg", address).href;
  var RETRY_MS = 4000;
  var GIVE_UP_MS = 180000;
  var startedAt = Date.now();
  var done = false;

  function say(state, text) { root.dataset.demo = state; words.textContent = text; }

  function ask() {
    if (done) return;
    var image = new Image();
    var settled = false;
    var timer = setTimeout(function () { if (!settled) { settled = true; again(); } }, 20000);
    image.onload = function () {
      if (settled) return;
      settled = true; done = true; clearTimeout(timer);
      say("awake", "The live demo is awake. Ask it a question twice, in other words, and watch the cache answer.");
    };
    image.onerror = function () { if (settled) return; settled = true; clearTimeout(timer); again(); };
    image.src = ICON + "?t=" + Date.now();
  }
  function again() {
    var waited = Date.now() - startedAt;
    if (waited > GIVE_UP_MS) {
      done = true;
      say("asleep", "The live demo has not answered for three minutes. The recorded requests below show what it does.");
      return;
    }
    if (waited > 4000) {
      say("waking", "Waking the live demo. It runs on free hosting and sleeps between visits, so this takes about a minute.");
    }
    setTimeout(ask, RETRY_MS);
  }
  ask();
})();
