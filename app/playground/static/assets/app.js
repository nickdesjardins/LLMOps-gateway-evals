// The playground: asks the gateway a question (JSON or Server-Sent Events), then shows what the
// gateway did with it: the cache lookup, the router's decision, every model attempt, tokens,
// cost, and the trace of spans on one time axis. Everything comes from this service's own API:
// /api/v1/ask, /api/v1/ask/stream, /api/v1/traces/{id}, /api/v1/metrics, /api/v1/similarity.
// All text is set with textContent: nothing from a response is ever parsed as HTML.
(function () {
  "use strict";

  var root = document.documentElement;
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

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
    if (value >= 1000) return (value / 1000).toFixed(2) + " s";
    if (value >= 10) return Math.round(value) + " ms";
    return value.toFixed(2) + " ms";
  }
  function usd(text) {
    if (text === null || text === undefined) return "unknown";
    var value = Number(text);
    if (value === 0) return "$0";
    if (value < 0.01) return "$" + value.toFixed(6);
    return "$" + value.toFixed(4);
  }
  function pct(value) { return value === null || value === undefined ? "-" : (value * 100).toFixed(1) + "%"; }
  function say(text, tone) { var s = $("status"); s.textContent = text || ""; s.dataset.tone = tone || ""; }

  async function api(path, body) {
    var options = { headers: { "Accept": "application/json" } };
    if (body !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    var response = await fetch(path, options);
    var data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    if (!response.ok) {
      var error = (data && data.error) || { message: "HTTP " + response.status, incident_id: "" };
      var err = new Error(error.message + (error.incident_id ? " (incident " + error.incident_id + ")" : ""));
      err.status = response.status;
      throw err;
    }
    return data;
  }

  // ------------------------------------------------------------------ config ---
  var config = null;
  api("/api/v1/playground/config").then(function (data) {
    config = data;
    var words = $("provider-words");
    var chip = $("provider-chip");
    if (data.provider === "offline") {
      words.textContent = "Offline provider";
      $("offline-banner").hidden = false;
      root.dataset.demo = "waking";
    } else if (!data.provider_ready) {
      words.textContent = data.provider + ": no key";
      root.dataset.demo = "asleep";
    } else {
      words.textContent = data.provider + (data.demo_mode ? " (demo)" : "");
      root.dataset.demo = "awake";
    }
    chip.title = "light: " + data.chains.light.join(", ") + "\nheavy: " + data.chains.heavy.join(", ") +
      "\ncache: " + data.cache_backend + ", threshold " + data.threshold;
    var box = $("samples");
    data.samples.forEach(function (sample) {
      var button = el("button", "sample");
      button.type = "button";
      button.dataset.group = sample.group;
      button.appendChild(el("span", "sample__group", sample.group));
      button.appendChild(document.createTextNode(sample.question));
      button.addEventListener("click", function () { $("question").value = sample.question; ask(); });
      box.appendChild(button);
    });
  }).catch(function (error) { say("The gateway did not answer: " + error.message, "bad"); });

  // --------------------------------------------------------------- the answer ---
  function startAnswer() {
    $("answer").hidden = false;
    $("answer").dataset.refused = "false";
    $("answer-text").textContent = "";
    clear($("citations"));
    $("answer-badge").textContent = "";
    $("answer-meta").textContent = "";
  }

  function showMeta(meta) {
    var cache = meta.cache || {};
    var badge = $("answer-badge");
    if (cache.outcome === "hit") {
      badge.textContent = cache.match === "exact" ? "cache hit (exact)" : "cache hit (semantic)";
      badge.dataset.tone = "ok";
    } else {
      badge.textContent = "cache " + (cache.outcome || "off");
      badge.dataset.tone = "muted";
    }
  }

  function showDone(result) {
    var answer = $("answer");
    answer.dataset.refused = result.refused ? "true" : "false";
    $("answer-text").textContent = result.answer;
    var meta = [result.model, result.tier + " tier", ms(result.latency_ms)];
    if (result.ttft_ms !== null && result.ttft_ms !== undefined) meta.push("first text after " + ms(result.ttft_ms));
    $("answer-meta").textContent = meta.join(" · ");
    var list = $("citations");
    clear(list);
    result.citations.forEach(function (c) { list.appendChild(el("li", null, "[" + c.label + "] " + c.title + " (" + c.chunk_id + ")")); });
    renderDid(result);
    loadTrace(result.trace_id);
    loadMetrics();
  }

  function renderDid(result) {
    $("did-empty").hidden = true;
    $("did").hidden = false;
    var cache = result.cache || {};
    var words = cache.outcome === "hit"
      ? (cache.match === "exact" ? "exact hit: one key lookup, no embedding" : "semantic hit: nearest cached question “" + cache.nearest_question + "”")
      : cache.outcome === "miss" ? "miss" + (cache.nearest_question ? ": nearest cached question “" + cache.nearest_question + "”" : ": nothing cached yet")
      : cache.outcome;
    $("did-cache").textContent = words + (cache.total_ms !== null && cache.total_ms !== undefined ? " (" + ms(cache.total_ms) + ")" : "");
    var gauge = $("gauge");
    var similarity = cache.similarity;
    var threshold = cache.threshold || (config && config.threshold) || 0.96;
    $("gauge-bar").style.width = similarity === null || similarity === undefined ? "0" : Math.max(0, Math.min(1, similarity)) * 100 + "%";
    $("gauge-mark").style.left = threshold * 100 + "%";
    gauge.dataset.tone = cache.outcome === "hit" ? "ok" : (similarity >= threshold ? "warn" : "");
    var simWords = similarity === null || similarity === undefined ? "no cached question to compare" : similarity.toFixed(4) + " (threshold " + threshold + ")";
    if (cache.blocked && cache.blocked.length) {
      simWords += "; the contrast guard refused " + cache.blocked.map(function (b) { return "“" + b.question + "” (" + b.similarity + ", " + b.reason + ")"; }).join(", ");
    }
    $("did-similarity").textContent = simWords;
    var route = result.route;
    $("did-route").textContent = route
      ? route.tier + " tier, score " + route.score.toFixed(2) + " against " + route.threshold + " (" + route.decided_by + ")" +
        ": base " + route.base + (route.signals.length ? ", " + route.signals.map(function (s) { return s.name + " " + (s.weight > 0 ? "+" : "") + s.weight; }).join(", ") : "")
      : "not consulted: the cache answered";
    var model = $("did-model");
    clear(model);
    if (!result.attempts.length) {
      model.textContent = "none called (answer first produced by " + result.model + ")";
    } else {
      var ul = el("ul");
      result.attempts.forEach(function (a) {
        ul.appendChild(el("li", null, a.model + " (" + a.tier + "): " + (a.outcome === "ok" ? "answered" : a.outcome + (a.detail ? ", " + a.detail : "")) + ", " + ms(a.latency_ms)));
      });
      model.appendChild(ul);
    }
    var usage = result.usage;
    $("did-tokens").textContent = usage.input_tokens + " in, " + usage.output_tokens + " out" + (usage.cached_input_tokens ? ", " + usage.cached_input_tokens + " from the provider's prompt cache" : "") + (usage.estimated ? " (estimated)" : "");
    var cost = result.cost;
    $("did-cost").textContent = cache.outcome === "hit"
      ? "$0 now; the hit avoided " + usd(cost.avoided_usd) + " at list price"
      : usd(cost.actual_usd) + " actual, " + usd(cost.reference_usd) + " at the list price of " + ((config && config.reference_models[result.tier]) || "the reference model");
    $("did-latency").textContent = ms(result.latency_ms) + (result.ttft_ms !== null && result.ttft_ms !== undefined ? ", first text after " + ms(result.ttft_ms) : "");
    var inj = result.injection;
    $("did-guard").textContent = inj.detected ? "flagged: " + inj.signals.join(", ") + " (answered as data, not cached)" : "nothing flagged";
  }

  // -------------------------------------------------------------- the trace ---
  async function loadTrace(traceId) {
    try {
      var trace = await api("/api/v1/traces/" + encodeURIComponent(traceId));
      renderTrace(trace);
    } catch (e) { /* the trace may have been evicted: nothing to draw */ }
  }

  function spanDetail(span) {
    var a = span.attributes || {};
    switch (span.name) {
      case "guard.defang": return a.detected ? "flagged: " + a.signals.join(", ") : "nothing flagged";
      case "cache.lookup": return a.outcome + (a.match ? " (" + a.match + ")" : "") + (a.similarity !== null && a.similarity !== undefined ? ", similarity " + a.similarity : "") + ", embed " + ms(a.embed_ms) + ", search " + ms(a.search_ms) + ", " + a.backend;
      case "retrieval.hybrid": return (a.chunks || []).map(function (c) { return c.chunk_id + " (dense " + c.dense + ")"; }).join(", ");
      case "router.complexity": return a.tier + ", score " + a.score + ", " + a.documents_in_play + " document(s) in play";
      case "cache.write": return a.stored ? "stored" : "refused by the store";
      default:
        if (span.kind === "generation") {
          var u = span.usage || {};
          return span.model + ": " + (span.status === "ok" ? "answered" : (a.failure || "failed") + (a.detail ? ", " + a.detail : "")) +
            (u.input_tokens !== undefined ? ", " + u.input_tokens + " in / " + u.output_tokens + " out" : "") +
            (span.reference_cost_usd ? ", list price " + usd(span.reference_cost_usd) : "");
        }
        return "";
    }
  }

  function renderTrace(trace) {
    $("trace-empty").hidden = true;
    $("trace-id").textContent = "trace " + trace.trace_id + " · " + ms(trace.latency_ms);
    var list = $("waterfall");
    clear(list);
    var total = Math.max(trace.latency_ms || 1, 0.001);
    trace.spans.forEach(function (span) {
      var li = el("li");
      li.appendChild(el("span", "wf__name", span.name));
      var track = el("span", "wf__track");
      var bar = el("span", "wf__bar");
      bar.dataset.kind = span.kind;
      bar.dataset.status = span.status;
      bar.style.left = (span.start_ms / total) * 100 + "%";
      bar.style.width = Math.max(0.4, ((span.duration_ms || 0) / total) * 100) + "%";
      track.appendChild(bar);
      li.appendChild(track);
      li.appendChild(el("span", "wf__ms", ms(span.duration_ms)));
      var detail = spanDetail(span);
      if (detail) li.appendChild(el("span", "wf__detail", detail));
      list.appendChild(li);
    });
  }

  // ---------------------------------------------------------------- metrics ---
  async function loadMetrics() {
    try {
      var m = await api("/api/v1/metrics");
      $("m-requests").textContent = m.requests;
      $("m-hit-rate").textContent = pct(m.cache_hit_rate);
      $("m-hit-p95").textContent = m.latency_ms.hit.count ? ms(m.latency_ms.hit.p95) : "-";
      $("m-miss-p95").textContent = m.latency_ms.miss.count ? ms(m.latency_ms.miss.p95) : "-";
      $("m-reduction").textContent = pct(m.cost.cost_reduction);
      $("m-fallback").textContent = pct(m.fallback_rate);
      $("metrics-since").textContent = "since " + m.since + " · " + m.cache_entries + " cached answers";
    } catch (e) { /* the next poll tries again */ }
  }
  loadMetrics();
  setInterval(loadMetrics, 15000);

  // -------------------------------------------------------------------- ask ---
  var busy = false;
  async function ask() {
    if (busy) return;
    var question = $("question").value.trim();
    if (!question) { say("Type a question, or pick a sample.", "bad"); return; }
    busy = true;
    $("ask-button").disabled = true;
    say("Asking the gateway...");
    startAnswer();
    var body = { question: question, cache: $("cache-mode").value };
    try {
      if ($("stream").checked) await askStream(body);
      else {
        var result = await api("/api/v1/ask", body);
        showMeta(result);
        showDone(result);
      }
      say("");
    } catch (error) {
      $("answer").hidden = true;
      say(error.message, "bad");
    } finally {
      busy = false;
      $("ask-button").disabled = false;
    }
  }

  async function askStream(body) {
    var response = await fetch("/api/v1/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(body)
    });
    if (!response.ok) {
      var data = null;
      try { data = await response.json(); } catch (e) { data = null; }
      var error = (data && data.error) || { message: "HTTP " + response.status };
      throw new Error(error.message + (error.incident_id ? " (incident " + error.incident_id + ")" : ""));
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var text = $("answer-text");
    for (;;) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      var cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        var block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        var name = "message", payload = "";
        block.split("\n").forEach(function (line) {
          if (line.indexOf("event: ") === 0) name = line.slice(7);
          else if (line.indexOf("data: ") === 0) payload += line.slice(6);
        });
        var data2 = payload ? JSON.parse(payload) : {};
        if (name === "meta") { showMeta(data2); say("Streaming..."); }
        else if (name === "delta") text.textContent += data2.text;
        else if (name === "done") showDone(data2);
        else if (name === "error") throw new Error(data2.message + (data2.incident_id ? " (incident " + data2.incident_id + ")" : ""));
      }
    }
  }

  $("ask-button").addEventListener("click", ask);
  $("question").addEventListener("keydown", function (event) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) ask();
  });

  // ---------------------------------------------------------- similarity lab ---
  $("lab-button").addEventListener("click", async function () {
    var out = $("lab-result");
    out.textContent = "Comparing...";
    try {
      var r = await api("/api/v1/similarity", { a: $("lab-a").value, b: $("lab-b").value });
      var words = "Cosine similarity " + r.similarity.toFixed(4) + " against the threshold " + r.threshold + ". ";
      if (r.guard) words += "The contrast guard would refuse the hit: " + r.guard + ". ";
      words += r.would_hit ? "The cache would serve one's answer for the other." : "The cache would treat them as different questions.";
      out.textContent = words;
    } catch (error) { out.textContent = error.message; }
  });
})();
