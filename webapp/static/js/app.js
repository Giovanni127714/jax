(() => {
  "use strict";

  const chatLog = document.getElementById("chat-log");
  const composerForm = document.getElementById("composer-form");
  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");
  const chipsEl = document.getElementById("chips");

  let chartCounter = 0;
  let isBusy = false;

  // ---------------------------------------------------------------------
  // Theme-aware colors (read live so charts match light/dark mode)
  // ---------------------------------------------------------------------
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---------------------------------------------------------------------
  // Status pill
  // ---------------------------------------------------------------------
  const STATUS_LABELS = { idle: "Inactief", training: "Bezig met trainen…", done: "Klaar", error: "Fout" };

  function setStatus(status) {
    const pill = document.getElementById("status-pill");
    pill.dataset.status = status;
    document.getElementById("status-text").textContent = STATUS_LABELS[status] || status;
  }

  // ---------------------------------------------------------------------
  // Chat rendering
  // ---------------------------------------------------------------------
  function scrollToBottom() {
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg-row user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function addAssistantMessage(html) {
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    row.innerHTML = `<div class="msg-avatar">◆</div><div class="bubble">${html}</div>`;
    chatLog.appendChild(row);
    scrollToBottom();
    return row.querySelector(".bubble");
  }

  function addTypingIndicator() {
    return addAssistantMessage(
      '<div class="typing-dots"><span></span><span></span><span></span></div>'
    );
  }

  function showError(bubble, text) {
    bubble.innerHTML = `<p class="error-text">${escapeHtml(text)}</p>`;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Very small markdown-ish formatter for Claude's free-text replies:
  // paragraphs, **bold**, `code`, and line breaks. Deliberately minimal.
  function formatReply(text) {
    const escaped = escapeHtml(text);
    const withInline = escaped
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    return withInline
      .split(/\n{2,}/)
      .map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`)
      .join("");
  }

  function setChips(items) {
    chipsEl.innerHTML = "";
    items.forEach((label) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = label;
      chip.addEventListener("click", () => {
        if (isBusy) return;
        addUserMessage(label);
        sendMessage(label);
      });
      chipsEl.appendChild(chip);
    });
  }

  const DEFAULT_CHIPS = [
    "Train een regressiemodel",
    "Train een classificatiemodel",
    "Wat kan je allemaal?",
    "Leg uit hoe dit project werkt",
  ];

  function showWelcome() {
    addAssistantMessage(`
      <div class="welcome-title">Hoi! 👋</div>
      <p>Ik ben de JAX MLP Assistent &mdash; een echte Claude-gedreven chatbot die net zo vrij
      kan converseren als ChatGPT of Claude, én die een MLP kan trainen en testen wanneer je
      dat vraagt.</p>
      <p>Vraag me letterlijk alles, of probeer bijvoorbeeld:</p>
      <p>
        <code>train een regressiemodel</code><br>
        <code>train een classificatiemodel met 3 klassen</code><br>
        <code>voorspel voor 1 2 0 0</code><br>
        <code>laat de modelarchitectuur zien</code>
      </p>
    `);
    setChips(DEFAULT_CHIPS);
  }

  // ---------------------------------------------------------------------
  // Chart drawing (plain canvas, no dependencies, theme-aware colors)
  // ---------------------------------------------------------------------
  function drawLineChart(canvas, series, opts) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = { top: 12, right: 12, bottom: 20, left: 42 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    const gridColor = cssVar("--border");
    const mutedColor = cssVar("--text-muted");

    const allValues = series.flatMap((s) => s.data).filter((v) => v !== null && v !== undefined);
    if (allValues.length === 0) {
      ctx.fillStyle = mutedColor;
      ctx.font = "12px sans-serif";
      ctx.fillText("Nog geen data", padding.left, h / 2);
      return;
    }

    let minY = opts.minY !== undefined ? opts.minY : Math.min(...allValues);
    let maxY = opts.maxY !== undefined ? opts.maxY : Math.max(...allValues);
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const yPad = (maxY - minY) * 0.08;
    minY -= yPad;
    maxY += yPad;

    const maxLen = Math.max(...series.map((s) => s.data.length), 1);
    const xForIndex = (i) => padding.left + (maxLen <= 1 ? 0 : (i / (maxLen - 1)) * plotW);
    const yForValue = (v) => padding.top + plotH - ((v - minY) / (maxY - minY)) * plotH;

    ctx.strokeStyle = gridColor;
    ctx.fillStyle = mutedColor;
    ctx.font = "9px sans-serif";
    const gridLines = 3;
    for (let i = 0; i <= gridLines; i++) {
      const v = minY + ((maxY - minY) * i) / gridLines;
      const y = yForValue(v);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      ctx.fillText(opts.formatY ? opts.formatY(v) : v.toFixed(2), 2, y + 3);
    }

    series.forEach((s) => {
      const points = [];
      s.data.forEach((v, i) => {
        if (v === null || v === undefined) return;
        points.push([xForIndex(i), yForValue(v)]);
      });
      if (points.length === 0) return;

      if (s.fill) {
        const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
        gradient.addColorStop(0, `${s.color}33`);
        gradient.addColorStop(1, `${s.color}00`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(points[0][0], padding.top + plotH);
        points.forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.lineTo(points[points.length - 1][0], padding.top + plotH);
        ctx.closePath();
        ctx.fill();
      }

      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      points.forEach(([x, y], idx) => {
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  function renderLegendHtml(series) {
    return series
      .map((s) => `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.label}</span>`)
      .join("");
  }

  // ---------------------------------------------------------------------
  // Rich content renderers (shared by live responses and history replay)
  // ---------------------------------------------------------------------
  function renderPredictCard(result) {
    if (result.task === "classification") {
      const rows = result.probabilities
        .map((p, i) => {
          const predicted = i === result.predicted_class;
          return `
            <div class="class-bar-row">
              <span class="class-bar-label">Klasse ${i}</span>
              <div class="class-bar-track"><div class="class-bar-fill ${predicted ? "predicted" : ""}" style="width:${(p * 100).toFixed(1)}%"></div></div>
              <span class="class-bar-value">${(p * 100).toFixed(1)}%</span>
            </div>`;
        })
        .join("");
      return `
        <div class="rich-card">
          <div class="big-value">Klasse ${result.predicted_class}</div>
          <div class="sub-value">Kans: ${(result.probabilities[result.predicted_class] * 100).toFixed(1)}%</div>
          ${rows}
        </div>`;
    }
    const features = result.features || [];
    const formulaValue = 2 * (features[0] || 0) + 3 * (features[1] || 0);
    return `
      <div class="rich-card">
        <div class="big-value">${result.prediction.toFixed(3)}</div>
        <div class="sub-value">Formule 2·x1 + 3·x2 geeft (ter vergelijking): ${formulaValue.toFixed(3)}</div>
      </div>`;
  }

  function renderModelSummaryCard(result) {
    const rows = result.layers
      .map((l) => `<tr><td>Laag ${l.index}</td><td>${l.input}</td><td>${l.output}</td><td>${l.params.toLocaleString("nl-NL")}</td></tr>`)
      .join("");
    return `
      <div class="rich-card">
        <table class="data-table">
          <thead><tr><th>Laag</th><th>Input</th><th>Output</th><th>Parameters</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p class="sub-value" style="margin-top:10px">Totaal aantal parameters: ${result.total_params.toLocaleString("nl-NL")}</p>
      </div>`;
  }

  // ---------------------------------------------------------------------
  // Training progress (live card, polled until done)
  // ---------------------------------------------------------------------
  function renderTrainingCardShell(numSteps) {
    chartCounter += 1;
    const counter = chartCounter;
    const html = `
      <div class="rich-card">
        <div class="progress-row">
          <div class="progress-track"><div class="progress-fill" id="fill-${counter}"></div></div>
          <span class="progress-label" id="label-${counter}">0 / ${numSteps}</span>
        </div>
        <canvas class="chat-chart" id="chart-loss-${counter}" width="600" height="200"></canvas>
        <div class="legend">${renderLegendHtml([
          { label: "Train loss", color: cssVar("--accent") },
          { label: "Val loss", color: cssVar("--danger") },
        ])}</div>
        <div id="metric-wrap-${counter}"></div>
        <div class="log-console" id="log-${counter}"></div>
      </div>`;
    return { html, counter };
  }

  function pollTraining(counter) {
    const loggedSteps = new Set();
    return new Promise((resolve) => {
      const interval = setInterval(async () => {
        const res = await fetch("/api/train/status");
        const data = await res.json();
        setStatus(data.status);

        const pct = data.total_steps > 0 ? (data.current_step / data.total_steps) * 100 : 0;
        const fill = document.getElementById(`fill-${counter}`);
        const label = document.getElementById(`label-${counter}`);
        if (fill) fill.style.width = `${pct}%`;
        if (label) label.textContent = `${data.current_step} / ${data.total_steps}`;

        const history = data.history || { step: [], loss: [], val_loss: [], val_metric: [] };
        const lossCanvas = document.getElementById(`chart-loss-${counter}`);
        if (lossCanvas) {
          drawLineChart(
            lossCanvas,
            [
              { label: "Train loss", color: cssVar("--accent"), data: history.loss, fill: true },
              { label: "Val loss", color: cssVar("--danger"), data: history.val_loss },
            ],
            { formatY: (v) => v.toFixed(2) }
          );
        }

        const hasMetric = history.val_metric && history.val_metric.some((v) => v !== null && v !== undefined);
        const metricWrap = document.getElementById(`metric-wrap-${counter}`);
        if (hasMetric && metricWrap && !metricWrap.dataset.built) {
          metricWrap.dataset.built = "1";
          metricWrap.innerHTML = `
            <div class="legend" style="margin-top:10px">${renderLegendHtml([{ label: "Val accuracy", color: cssVar("--success") }])}</div>
            <canvas class="chat-chart" id="chart-metric-${counter}" width="600" height="120"></canvas>`;
        }
        if (hasMetric) {
          const metricCanvas = document.getElementById(`chart-metric-${counter}`);
          if (metricCanvas) {
            drawLineChart(
              metricCanvas,
              [{ label: "Val accuracy", color: cssVar("--success"), data: history.val_metric, fill: true }],
              { minY: 0, maxY: 1, formatY: (v) => v.toFixed(2) }
            );
          }
        }

        const logEl = document.getElementById(`log-${counter}`);
        history.step.forEach((step, i) => {
          if (loggedSteps.has(step)) return;
          loggedSteps.add(step);
          const loss = history.loss[i];
          const valLoss = history.val_loss[i];
          const valMetric = history.val_metric[i];
          let line = `Step ${step}/${data.total_steps} | loss: ${loss.toFixed(4)} | val_loss: ${valLoss.toFixed(4)}`;
          if (valMetric !== null && valMetric !== undefined) line += ` | val_metric: ${valMetric.toFixed(4)}`;
          if (logEl) {
            logEl.textContent += line + "\n";
            logEl.scrollTop = logEl.scrollHeight;
          }
        });

        scrollToBottom();

        if (data.status === "training") return;
        clearInterval(interval);
        resolve();
      }, 400);
    });
  }

  // ---------------------------------------------------------------------
  // Sending a message to the Claude-backed assistant
  // ---------------------------------------------------------------------
  async function sendMessage(text) {
    isBusy = true;
    sendBtn.disabled = true;

    const bubble = addTypingIndicator();

    let res;
    try {
      res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
    } catch (err) {
      showError(bubble, `Kon geen verbinding maken met de server: ${err}`);
      isBusy = false;
      sendBtn.disabled = false;
      return;
    }

    const data = await res.json();
    if (!res.ok) {
      showError(bubble, data.error || "Er ging iets mis.");
      isBusy = false;
      sendBtn.disabled = false;
      return;
    }

    let extraHtml = "";
    let trainingCounter = null;

    if (data.action) {
      if (data.action.type === "predict") {
        extraHtml = renderPredictCard(data.action.result);
      } else if (data.action.type === "model_summary") {
        extraHtml = renderModelSummaryCard(data.action.result);
      } else if (data.action.type === "start_training") {
        const shell = renderTrainingCardShell(data.action.num_steps);
        extraHtml = shell.html;
        trainingCounter = shell.counter;
        setStatus("training");
      }
    }

    bubble.innerHTML = formatReply(data.reply) + extraHtml;
    scrollToBottom();

    if (trainingCounter !== null) {
      await pollTraining(trainingCounter);
    }

    isBusy = false;
    sendBtn.disabled = false;
  }

  // ---------------------------------------------------------------------
  // Composer wiring
  // ---------------------------------------------------------------------
  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = `${Math.min(composerInput.scrollHeight, 140)}px`;
  });

  composerInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composerForm.requestSubmit();
    }
  });

  composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (isBusy) return;
    const text = composerInput.value.trim();
    if (!text) return;
    addUserMessage(text);
    composerInput.value = "";
    composerInput.style.height = "auto";
    sendMessage(text);
  });

  // ---------------------------------------------------------------------
  // Init: load persisted history, or show the welcome message
  // ---------------------------------------------------------------------
  (async () => {
    try {
      const [historyRes, statusRes] = await Promise.all([
        fetch("/api/history"),
        fetch("/api/train/status"),
      ]);
      const historyData = await historyRes.json();
      const statusData = await statusRes.json();
      setStatus(statusData.status);

      const messages = historyData.messages || [];
      if (messages.length === 0) {
        showWelcome();
        return;
      }

      messages.forEach((msg) => {
        if (msg.role === "user") {
          addUserMessage(msg.content);
          return;
        }
        let html = formatReply(msg.content);
        const attachment = msg.attachment;
        if (attachment) {
          if (attachment.type === "predict") html += renderPredictCard(attachment.result);
          else if (attachment.type === "model_summary") html += renderModelSummaryCard(attachment.result);
          else if (attachment.type === "start_training") {
            html += `<div class="rich-card"><p class="sub-value">🧠 Training gestart (${attachment.num_steps} stappen) in een eerdere sessie.</p></div>`;
          }
        }
        addAssistantMessage(html);
      });
      setChips(DEFAULT_CHIPS);
    } catch (err) {
      showWelcome();
    }
  })();
})();
