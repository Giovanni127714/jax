(() => {
  "use strict";

  const chatLog = document.getElementById("chat-log");
  const composerForm = document.getElementById("composer-form");
  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");
  const chipsEl = document.getElementById("chips");

  let chartCounter = 0;
  let lastTrainedTask = null;
  let lastInputDim = null;
  let lastNumClasses = null;
  let isBusy = false;

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

  function addErrorMessage(text) {
    const bubble = addAssistantMessage(`<p>${escapeHtml(text)}</p>`);
    bubble.classList.add("error-bubble");
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
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
        handleCommand(label);
      });
      chipsEl.appendChild(chip);
    });
  }

  const DEFAULT_CHIPS = [
    "Train een regressiemodel",
    "Train een classificatiemodel",
    "Doe een willekeurige voorspelling",
    "Toon modelinfo",
    "Help",
  ];

  // ---------------------------------------------------------------------
  // Welcome message
  // ---------------------------------------------------------------------
  function showWelcome() {
    addAssistantMessage(`
      <div class="welcome-title">Hoi! 👋</div>
      <p>Ik ben een chat-assistent voor je eigen JAX/Flax MLP-trainer. Geen groot taalmodel &mdash;
      ik begrijp een beperkte set commando's en bedien de trainer daarmee.</p>
      <p>Probeer bijvoorbeeld:</p>
      <p>
        <code>train een regressiemodel</code><br>
        <code>train een classificatiemodel met 3 klassen</code><br>
        <code>voorspel 1 2 0 0</code><br>
        <code>toon modelinfo</code><br>
        <code>help</code>
      </p>
    `);
    setChips(DEFAULT_CHIPS);
  }

  // ---------------------------------------------------------------------
  // Chart drawing (plain canvas, no dependencies)
  // ---------------------------------------------------------------------
  function drawLineChart(canvas, series, opts) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = { top: 12, right: 12, bottom: 20, left: 42 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    const allValues = series.flatMap((s) => s.data).filter((v) => v !== null && v !== undefined);
    if (allValues.length === 0) {
      ctx.fillStyle = "#9297ab";
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

    ctx.strokeStyle = "rgba(150,150,170,0.2)";
    ctx.fillStyle = "#9297ab";
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
      if (s.data.length === 0) return;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      s.data.forEach((v, i) => {
        if (v === null || v === undefined) return;
        const x = xForIndex(i);
        const y = yForValue(v);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
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
  // Command parsing helpers
  // ---------------------------------------------------------------------
  function extractNumbers(text) {
    const matches = text.match(/-?\d+(?:[.,]\d+)?/g) || [];
    return matches.map((m) => parseFloat(m.replace(",", ".")));
  }

  const OVERRIDE_ALIASES = {
    stappen: "num_steps",
    steps: "num_steps",
    lagen: "num_layers",
    layers: "num_layers",
    klassen: "num_classes",
    classes: "num_classes",
    samples: "num_samples",
    seed: "seed",
    hidden_dim: "hidden_dim",
    hidden: "hidden_dim",
    dropout: "dropout",
    num_steps: "num_steps",
    num_layers: "num_layers",
    num_classes: "num_classes",
    num_samples: "num_samples",
    input_dim: "input_dim",
    batch_size: "batch_size",
    learning_rate: "learning_rate",
    lr: "learning_rate",
    gradient_clip: "gradient_clip",
    weight_decay: "weight_decay",
    log_every: "log_every",
  };

  function extractOverrides(text) {
    const overrides = {};
    const re = /([a-z_]+)\s*=\s*(-?\d+(?:[.,]\d+)?)/gi;
    let match;
    while ((match = re.exec(text)) !== null) {
      const key = OVERRIDE_ALIASES[match[1].toLowerCase()];
      if (key) overrides[key] = parseFloat(match[2].replace(",", "."));
    }
    // "met 3 klassen" / "3 classes" shorthand
    const classesMatch = text.match(/(\d+)\s*klassen|klassen\s*[:=]?\s*(\d+)|(\d+)\s*classes/i);
    if (classesMatch && overrides.num_classes === undefined) {
      const n = classesMatch[1] || classesMatch[2] || classesMatch[3];
      if (n) overrides.num_classes = parseInt(n, 10);
    }
    return overrides;
  }

  // ---------------------------------------------------------------------
  // Command: train
  // ---------------------------------------------------------------------
  async function handleTrain(text) {
    const task = /classif/i.test(text) ? "classification" : "regression";
    const overrides = extractOverrides(text);

    const payload = {
      task,
      num_classes: overrides.num_classes ?? 4,
      input_dim: overrides.input_dim ?? 4,
      hidden_dim: overrides.hidden_dim ?? 64,
      num_layers: overrides.num_layers ?? 2,
      dropout: overrides.dropout ?? 0.1,
      num_steps: overrides.num_steps ?? 300,
      batch_size: overrides.batch_size ?? 32,
      learning_rate: overrides.learning_rate ?? 0.001,
      num_samples: overrides.num_samples ?? 1000,
      gradient_clip: overrides.gradient_clip ?? 1.0,
      weight_decay: overrides.weight_decay ?? 0.01,
      log_every: overrides.log_every ?? Math.max(Math.round((overrides.num_steps ?? 300) / 10), 1),
      seed: overrides.seed ?? 42,
    };

    const taskLabel = task === "classification" ? "classificatiemodel" : "regressiemodel";
    const overrideNote = Object.keys(overrides).length
      ? `<p class="sub-value">Aangepast: ${Object.entries(overrides).map(([k, v]) => `${k}=${v}`).join(", ")}</p>`
      : "";

    chartCounter += 1;
    const lossChartId = `chart-loss-${chartCounter}`;
    const metricChartId = `chart-metric-${chartCounter}`;
    const bubble = addAssistantMessage(`
      <p>🧠 Ik start het trainen van een <strong>${taskLabel}</strong>…</p>
      ${overrideNote}
      <div class="progress-row">
        <div class="progress-track"><div class="progress-fill" id="fill-${chartCounter}"></div></div>
        <span class="progress-label" id="label-${chartCounter}">0 / ${payload.num_steps}</span>
      </div>
      <canvas class="chat-chart" id="${lossChartId}" width="600" height="200"></canvas>
      <div class="legend">${renderLegendHtml([
        { label: "Train loss", color: "#4f46e5" },
        { label: "Val loss", color: "#dc2626" },
      ])}</div>
      <div id="metric-wrap-${chartCounter}"></div>
      <div class="log-console" id="log-${chartCounter}"></div>
    `);

    let res;
    try {
      res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      bubble.classList.add("error-bubble");
      bubble.innerHTML = `<p>Kon geen verbinding maken met de server: ${escapeHtml(String(err))}</p>`;
      return;
    }
    const body = await res.json();
    if (!res.ok) {
      bubble.classList.add("error-bubble");
      bubble.innerHTML = `<p>${escapeHtml(body.error || "Onbekende fout bij starten van training.")}</p>`;
      return;
    }

    setStatus("training");
    await pollTraining(bubble, chartCounter, task);
  }

  function pollTraining(bubble, counter, task) {
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
              { label: "Train loss", color: "#4f46e5", data: history.loss },
              { label: "Val loss", color: "#dc2626", data: history.val_loss },
            ],
            { formatY: (v) => v.toFixed(2) }
          );
        }

        const hasMetric = history.val_metric && history.val_metric.some((v) => v !== null && v !== undefined);
        const metricWrap = document.getElementById(`metric-wrap-${counter}`);
        if (hasMetric && metricWrap && !metricWrap.dataset.built) {
          metricWrap.dataset.built = "1";
          metricWrap.innerHTML = `
            <div class="legend" style="margin-top:10px">${renderLegendHtml([{ label: "Val accuracy", color: "#16a34a" }])}</div>
            <canvas class="chat-chart" id="chart-metric-${counter}" width="600" height="120"></canvas>`;
        }
        if (hasMetric) {
          const metricCanvas = document.getElementById(`chart-metric-${counter}`);
          if (metricCanvas) {
            drawLineChart(metricCanvas, [{ label: "Val accuracy", color: "#16a34a", data: history.val_metric }], {
              minY: 0,
              maxY: 1,
              formatY: (v) => v.toFixed(2),
            });
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

        if (data.status === "error") {
          const note = document.createElement("p");
          note.textContent = `❌ Training mislukt: ${data.error || "onbekende fout"}`;
          bubble.appendChild(note);
        } else if (data.status === "done") {
          lastTrainedTask = task;
          lastInputDim = data.config.input_dim;
          lastNumClasses = data.config.output_dim;
          const note = document.createElement("p");
          const finalLoss = history.loss[history.loss.length - 1];
          note.innerHTML = `✅ <strong>Klaar!</strong> Laatste train loss: ${finalLoss.toFixed(4)}. Vraag me nu om een voorspelling, bv. <code>voorspel ${Array(lastInputDim).fill(0).map(() => (Math.random() * 2 - 1).toFixed(1)).join(" ")}</code>.`;
          bubble.appendChild(note);
          setChips(["Doe een willekeurige voorspelling", "Toon modelinfo", "Train opnieuw"]);
        }
        scrollToBottom();
        resolve();
      }, 400);
    });
  }

  // ---------------------------------------------------------------------
  // Command: predict
  // ---------------------------------------------------------------------
  async function handlePredict(text) {
    if (lastInputDim === null) {
      addErrorMessage("Er is nog geen model getraind. Zeg bijvoorbeeld 'train een regressiemodel' om te beginnen.");
      return;
    }

    let features;
    if (/willekeurig|random/i.test(text)) {
      features = Array.from({ length: lastInputDim }, () => Math.round((Math.random() * 4 - 2) * 100) / 100);
    } else {
      features = extractNumbers(text);
      if (features.length !== lastInputDim) {
        addErrorMessage(
          `Ik heb ${features.length} getal(len) gevonden, maar dit model verwacht er ${lastInputDim} ` +
            `(x1 t/m x${lastInputDim}). Typ bijvoorbeeld: voorspel ${Array(lastInputDim).fill("0").join(" ")}`
        );
        return;
      }
    }

    const bubble = addAssistantMessage(`<p>🔮 Voorspellen op basis van <code>[${features.join(", ")}]</code>…</p>`);

    let res;
    try {
      res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
    } catch (err) {
      bubble.classList.add("error-bubble");
      bubble.innerHTML = `<p>Kon geen verbinding maken: ${escapeHtml(String(err))}</p>`;
      return;
    }
    const data = await res.json();
    if (!res.ok) {
      bubble.classList.add("error-bubble");
      bubble.innerHTML = `<p>${escapeHtml(data.error || "Voorspellen mislukt.")}</p>`;
      return;
    }

    if (data.task === "classification") {
      const rows = data.probabilities
        .map((p, i) => {
          const predicted = i === data.predicted_class;
          return `
            <div class="class-bar-row">
              <span class="class-bar-label">Klasse ${i}</span>
              <div class="class-bar-track"><div class="class-bar-fill ${predicted ? "predicted" : ""}" style="width:${(p * 100).toFixed(1)}%"></div></div>
              <span class="class-bar-value">${(p * 100).toFixed(1)}%</span>
            </div>`;
        })
        .join("");
      bubble.innerHTML = `
        <p>Voorspelling voor <code>[${features.join(", ")}]</code>:</p>
        <div class="big-value">Klasse ${data.predicted_class}</div>
        <div class="sub-value">Kans: ${(data.probabilities[data.predicted_class] * 100).toFixed(1)}%</div>
        ${rows}`;
    } else {
      const formulaValue = 2 * (features[0] || 0) + 3 * (features[1] || 0);
      bubble.innerHTML = `
        <p>Voorspelling voor <code>[${features.join(", ")}]</code>:</p>
        <div class="big-value">${data.prediction.toFixed(3)}</div>
        <div class="sub-value">Formule 2·x1 + 3·x2 geeft (ter vergelijking): ${formulaValue.toFixed(3)}</div>`;
    }
    scrollToBottom();
  }

  // ---------------------------------------------------------------------
  // Command: model info
  // ---------------------------------------------------------------------
  async function handleModelInfo() {
    const bubble = addAssistantMessage("<p>📊 Modelinfo ophalen…</p>");
    const res = await fetch("/api/model/summary");
    const data = await res.json();
    if (!res.ok) {
      bubble.classList.add("error-bubble");
      bubble.innerHTML = `<p>${escapeHtml(data.error || "Geen model beschikbaar.")}</p>`;
      return;
    }
    const rows = data.layers
      .map((l) => `<tr><td>Laag ${l.index}</td><td>${l.input}</td><td>${l.output}</td><td>${l.params.toLocaleString("nl-NL")}</td></tr>`)
      .join("");
    bubble.innerHTML = `
      <p>Huidige architectuur:</p>
      <table class="data-table">
        <thead><tr><th>Laag</th><th>Input</th><th>Output</th><th>Parameters</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="sub-value">Totaal aantal parameters: ${data.total_params.toLocaleString("nl-NL")}</p>`;
    scrollToBottom();
  }

  // ---------------------------------------------------------------------
  // Command: status
  // ---------------------------------------------------------------------
  async function handleStatus() {
    const res = await fetch("/api/train/status");
    const data = await res.json();
    if (data.status === "idle") {
      addAssistantMessage("<p>Er is nog geen training gestart. Zeg 'train een regressiemodel' om te beginnen.</p>");
    } else if (data.status === "training") {
      addAssistantMessage(`<p>⏳ Bezig: stap ${data.current_step} van ${data.total_steps}.</p>`);
    } else if (data.status === "done") {
      addAssistantMessage("<p>✅ Er staat een getraind model klaar. Je kan nu een voorspelling vragen.</p>");
    } else {
      addAssistantMessage(`<p>❌ Laatste training eindigde met een fout: ${escapeHtml(data.error || "onbekend")}</p>`);
    }
  }

  // ---------------------------------------------------------------------
  // Command: help
  // ---------------------------------------------------------------------
  function handleHelp() {
    addAssistantMessage(`
      <p>Dit begrijp ik op dit moment:</p>
      <p>
        <code>train een regressiemodel</code> &mdash; start training (evt. met opties, zie hieronder)<br>
        <code>train een classificatiemodel met 5 klassen</code><br>
        <code>voorspel 1 2 0 0</code> &mdash; voorspelling met die featurewaarden<br>
        <code>doe een willekeurige voorspelling</code><br>
        <code>toon modelinfo</code> &mdash; architectuur en parameters<br>
        <code>status</code> &mdash; huidige trainingsstatus<br>
        <code>help</code> &mdash; dit overzicht
      </p>
      <p>Bij trainen kun je opties toevoegen als <code>sleutel=waarde</code>, bijvoorbeeld:<br>
      <code>train een regressiemodel hidden_dim=128 num_steps=1000 learning_rate=0.0005</code></p>
      <p>Beschikbare opties: hidden_dim, num_layers, dropout, num_steps, batch_size,
      learning_rate, num_samples, input_dim, num_classes, seed, gradient_clip, weight_decay, log_every.</p>
    `);
  }

  // ---------------------------------------------------------------------
  // Command router
  // ---------------------------------------------------------------------
  async function handleCommand(rawText) {
    const text = rawText.trim();
    if (!text) return;

    isBusy = true;
    sendBtn.disabled = true;

    try {
      if (/^help$|wat kan je|commando/i.test(text)) {
        handleHelp();
      } else if (/train/i.test(text)) {
        await handleTrain(text);
      } else if (/voorspel|predict|test/i.test(text)) {
        await handlePredict(text);
      } else if (/model ?info|architectuur|samenvatting/i.test(text)) {
        await handleModelInfo();
      } else if (/^status$|hoe gaat het|voortgang/i.test(text)) {
        await handleStatus();
      } else {
        addAssistantMessage(`
          <p>Dat begrijp ik niet helemaal. Ik ondersteun op dit moment: trainen, voorspellen,
          modelinfo tonen, status opvragen, en help.</p>
          <p>Typ <code>help</code> voor een volledig overzicht met voorbeelden.</p>
        `);
      }
    } finally {
      isBusy = false;
      sendBtn.disabled = false;
    }
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
    handleCommand(text);
  });

  // ---------------------------------------------------------------------
  // Init: check whether a model already exists server-side (e.g. after refresh)
  // ---------------------------------------------------------------------
  (async () => {
    showWelcome();
    try {
      const res = await fetch("/api/train/status");
      const data = await res.json();
      setStatus(data.status);
      if (data.status === "done") {
        lastTrainedTask = data.task;
        lastInputDim = data.config.input_dim;
        lastNumClasses = data.config.output_dim;
        addAssistantMessage(
          `<p>Ik zie dat er al een getraind ${data.task === "classification" ? "classificatie" : "regressie"}model klaarstaat van een vorige sessie. Je kan direct een voorspelling vragen.</p>`
        );
        setChips(["Doe een willekeurige voorspelling", "Toon modelinfo", "Train opnieuw"]);
      }
    } catch (err) {
      // Server not reachable yet; ignore, welcome message already shown.
    }
  })();
})();
