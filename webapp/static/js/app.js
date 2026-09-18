(() => {
  "use strict";

  const state = {
    task: "regression",
    polling: null,
    inputDim: 4,
    numClasses: 4,
  };

  // ---------------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // ---------------------------------------------------------------------
  // Task toggle
  // ---------------------------------------------------------------------
  const taskHint = document.getElementById("task-hint");
  document.querySelectorAll(".segmented-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".segmented-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.task = btn.dataset.task;

      const classificationOnly = document.getElementById("classification-only");
      if (state.task === "classification") {
        classificationOnly.classList.remove("hidden");
        taskHint.innerHTML =
          "Het model leert willekeurig gegenereerde punten in <code>N</code> klassen te classificeren.";
      } else {
        classificationOnly.classList.add("hidden");
        taskHint.innerHTML =
          "Het model leert de formule <code>y = 2·x1 + 3·x2 + ruis</code> te benaderen.";
      }
    });
  });

  // ---------------------------------------------------------------------
  // Status pill
  // ---------------------------------------------------------------------
  const STATUS_LABELS = {
    idle: "Inactief",
    training: "Bezig met trainen…",
    done: "Klaar",
    error: "Fout",
  };

  function setStatus(status) {
    const pill = document.getElementById("status-pill");
    pill.dataset.status = status;
    document.getElementById("status-text").textContent = STATUS_LABELS[status] || status;
  }

  // ---------------------------------------------------------------------
  // Chart drawing (plain canvas, no dependencies)
  // ---------------------------------------------------------------------
  function drawLineChart(canvas, series, opts) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = { top: 14, right: 14, bottom: 24, left: 46 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    const allValues = series.flatMap((s) => s.data).filter((v) => v !== null && v !== undefined);
    if (allValues.length === 0) {
      ctx.fillStyle = "#9297ab";
      ctx.font = "13px sans-serif";
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

    // Gridlines + y-axis labels
    ctx.strokeStyle = "rgba(150,150,170,0.2)";
    ctx.fillStyle = "#9297ab";
    ctx.font = "10px sans-serif";
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const v = minY + ((maxY - minY) * i) / gridLines;
      const y = yForValue(v);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      ctx.fillText(opts.formatY ? opts.formatY(v) : v.toFixed(2), 2, y + 3);
    }

    // Lines
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

    // X axis label (step range)
    ctx.fillStyle = "#9297ab";
    if (opts.xLabels && opts.xLabels.length) {
      ctx.fillText(String(opts.xLabels[0]), padding.left, h - 6);
      ctx.fillText(
        String(opts.xLabels[opts.xLabels.length - 1]),
        w - padding.right - 24,
        h - 6
      );
    }
  }

  function renderLegend(container, series) {
    container.innerHTML = "";
    series.forEach((s) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-swatch" style="background:${s.color}"></span>${s.label}`;
      container.appendChild(item);
    });
  }

  // ---------------------------------------------------------------------
  // Training form submit
  // ---------------------------------------------------------------------
  const trainForm = document.getElementById("train-form");
  const trainBtn = document.getElementById("train-btn");
  const trainError = document.getElementById("train-error");
  const logConsole = document.getElementById("log-console");
  let loggedSteps = new Set();

  function fieldValue(id) {
    return document.getElementById(id).value;
  }

  trainForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    trainError.classList.add("hidden");
    loggedSteps = new Set();
    logConsole.textContent = "";

    const payload = {
      task: state.task,
      num_classes: fieldValue("num_classes"),
      input_dim: fieldValue("input_dim"),
      hidden_dim: fieldValue("hidden_dim"),
      num_layers: fieldValue("num_layers"),
      dropout: fieldValue("dropout"),
      num_steps: fieldValue("num_steps"),
      batch_size: fieldValue("batch_size"),
      learning_rate: fieldValue("learning_rate"),
      num_samples: fieldValue("num_samples"),
      gradient_clip: fieldValue("gradient_clip"),
      weight_decay: fieldValue("weight_decay"),
      log_every: fieldValue("log_every"),
      seed: fieldValue("seed"),
    };

    trainBtn.disabled = true;
    trainBtn.textContent = "Bezig…";

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.error || "Onbekende fout bij starten van training.");
      }
      setStatus("training");
      startPolling();
    } catch (err) {
      trainError.textContent = err.message;
      trainError.classList.remove("hidden");
      trainBtn.disabled = false;
      trainBtn.textContent = "Start training";
    }
  });

  function startPolling() {
    if (state.polling) clearInterval(state.polling);
    state.polling = setInterval(pollStatus, 400);
    pollStatus();
  }

  async function pollStatus() {
    const res = await fetch("/api/train/status");
    const data = await res.json();

    setStatus(data.status);

    const pct = data.total_steps > 0 ? (data.current_step / data.total_steps) * 100 : 0;
    document.getElementById("progress-fill").style.width = `${pct}%`;
    document.getElementById("progress-label").textContent = `${data.current_step} / ${data.total_steps}`;

    const history = data.history || { step: [], loss: [], val_loss: [], val_metric: [] };

    const lossSeries = [
      { label: "Train loss", color: "#4f46e5", data: history.loss },
      { label: "Val loss", color: "#dc2626", data: history.val_loss },
    ];
    drawLineChart(document.getElementById("loss-chart"), lossSeries, {
      xLabels: history.step,
      formatY: (v) => v.toFixed(2),
    });
    renderLegend(document.getElementById("chart-legend"), lossSeries);

    const metricWrap = document.getElementById("metric-chart-wrap");
    const hasMetric = history.val_metric && history.val_metric.some((v) => v !== null && v !== undefined);
    if (hasMetric) {
      metricWrap.classList.remove("hidden");
      drawLineChart(
        document.getElementById("metric-chart"),
        [{ label: "Val accuracy", color: "#16a34a", data: history.val_metric }],
        { xLabels: history.step, minY: 0, maxY: 1, formatY: (v) => v.toFixed(2) }
      );
    } else {
      metricWrap.classList.add("hidden");
    }

    history.step.forEach((step, i) => {
      if (loggedSteps.has(step)) return;
      loggedSteps.add(step);
      const loss = history.loss[i];
      const valLoss = history.val_loss[i];
      const valMetric = history.val_metric[i];
      let line = `Step ${step}/${data.total_steps} | loss: ${loss.toFixed(4)} | val_loss: ${valLoss.toFixed(4)}`;
      if (valMetric !== null && valMetric !== undefined) {
        line += ` | val_metric: ${valMetric.toFixed(4)}`;
      }
      logConsole.textContent += line + "\n";
      logConsole.scrollTop = logConsole.scrollHeight;
    });

    if (data.status === "training") {
      return;
    }

    clearInterval(state.polling);
    state.polling = null;
    trainBtn.disabled = false;
    trainBtn.textContent = "Start training";

    if (data.status === "error") {
      trainError.textContent = data.error || "Er ging iets mis tijdens het trainen.";
      trainError.classList.remove("hidden");
    } else if (data.status === "done") {
      state.inputDim = data.config.input_dim;
      state.numClasses = data.config.output_dim;
      state.trainedTask = data.task;
      buildPredictInputs();
      loadModelSummary();
    }
  }

  // ---------------------------------------------------------------------
  // Test / predict tab
  // ---------------------------------------------------------------------
  const predictInputsEl = document.getElementById("predict-inputs");
  const predictBtn = document.getElementById("predict-btn");
  const randomizeBtn = document.getElementById("randomize-btn");
  const predictResult = document.getElementById("predict-result");
  const testHint = document.getElementById("test-hint");

  function buildPredictInputs() {
    predictInputsEl.innerHTML = "";
    for (let i = 0; i < state.inputDim; i++) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = `x${i + 1}`;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = "0";
      input.id = `predict-x-${i}`;
      wrap.appendChild(label);
      wrap.appendChild(input);
      predictInputsEl.appendChild(wrap);
    }
    predictBtn.disabled = false;
    randomizeBtn.disabled = false;
    testHint.textContent =
      state.trainedTask === "classification"
        ? `Model getraind op ${state.inputDim} features, ${state.numClasses} klassen.`
        : `Model getraind op ${state.inputDim} features. x1 en x2 sturen de formule 2·x1 + 3·x2.`;
  }

  randomizeBtn.addEventListener("click", () => {
    for (let i = 0; i < state.inputDim; i++) {
      const input = document.getElementById(`predict-x-${i}`);
      input.value = (Math.random() * 4 - 2).toFixed(2);
    }
  });

  predictBtn.addEventListener("click", async () => {
    const features = [];
    for (let i = 0; i < state.inputDim; i++) {
      features.push(parseFloat(document.getElementById(`predict-x-${i}`).value || "0"));
    }

    predictBtn.disabled = true;
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Voorspellen mislukt.");
      renderPredictResult(data, features);
    } catch (err) {
      predictResult.innerHTML = `<p class="error-banner">${err.message}</p>`;
    } finally {
      predictBtn.disabled = false;
    }
  });

  function renderPredictResult(data, features) {
    if (data.task === "classification") {
      const rows = data.probabilities
        .map((p, i) => {
          const predicted = i === data.predicted_class;
          return `
            <div class="class-bar-row">
              <span class="class-bar-label">Klasse ${i}</span>
              <div class="class-bar-track">
                <div class="class-bar-fill ${predicted ? "predicted" : ""}" style="width:${(p * 100).toFixed(1)}%"></div>
              </div>
              <span class="class-bar-value">${(p * 100).toFixed(1)}%</span>
            </div>`;
        })
        .join("");
      predictResult.innerHTML = `
        <div class="big-value">Klasse ${data.predicted_class}</div>
        <div class="sub-value">Voorspelde kans: ${(data.probabilities[data.predicted_class] * 100).toFixed(1)}%</div>
        <div style="margin-top:16px">${rows}</div>`;
    } else {
      const formulaValue = 2 * (features[0] || 0) + 3 * (features[1] || 0);
      predictResult.innerHTML = `
        <div class="big-value">${data.prediction.toFixed(3)}</div>
        <div class="sub-value">Formule 2·x1 + 3·x2 geeft: ${formulaValue.toFixed(3)}</div>`;
    }
  }

  // ---------------------------------------------------------------------
  // Model info tab
  // ---------------------------------------------------------------------
  async function loadModelSummary() {
    const res = await fetch("/api/model/summary");
    const data = await res.json();
    if (!res.ok) return;

    const tbody = document.querySelector("#layers-table tbody");
    tbody.innerHTML = data.layers
      .map(
        (l) =>
          `<tr><td>Laag ${l.index}</td><td>${l.input}</td><td>${l.output}</td><td>${l.params.toLocaleString("nl-NL")}</td></tr>`
      )
      .join("");
    document.getElementById("total-params").textContent = `Totaal aantal parameters: ${data.total_params.toLocaleString("nl-NL")}`;
    document.getElementById("config-dump").textContent = JSON.stringify(data.config, null, 2);
  }

  // Initial paint of empty chart
  drawLineChart(document.getElementById("loss-chart"), [
    { label: "Train loss", color: "#4f46e5", data: [] },
    { label: "Val loss", color: "#dc2626", data: [] },
  ], {});
  renderLegend(document.getElementById("chart-legend"), [
    { label: "Train loss", color: "#4f46e5" },
    { label: "Val loss", color: "#dc2626" },
  ]);
})();
