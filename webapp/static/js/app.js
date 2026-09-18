(() => {
  "use strict";

  // If the session has gone stale (e.g. the account no longer exists),
  // every API call starts returning 401 - bounce to login instead of
  // leaving the page stuck showing "Er ging iets mis" forever.
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await nativeFetch(...args);
    if (res.status === 401) {
      window.location.href = "/login";
    }
    return res;
  };

  const chatLog = document.getElementById("chat-log");
  const composerArea = document.getElementById("composer-area");
  const composerForm = document.getElementById("composer-form");
  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");
  const chipsEl = document.getElementById("chips");
  const conversationListEl = document.getElementById("conversation-list");
  const conversationTitleEl = document.getElementById("conversation-title");
  const newChatBtn = document.getElementById("new-chat-btn");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");
  const chatWrap = document.getElementById("chat-wrap");
  const emptyHero = document.getElementById("empty-hero");
  const emptyGreeting = document.getElementById("empty-greeting");
  const emptyComposerMount = document.getElementById("empty-composer-mount");

  let chartCounter = 0;
  let isBusy = false;
  let currentConversationId = null;
  let conversationsCache = [];

  // ---------------------------------------------------------------------
  // Sidebar (mobile overlay toggle + conversation list)
  // ---------------------------------------------------------------------
  function openSidebar() {
    sidebar.classList.add("open");
    sidebarBackdrop.classList.add("open");
  }
  function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("open");
  }
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
  });
  sidebarBackdrop.addEventListener("click", closeSidebar);

  async function refreshConversationList() {
    const res = await fetch("/api/conversations");
    const data = await res.json();
    conversationsCache = data.conversations || [];
    renderConversationList();
  }

  const DATE_BUCKET_ORDER = ["Vandaag", "Gisteren", "Vorige 7 dagen", "Vorige 30 dagen", "Ouder"];

  function dateBucket(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);
    const start7 = new Date(startOfToday);
    start7.setDate(start7.getDate() - 7);
    const start30 = new Date(startOfToday);
    start30.setDate(start30.getDate() - 30);

    if (date >= startOfToday) return "Vandaag";
    if (date >= startOfYesterday) return "Gisteren";
    if (date >= start7) return "Vorige 7 dagen";
    if (date >= start30) return "Vorige 30 dagen";
    return "Ouder";
  }

  function renderConversationList() {
    conversationListEl.innerHTML = "";
    if (conversationsCache.length === 0) {
      conversationListEl.innerHTML = '<div class="conversation-empty">Nog geen gesprekken</div>';
      return;
    }

    const buckets = new Map(DATE_BUCKET_ORDER.map((label) => [label, []]));
    conversationsCache.forEach((conv) => buckets.get(dateBucket(conv.updated_at)).push(conv));

    DATE_BUCKET_ORDER.forEach((label) => {
      const items = buckets.get(label);
      if (items.length === 0) return;

      const heading = document.createElement("div");
      heading.className = "sidebar-section-label";
      heading.textContent = label;
      conversationListEl.appendChild(heading);

      items.forEach((conv) => {
        const item = document.createElement("div");
        item.className = "conversation-item" + (conv.id === currentConversationId ? " active" : "");
        item.innerHTML = `
          <span class="conversation-item-title">${escapeHtml(conv.title)}</span>
          <button type="button" class="conversation-delete" aria-label="Verwijder gesprek">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6" />
            </svg>
          </button>`;
        item.addEventListener("click", (event) => {
          if (event.target.closest(".conversation-delete")) return;
          if (isBusy) return;
          loadConversation(conv.id);
          closeSidebar();
        });
        item.querySelector(".conversation-delete").addEventListener("click", async (event) => {
          event.stopPropagation();
          await fetch(`/api/conversations/${conv.id}`, { method: "DELETE" });
          const wasActive = conv.id === currentConversationId;
          await refreshConversationList();
          if (wasActive) {
            if (conversationsCache.length > 0) loadConversation(conversationsCache[0].id);
            else startNewConversation();
          }
        });
        conversationListEl.appendChild(item);
      });
    });
  }

  function setActiveConversationTitle(title) {
    conversationTitleEl.textContent = title || "Nieuw gesprek";
  }

  async function loadConversation(conversationId) {
    currentConversationId = conversationId;
    renderConversationList();
    const conv = conversationsCache.find((c) => c.id === conversationId);
    setActiveConversationTitle(conv ? conv.title : "Gesprek");

    chatLog.innerHTML = "";
    const res = await fetch(`/api/conversations/${conversationId}/messages`);
    const data = await res.json();
    const messages = data.messages || [];

    if (messages.length === 0) {
      showWelcome();
      return;
    }

    showActiveChat();
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
  }

  function startNewConversation() {
    currentConversationId = null;
    setActiveConversationTitle("Nieuw gesprek");
    chatLog.innerHTML = "";
    renderConversationList();
    showWelcome();
    closeSidebar();
  }

  newChatBtn.addEventListener("click", () => {
    if (isBusy) return;
    startNewConversation();
  });

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
    items.forEach((item) => {
      const { icon, label } = typeof item === "string" ? { icon: "💬", label: item } : item;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.innerHTML = `<span class="chip-icon">${icon}</span>${escapeHtml(label)}`;
      chip.addEventListener("click", () => {
        if (isBusy) return;
        showActiveChat();
        addUserMessage(label);
        sendMessage(label);
      });
      chipsEl.appendChild(chip);
    });
  }

  const DEFAULT_CHIPS = [
    { icon: "📈", label: "Train een regressiemodel" },
    { icon: "🧮", label: "Train een classificatiemodel" },
    { icon: "💡", label: "Wat kan je allemaal?" },
    { icon: "📖", label: "Leg uit hoe dit project werkt" },
  ];

  function greetingTimeOfDay() {
    const hour = new Date().getHours();
    if (hour < 6) return "Goedenacht";
    if (hour < 12) return "Goedemorgen";
    if (hour < 18) return "Goedemiddag";
    return "Goedenavond";
  }

  function showWelcome() {
    const username = document.querySelector(".user-name")?.textContent?.trim();
    emptyGreeting.textContent = username
      ? `${greetingTimeOfDay()}, ${username}`
      : greetingTimeOfDay();
    emptyComposerMount.appendChild(composerArea);
    emptyHero.hidden = false;
    chatLog.hidden = true;
    chatLog.innerHTML = "";
    setChips(DEFAULT_CHIPS);
  }

  function showActiveChat() {
    emptyHero.hidden = true;
    chatLog.hidden = false;
    chatWrap.appendChild(composerArea);
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
  function actionToHtml(action) {
    if (action.type === "predict") return renderPredictCard(action.result);
    if (action.type === "model_summary") return renderModelSummaryCard(action.result);
    return "";
  }

  async function sendMessage(text) {
    isBusy = true;
    sendBtn.disabled = true;

    const bubble = addTypingIndicator();
    const isNewConversation = currentConversationId === null;

    let res;
    try {
      res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, conversation_id: currentConversationId }),
      });
    } catch (err) {
      showError(bubble, `Kon geen verbinding maken met de server: ${err}`);
      isBusy = false;
      sendBtn.disabled = false;
      return;
    }

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      showError(bubble, data.error || "Er ging iets mis.");
      isBusy = false;
      sendBtn.disabled = false;
      return;
    }

    let accumulatedText = "";
    let started = false;
    let actionHtml = "";
    let trainingCounter = null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();

      for (const chunk of chunks) {
        let eventType = "message";
        let dataLine = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event: ")) eventType = line.slice(7);
          else if (line.startsWith("data: ")) dataLine += line.slice(6);
        }
        if (!dataLine) continue;
        const parsed = JSON.parse(dataLine);

        if (eventType === "meta") {
          currentConversationId = parsed.conversation_id;
        } else if (eventType === "text_delta") {
          if (!started) {
            started = true;
            bubble.innerHTML = "";
          }
          accumulatedText += parsed.text;
          bubble.innerHTML = formatReply(accumulatedText) + actionHtml;
          scrollToBottom();
        } else if (eventType === "action") {
          if (parsed.type === "start_training") {
            const shell = renderTrainingCardShell(parsed.num_steps);
            actionHtml = shell.html;
            trainingCounter = shell.counter;
            setStatus("training");
          } else {
            actionHtml = actionToHtml(parsed);
          }
          bubble.innerHTML = formatReply(accumulatedText) + actionHtml;
          scrollToBottom();
        } else if (eventType === "done") {
          if (!started) bubble.innerHTML = formatReply(parsed.reply) + actionHtml;
        }
      }
    }

    await refreshConversationList();
    if (isNewConversation) {
      const conv = conversationsCache.find((c) => c.id === currentConversationId);
      setActiveConversationTitle(conv ? conv.title : text.slice(0, 60));
    }

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
    showActiveChat();
    addUserMessage(text);
    composerInput.value = "";
    composerInput.style.height = "auto";
    sendMessage(text);
  });

  // ---------------------------------------------------------------------
  // Init: load conversation list, open the most recent one (or a blank
  // welcome state if the user has none yet)
  // ---------------------------------------------------------------------
  (async () => {
    try {
      const statusRes = await fetch("/api/train/status");
      const statusData = await statusRes.json();
      setStatus(statusData.status);

      await refreshConversationList();
      if (conversationsCache.length > 0) {
        await loadConversation(conversationsCache[0].id);
      } else {
        startNewConversation();
      }
    } catch (err) {
      showWelcome();
    }
  })();
})();
