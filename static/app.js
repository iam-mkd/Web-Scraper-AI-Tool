/**
 * Autonomous Web Scraper AI - Interactive Client App
 * Handles Server-Sent Events (SSE) tool execution traces, markdown rendering,
 * product comparison cards, and decoupled RAG QA pipeline.
 */

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initTabs();
  initPresetChips();
  initAgentStream();
  initRagForms();
  initCopyButton();
});

// -------------------------------------------------------------
// 1. Health Telemetry
// -------------------------------------------------------------
async function initHealthCheck() {
  const badge = document.getElementById("healthBadge");
  const text = document.getElementById("healthStatusText");
  const llmBadge = document.getElementById("llmModelBadge");
  const embedBadge = document.getElementById("embedModelBadge");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.status === "healthy") {
      badge.className = "telemetry-pill status-online";
      text.textContent = "Online • Groq & ChromaDB";
    } else {
      badge.className = "telemetry-pill status-warning";
      text.textContent = "Degraded Service";
    }

    if (data.llm_model) llmBadge.textContent = data.llm_model;
    if (data.embedding_model) embedBadge.textContent = data.embedding_model;
  } catch (err) {
    badge.className = "telemetry-pill status-error";
    text.textContent = "API Offline";
    console.warn("Health check error:", err);
  }
}

// -------------------------------------------------------------
// 2. Tab Navigation
// -------------------------------------------------------------
function initTabs() {
  // Main Navigation Tabs
  const navTabs = document.querySelectorAll(".nav-tab");
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      navTabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Results Sub-Tabs
  const resTabs = document.querySelectorAll(".res-tab");
  resTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      resTabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".res-pane").forEach((pane) => pane.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.resTab);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// -------------------------------------------------------------
// 3. Preset Query Chips
// -------------------------------------------------------------
function initPresetChips() {
  const taskInput = document.getElementById("taskInput");
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      taskInput.value = chip.dataset.query;
      taskInput.focus();
    });
  });

  const clearBtn = document.getElementById("clearBtn");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      taskInput.value = "";
      taskInput.focus();
    });
  }
}

// -------------------------------------------------------------
// 4. Real-Time Tool Execution Trace (SSE Streaming)
// -------------------------------------------------------------
function initAgentStream() {
  const form = document.getElementById("agentForm");
  const taskInput = document.getElementById("taskInput");
  const runBtn = document.getElementById("runAgentBtn");
  const traceEmptyState = document.getElementById("traceEmptyState");
  const traceTimeline = document.getElementById("traceTimeline");
  const traceStatusBadge = document.getElementById("traceStatusBadge");
  const resultsSection = document.getElementById("resultsSection");

  let activeToolNodes = {};

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const task = taskInput.value.trim();
    if (!task) return;

    // Reset UI State
    runBtn.disabled = true;
    runBtn.querySelector(".btn-text").textContent = "Agent Working...";
    traceStatusBadge.className = "badge badge-running";
    traceStatusBadge.textContent = "Autonomous Execution Live";
    traceEmptyState.style.display = "none";
    traceTimeline.style.display = "flex";
    traceTimeline.innerHTML = "";
    resultsSection.style.display = "none";
    activeToolNodes = {};

    try {
      // Connect to SSE streaming endpoint
      const response = await fetch("/api/agent/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });

      if (!response.ok) {
        throw new Error(`Streaming failed: HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep last incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const jsonStr = trimmed.slice(6);
            try {
              const event = JSON.parse(jsonStr);
              handleStreamEvent(event);
            } catch (err) {
              console.warn("Failed to parse SSE JSON:", jsonStr, err);
            }
          }
        }
      }

      traceStatusBadge.className = "badge badge-complete";
      traceStatusBadge.textContent = "Execution Complete";
    } catch (err) {
      console.error("Stream error:", err);
      traceStatusBadge.className = "badge badge-idle";
      traceStatusBadge.textContent = "Error";
      addTraceNode({
        icon: "❌",
        title: "Execution Error",
        summary: err.message,
        isError: true,
      });
    } finally {
      runBtn.disabled = false;
      runBtn.querySelector(".btn-text").textContent = "Execute Multi-Step Agent";
    }
  });

  // Event dispatcher for trace nodes
  function handleStreamEvent(event) {
    const type = event.type;

    if (type === "start") {
      // Start event is silent; planning follows immediately
      return;
    } else if (type === "planning") {
      addTraceNode({
        icon: "🧠",
        title: "Planning",
        summary: event.message || `Understanding user intent: ${event.intent || "raw performance"}`,
        considerations: event.considerations,
      });
    } else if (type === "step_start") {
      // Internal step marker; kept clean without extra card
      return;
    } else if (type === "tool_start") {
      const toolIcons = {
        search_amazon: "🔍",
        search_web: "🔍",
        get_product_details: "🔎",
        compare_products: "⚖️",
        search_rag: "📚",
        scrape_url: "🌐",
      };
      const icon = toolIcons[event.tool] || "⚙️";
      const title = event.title || "Tool selected";
      const toolSubName = event.tool_display || event.tool;

      const node = addTraceNode({
        icon,
        title,
        summary: event.summary || "",
        toolSubName,
        formattedParams: event.formatted_params,
        isActive: true,
        args: event.args,
      });
      activeToolNodes[event.tool_id || event.tool] = node;
    } else if (type === "tool_end") {
      const key = event.tool_id || event.tool;
      const node = activeToolNodes[key];
      if (node) {
        node.classList.remove("active-node");
        const statusSpan = node.querySelector(".node-status-badge");
        if (statusSpan) {
          statusSpan.textContent = `✓ Done (${event.duration_s || 0}s)`;
          statusSpan.style.color = "var(--color-success)";
        }
        if (event.tool === "compare_products") {
          const summaryDiv = node.querySelector(".node-summary");
          if (summaryDiv) {
            summaryDiv.textContent = "Comparing frequency, capacity, CAS latency and voltage";
          }
        }
        // Append data preview in details if provided
        if (event.data) {
          let detailsEl = node.querySelector(".node-details");
          if (!detailsEl) {
            detailsEl = document.createElement("details");
            detailsEl.className = "node-details";
            detailsEl.innerHTML = `<summary>Inspect Output Data</summary><pre>${escapeHtml(JSON.stringify(event.data, null, 2))}</pre>`;
            node.appendChild(detailsEl);
          } else {
            const pre = document.createElement("pre");
            pre.textContent = JSON.stringify(event.data, null, 2);
            detailsEl.appendChild(pre);
          }
        }
      }
    } else if (type === "deterministic_ranking") {
      addTraceNode({
        icon: "⚙️",
        title: event.title || "Deterministic ranking",
        summary: event.summary || `Evaluated ${event.evaluated_count || 22} products\nSelected top ${event.selected_count || 5} candidates`,
        args: event.data || (event.best_value ? { best_value_product: event.best_value, cheapest_product: event.cheapest } : null),
      });
    } else if (type === "agent_thought") {
      addTraceNode({
        icon: "🧠",
        title: event.title || "Agent decision",
        summary: event.message,
      });
    } else if (type === "synthesis_start") {
      addTraceNode({
        icon: "📝",
        title: event.title || "Final synthesis",
        summary: event.message || "Generating recommendation",
      });
    } else if (type === "warning") {
      addTraceNode({
        icon: "⚠️",
        title: "Rate Limit Pacing",
        stepTag: "Warning",
        summary: event.message,
      });
    } else if (type === "final_response") {
      // Render Final Results
      renderFinalResults(event);
    } else if (type === "error") {
      addTraceNode({
        icon: "❌",
        title: "Execution Error",
        summary: event.error,
        isError: true,
      });
    }
  }

  // Helper to add node to trace timeline DOM
  function addTraceNode({
    icon,
    title,
    stepTag,
    summary,
    considerations,
    toolSubName,
    formattedParams,
    isActive,
    args,
    isError,
  }) {
    const node = document.createElement("div");
    node.className = `trace-node ${isActive ? "active-node" : ""}`;
    if (isError) node.style.borderColor = "var(--color-danger)";

    let detailsHtml = "";
    if (args) {
      detailsHtml = `
        <details class="node-details">
          <summary>Inspect Parameters / Data</summary>
          <pre>${escapeHtml(JSON.stringify(args, null, 2))}</pre>
        </details>
      `;
    }

    let considerationsHtml = "";
    if (considerations && Object.keys(considerations).length > 0) {
      const items = Object.entries(considerations)
        .map(
          ([k, v]) => `
          <div class="cons-line">
            <span class="cons-k">${escapeHtml(k)}:</span>
            <span class="cons-v">${escapeHtml(String(v))}</span>
          </div>`
        )
        .join("");
      considerationsHtml = `<div class="trace-considerations">${items}</div>`;
    }

    let paramsHtml = "";
    if (toolSubName || (formattedParams && Object.keys(formattedParams).length > 0)) {
      let paramLines = "";
      if (formattedParams) {
        paramLines = Object.entries(formattedParams)
          .map(
            ([k, v]) => `
            <div class="param-line">
              <span class="param-k">${escapeHtml(k)}:</span>
              <span class="param-v">${escapeHtml(String(v))}</span>
            </div>`
          )
          .join("");
      }
      paramsHtml = `
        <div class="trace-params-block">
          ${toolSubName ? `<div class="tool-sub-name">${escapeHtml(toolSubName)}</div>` : ""}
          ${paramLines}
        </div>
      `;
    }

    let summaryHtml = "";
    if (summary) {
      const lines = summary.split("\n");
      if (lines.length > 1) {
        summaryHtml = `
          <div class="node-summary multiline-summary">
            ${lines.map((l) => `<div>${escapeHtml(l)}</div>`).join("")}
          </div>
        `;
      } else {
        summaryHtml = `<div class="node-summary">${escapeHtml(summary)}</div>`;
      }
    }

    node.innerHTML = `
      <div class="trace-bullet">${icon}</div>
      <div class="node-header">
        <div class="node-title-group">
          <span class="node-tool-name">${escapeHtml(title)}</span>
          ${stepTag ? `<span class="node-step-tag">${escapeHtml(stepTag)}</span>` : ""}
        </div>
        <div class="node-meta">
          <span class="node-status-badge">${isActive ? "● Executing..." : ""}</span>
        </div>
      </div>
      ${summaryHtml}
      ${considerationsHtml}
      ${paramsHtml}
      ${detailsHtml}
    `;

    traceTimeline.appendChild(node);
    // Smooth scroll to bottom of trace
    traceTimeline.scrollTop = traceTimeline.scrollHeight;
    return node;
  }
}

// -------------------------------------------------------------
// 5. Render Final Results (Markdown, Cards, Comparison, JSON)
// -------------------------------------------------------------
function renderFinalResults(data) {
  const resultsSection = document.getElementById("resultsSection");
  const markdownContainer = document.getElementById("recommendationContent");
  const productsGrid = document.getElementById("productsGrid");
  const productsCountBadge = document.getElementById("productsCountBadge");
  const comparisonContainer = document.getElementById("comparisonContainer");
  const rawJsonContent = document.getElementById("rawJsonContent");

  resultsSection.style.display = "block";

  // 1. Render Markdown
  if (data.final_answer) {
    if (window.marked) {
      markdownContainer.innerHTML = marked.parse(data.final_answer);
    } else {
      markdownContainer.textContent = data.final_answer;
    }
  }

  // 2. Render Product Cards Grid
  const products = data.products || [];
  productsCountBadge.textContent = products.length;
  productsGrid.innerHTML = "";

  if (products.length === 0) {
    productsGrid.innerHTML = `<p style="color: var(--text-muted); grid-column: 1/-1;">No product records available.</p>`;
  } else {
    products.forEach((prod) => {
      const card = document.createElement("div");
      card.className = "product-card";

      const formFactor = prod.form_factor || "UDIMM";
      const isLaptop = formFactor.toUpperCase().includes("SODIMM");
      const badgeClass = isLaptop ? "badge-sodimm" : "badge-udimm";
      const valueScore = prod.value_score != null ? `${prod.value_score}/100` : "N/A";
      const priceFmt = prod.price ? `₹${Number(prod.price).toLocaleString("en-IN")}` : "Check Price";
      const pricePerGb = prod.price_per_gb ? `(₹${prod.price_per_gb}/GB)` : "";

      card.innerHTML = `
        <div>
          <div class="card-top">
            <span class="card-badge ${badgeClass}">${escapeHtml(formFactor)}</span>
            <span class="score-badge">Score: ${escapeHtml(valueScore)}</span>
          </div>
          <h3 class="card-title" title="${escapeHtml(prod.name)}">${escapeHtml(prod.name)}</h3>
          
          <div class="card-specs">
            <div class="spec-item">Capacity: <span>${prod.capacity_gb ? prod.capacity_gb + "GB" : "N/A"}</span></div>
            <div class="spec-item">Kit: <span>${escapeHtml(prod.kit_size || "1x" + (prod.capacity_gb || "") + "GB")}</span></div>
            <div class="spec-item">Speed: <span>${prod.speed_mhz ? prod.speed_mhz + "MHz" : "N/A"}</span></div>
            <div class="spec-item">Latency: <span>${prod.cl_latency ? "CL" + prod.cl_latency : "N/A"}</span></div>
          </div>
        </div>

        <div class="card-bottom">
          <div class="card-price">
            <span class="price-inr">${priceFmt}</span>
            <span class="price-per-gb">${pricePerGb}</span>
          </div>
          <a href="${prod.url}" target="_blank" rel="noopener noreferrer" class="card-link-btn">
            View on Amazon ↗
          </a>
        </div>
      `;
      productsGrid.appendChild(card);
    });
  }

  // 3. Render Head-to-Head Comparison Table
  comparisonContainer.innerHTML = "";
  if (data.comparison && data.comparison.comparison_table) {
    const compTable = data.comparison.comparison_table;
    const verdict = data.comparison.verdict || {};

    let verdictHtml = "";
    if (verdict.best_value) {
      verdictHtml = `
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid var(--color-success); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
          <h4 style="color: var(--color-success); font-size: 0.95rem; margin-bottom: 0.35rem;">🏆 Best Value Verdict</h4>
          <p style="font-size: 0.85rem; color: #f1f5f9;"><strong>${escapeHtml(verdict.best_value.name || "N/A")}</strong> (Score: ${verdict.best_value.value_score || "N/A"}, Price: ₹${verdict.best_value.price || "N/A"})</p>
          <span style="font-size: 0.78rem; color: var(--text-secondary);">${escapeHtml(verdict.best_value.reason || "")}</span>
        </div>
      `;
    }

    let rowsHtml = compTable
      .map((item) => `
        <tr>
          <td><strong>${escapeHtml(item.name.substring(0, 60))}...</strong></td>
          <td><span class="score-badge">${item.value_score || "N/A"}</span></td>
          <td style="color: var(--color-success); font-weight: 600;">₹${Number(item.price).toLocaleString("en-IN")}</td>
          <td>${item.capacity_gb || "N/A"}GB</td>
          <td>${item.speed_mhz || "N/A"}MHz</td>
          <td>${item.cl_latency ? "CL" + item.cl_latency : "N/A"}</td>
          <td>${escapeHtml(item.voltage || "1.25V")}</td>
          <td>${escapeHtml(item.warranty || "Manufacturer")}</td>
          <td>
            <a href="${item.url}" target="_blank" rel="noopener noreferrer" class="card-link-btn" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;">
              Amazon ↗
            </a>
          </td>
        </tr>
      `)
      .join("");

    comparisonContainer.innerHTML = `
      ${verdictHtml}
      <div style="overflow-x: auto;">
        <table class="markdown-body" style="width: 100%;">
          <thead>
            <tr>
              <th>Product</th>
              <th>Score</th>
              <th>Price</th>
              <th>Capacity</th>
              <th>Speed</th>
              <th>Latency</th>
              <th>Voltage</th>
              <th>Warranty</th>
              <th>Link</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
    `;
  } else {
    comparisonContainer.innerHTML = `<p style="color: var(--text-muted); padding: 1rem;">No head-to-head comparison generated for this query.</p>`;
  }

  // 4. Render Raw JSON
  rawJsonContent.textContent = JSON.stringify(data, null, 2);

  // Scroll results into view smoothly
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// -------------------------------------------------------------
// 6. Tab 2: Web Scraper & RAG QA Forms
// -------------------------------------------------------------
function initRagForms() {
  const scrapeForm = document.getElementById("scrapeForm");
  const scrapeBtn = document.getElementById("scrapeBtn");
  const scrapeResult = document.getElementById("scrapeResult");

  scrapeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = document.getElementById("scrapeUrlInput").value.trim();
    const force_refresh = document.getElementById("forceRefreshCheckbox").checked;

    scrapeBtn.disabled = true;
    scrapeBtn.textContent = "Ingesting Webpage...";
    scrapeResult.style.display = "block";
    scrapeResult.innerHTML = `<span style="color: var(--color-cyan);">Fetching, cleaning, and storing chunks into ChromaDB...</span>`;

    try {
      const res = await fetch("/api/scrape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, force_refresh }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || data.detail || "Ingestion failed");

      scrapeResult.innerHTML = `
        <div style="color: var(--color-success); font-weight: 600; margin-bottom: 0.5rem;">✓ Webpage Ingested Successfully!</div>
        <p><strong>Title:</strong> ${escapeHtml(data.title)}</p>
        <p><strong>Chunks Indexed:</strong> ${data.chunks_indexed}</p>
        <p><strong>Characters Cleaned:</strong> ${data.char_count || "N/A"}</p>
        <details style="margin-top: 0.5rem;">
          <summary style="cursor: pointer; color: var(--text-muted);">View Sample Chunk</summary>
          <pre style="margin-top: 0.5rem; font-size: 0.75rem; color: #cbd5e1; max-height: 120px; overflow-y: auto;">${escapeHtml(data.sample_chunk || "")}</pre>
        </details>
      `;
      // Auto-fill query URL in section 2
      document.getElementById("queryUrlInput").value = url;
    } catch (err) {
      scrapeResult.innerHTML = `<div style="color: var(--color-danger); font-weight: 600;">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      scrapeBtn.disabled = false;
      scrapeBtn.textContent = "📥 Scrape & Store in ChromaDB";
    }
  });

  const queryForm = document.getElementById("queryForm");
  const queryBtn = document.getElementById("queryBtn");
  const queryResult = document.getElementById("queryResult");

  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = document.getElementById("queryUrlInput").value.trim();
    const query = document.getElementById("queryQuestionInput").value.trim();

    queryBtn.disabled = true;
    queryBtn.textContent = "Retrieving & Asking LLM...";
    queryResult.style.display = "block";
    queryResult.innerHTML = `<span style="color: var(--color-cyan);">Searching ChromaDB top-k semantic chunks and synthesizing answer...</span>`;

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, query }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || data.detail || "QA failed");

      const answerHtml = window.marked ? marked.parse(data.answer) : escapeHtml(data.answer);
      let sourcesHtml = "";
      if (data.sources && data.sources.length) {
        sourcesHtml = `
          <div style="margin-top: 1rem; border-top: 1px solid var(--border-color); padding-top: 0.75rem;">
            <strong style="font-size: 0.8rem; color: var(--text-secondary);">Top Retrieved Context Chunks (${data.sources.length}):</strong>
            ${data.sources
              .map(
                (s, i) => `
                <div style="background: #111827; border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem; margin-top: 0.4rem; font-size: 0.75rem;">
                  <span style="color: var(--color-cyan); font-weight: 600;">Chunk #${s.metadata ? s.metadata.chunk_id : i} (Score: ${s.score || "N/A"})</span>
                  <p style="color: #cbd5e1; margin-top: 0.25rem;">${escapeHtml(s.content.substring(0, 200))}...</p>
                </div>
              `
              )
              .join("")}
          </div>
        `;
      }

      queryResult.innerHTML = `
        <div style="font-size: 0.95rem; line-height: 1.6;">${answerHtml}</div>
        ${sourcesHtml}
      `;
    } catch (err) {
      queryResult.innerHTML = `<div style="color: var(--color-danger); font-weight: 600;">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      queryBtn.disabled = false;
      queryBtn.textContent = "❓ Ask QA Pipeline";
    }
  });
}

// -------------------------------------------------------------
// 7. Utility Functions
// -------------------------------------------------------------
function initCopyButton() {
  const copyBtn = document.getElementById("copyRecommendationBtn");
  if (!copyBtn) return;

  copyBtn.addEventListener("click", () => {
    const content = document.getElementById("recommendationContent").innerText;
    if (!content) return;

    navigator.clipboard.writeText(content).then(() => {
      const originalText = copyBtn.textContent;
      copyBtn.textContent = "✓ Copied!";
      setTimeout(() => {
        copyBtn.textContent = originalText;
      }, 2000);
    });
  });
}

function escapeHtml(text) {
  if (text == null) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}
