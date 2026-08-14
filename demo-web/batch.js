const batchElements = {
  singleModeButton: document.querySelector("#singleModeButton"),
  batchModeButton: document.querySelector("#batchModeButton"),
  singleControls: document.querySelector("#singleControls"),
  batchControls: document.querySelector("#batchControls"),
  singleStage: document.querySelector("#singleStage"),
  batchStage: document.querySelector("#batchStage"),
  datasetSelect: document.querySelector("#batchDatasetSelect"),
  runButton: document.querySelector("#batchRunButton"),
  title: document.querySelector("#batchTitle"),
  source: document.querySelector("#batchSource"),
  progressText: document.querySelector("#batchProgressText"),
  progressBar: document.querySelector("#batchProgressBar"),
  runMeta: document.querySelector("#batchRunMeta"),
  metrics: document.querySelector("#batchMetrics"),
  rowCount: document.querySelector("#batchRowCount"),
  tableBody: document.querySelector("#batchTableBody"),
  labelFilter: document.querySelector("#labelFilter"),
  outcomeFilter: document.querySelector("#outcomeFilter"),
  search: document.querySelector("#caseSearch"),
  previousPage: document.querySelector("#previousPage"),
  nextPage: document.querySelector("#nextPage"),
  pageIndicator: document.querySelector("#pageIndicator"),
  detailCaseId: document.querySelector("#detailCaseId"),
  detailOutcome: document.querySelector("#detailOutcome"),
  detailContent: document.querySelector("#caseDetailContent"),
  executionMode: document.querySelector("#executionMode"),
  runId: document.querySelector("#runId"),
};

let batchResult = null;
let filteredRows = [];
let selectedCaseId = "";
let currentPage = 1;
let activeJobId = "";
const pageSize = 10;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sleep(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

function setMode(mode) {
  const batchMode = mode === "batch";
  batchElements.singleModeButton.classList.toggle("is-active", !batchMode);
  batchElements.batchModeButton.classList.toggle("is-active", batchMode);
  batchElements.singleControls.classList.toggle("is-hidden", batchMode);
  batchElements.batchControls.classList.toggle("is-hidden", !batchMode);
  batchElements.singleStage.classList.toggle("is-hidden", batchMode);
  batchElements.batchStage.classList.toggle("is-hidden", !batchMode);
  if (batchMode) {
    batchElements.executionMode.textContent = batchResult ? "REAL BATCH RUN" : "批量模块已就绪";
    batchElements.runId.textContent = batchResult?.run_id || "BATCH-PENDING";
  } else {
    document.querySelector("#resetButton").click();
  }
}

async function loadDatasets() {
  try {
    const response = await fetch("/api/batch/datasets");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取数据集");
    batchElements.datasetSelect.innerHTML = payload.datasets.map(dataset =>
      `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.label)}</option>`
    ).join("");
    batchElements.runMeta.textContent = `SEED ${payload.seed}`;
  } catch (error) {
    batchElements.runMeta.textContent = "数据集接口不可用";
    batchElements.runButton.disabled = true;
  }
}

function renderMetrics() {
  if (!batchResult) return;
  batchElements.metrics.innerHTML = batchResult.cards.map(card => `
    <div class="batch-metric">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.live)}</strong>
      <div class="metric-reference">
        <span>报告 ${escapeHtml(card.reference)}</span>
        <b class="${card.delta === "一致" ? "" : "has-delta"}">${escapeHtml(card.delta)}</b>
      </div>
    </div>
  `).join("");
}

function matchesFilters(row) {
  const label = batchElements.labelFilter.value;
  const outcome = batchElements.outcomeFilter.value;
  const query = batchElements.search.value.trim().toLowerCase();
  if (label !== "all" && row.label !== label) return false;
  if (outcome === "errors" && !row.isError) return false;
  if (outcome === "blocked" && !String(row.protected).includes("阻断")) return false;
  if (query && !`${row.id} ${row.category}`.toLowerCase().includes(query)) return false;
  return true;
}

function outcomeClass(row) {
  if (row.isError) return "warn";
  if (row.label === "attack") return "safe";
  return "safe";
}

function applyFilters(resetPage = true) {
  if (!batchResult) return;
  filteredRows = batchResult.rows.filter(matchesFilters);
  if (resetPage) currentPage = 1;
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  currentPage = Math.min(currentPage, pageCount);
  renderTable();
}

function renderTable() {
  const pageCount = filteredRows.length ? Math.ceil(filteredRows.length / pageSize) : 0;
  const visible = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  batchElements.rowCount.textContent = `${filteredRows.length} / ${batchResult?.sampleCount || 0} 条`;
  batchElements.pageIndicator.textContent = `${pageCount ? currentPage : 0} / ${pageCount}`;
  batchElements.previousPage.disabled = currentPage <= 1;
  batchElements.nextPage.disabled = currentPage >= pageCount;
  if (!visible.length) {
    batchElements.tableBody.innerHTML = `<tr><td colspan="7" class="table-empty">当前筛选条件没有样本</td></tr>`;
    return;
  }
  batchElements.tableBody.innerHTML = visible.map(row => `
    <tr data-case-id="${escapeHtml(row.id)}" class="${row.id === selectedCaseId ? "is-selected" : ""}">
      <td title="${escapeHtml(row.id)}">${escapeHtml(row.id)}</td>
      <td><span class="case-label ${escapeHtml(row.label)}">${row.label === "attack" ? "攻击" : "良性"}</span></td>
      <td title="${escapeHtml(row.category)}">${escapeHtml(row.category)}</td>
      <td title="${escapeHtml(row.baseline)}">${escapeHtml(row.baseline)}</td>
      <td><span class="case-outcome ${outcomeClass(row)}">${escapeHtml(row.protected)}</span></td>
      <td title="${escapeHtml(row.blockedStage)}">${escapeHtml(row.blockedStage)}</td>
      <td>${Number(row.elapsedMs).toFixed(Number(row.elapsedMs) < 0.01 ? 4 : 2)} ms</td>
    </tr>
  `).join("");
  batchElements.tableBody.querySelectorAll("tr[data-case-id]").forEach(row => {
    row.addEventListener("click", () => selectCase(row.dataset.caseId));
  });
}

function detailJson(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function selectCase(caseId) {
  const row = batchResult?.rows.find(item => item.id === caseId);
  if (!row) return;
  selectedCaseId = caseId;
  renderTable();
  batchElements.detailCaseId.textContent = row.id;
  batchElements.detailOutcome.textContent = row.protected;
  batchElements.detailOutcome.style.color = row.isError ? "var(--amber)" : "var(--green)";
  const pipelineReview = row.pipelineReview ? `
    <div class="detail-section">
      <strong>生产流水线复核</strong>
      <pre>${detailJson(row.pipelineReview)}</pre>
    </div>` : "";
  batchElements.detailContent.innerHTML = `
    <div class="detail-section">
      <strong>样本信息</strong>
      <div class="detail-facts">
        <div class="detail-fact"><span>预期标签</span><strong>${row.label === "attack" ? "攻击" : "良性"}</strong></div>
        <div class="detail-fact"><span>攻击类型</span><strong>${escapeHtml(row.category)}</strong></div>
        <div class="detail-fact"><span>无防护</span><strong>${escapeHtml(row.baseline)}</strong></div>
        <div class="detail-fact"><span>AgentArmor</span><strong>${escapeHtml(row.protected)}</strong></div>
        <div class="detail-fact"><span>阻断阶段</span><strong>${escapeHtml(row.blockedStage)}</strong></div>
        <div class="detail-fact"><span>本次耗时</span><strong>${escapeHtml(row.elapsedMs)} ms</strong></div>
      </div>
    </div>
    <div class="detail-section"><strong>原始测试内容</strong><p>${escapeHtml(row.input)}</p></div>
    <div class="detail-section"><strong>数据来源</strong><p>${escapeHtml(row.source)}</p></div>
    <div class="detail-section"><strong>无防护原始结果</strong><pre>${detailJson(row.baselineDetail)}</pre></div>
    <div class="detail-section"><strong>AgentArmor 原始结果</strong><pre>${detailJson(row.protectedDetail)}</pre></div>
    ${pipelineReview}
    <div class="detail-section"><strong>本批次实际调用</strong><p>${escapeHtml(batchResult.moduleCalls.join("\n"))}</p></div>
    <div class="detail-section"><strong>证据目录</strong><p>${escapeHtml(batchResult.artifactDir)}</p></div>
  `;
}

function acceptBatchResult(result) {
  if (result.real_execution !== true) throw new Error("服务端未确认真实模块执行");
  batchResult = result;
  selectedCaseId = result.rows[0]?.id || "";
  currentPage = 1;
  batchElements.title.textContent = result.title;
  batchElements.source.textContent = `${result.sourceLabel}；报告参照：${result.referenceSource}`;
  batchElements.runMeta.textContent = `${result.run_id} · ${result.executionCount} 次模块执行`;
  batchElements.progressText.textContent = `完成 ${result.sampleCount} / ${result.sampleCount}`;
  batchElements.progressBar.style.width = "100%";
  batchElements.executionMode.textContent = "REAL BATCH RUN";
  batchElements.runId.textContent = result.run_id;
  renderMetrics();
  applyFilters();
  if (selectedCaseId) selectCase(selectedCaseId);
}

async function pollJob(jobId) {
  while (activeJobId === jobId) {
    const response = await fetch(`/api/batch/status?job_id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "批量任务状态读取失败");
    const percent = job.total ? (job.completed / job.total) * 100 : 0;
    batchElements.progressBar.style.width = `${percent}%`;
    batchElements.progressText.textContent = `${job.status === "queued" ? "排队" : "执行"} ${job.completed} / ${job.total}`;
    batchElements.runMeta.textContent = `${job.job_id} · ${job.current}`;
    if (job.status === "complete") {
      acceptBatchResult(job.result);
      return;
    }
    if (job.status === "error") throw new Error(job.error || job.current || "批量任务失败");
    await sleep(120);
  }
}

async function startBatch() {
  batchElements.runButton.disabled = true;
  batchElements.datasetSelect.disabled = true;
  batchElements.runButton.textContent = "真实模块运行中…";
  batchElements.progressBar.style.width = "0%";
  batchElements.progressText.textContent = "创建批量任务…";
  batchElements.executionMode.textContent = "正在运行批量源码…";
  batchElements.metrics.innerHTML = `<div class="batch-empty">正在逐条调用项目测试链路，请稍候</div>`;
  try {
    const response = await fetch("/api/batch/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: batchElements.datasetSelect.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法启动批量任务");
    activeJobId = payload.job_id;
    await pollJob(activeJobId);
  } catch (error) {
    batchElements.progressText.textContent = "批量任务失败";
    batchElements.executionMode.textContent = "批量模块运行失败";
    batchElements.metrics.innerHTML = `<div class="batch-empty">${escapeHtml(error.message)}</div>`;
  } finally {
    batchElements.runButton.disabled = false;
    batchElements.datasetSelect.disabled = false;
    batchElements.runButton.textContent = "开始真实批量测试";
  }
}

batchElements.singleModeButton.addEventListener("click", () => setMode("single"));
batchElements.batchModeButton.addEventListener("click", () => setMode("batch"));
batchElements.runButton.addEventListener("click", startBatch);
batchElements.labelFilter.addEventListener("change", () => applyFilters());
batchElements.outcomeFilter.addEventListener("change", () => applyFilters());
batchElements.search.addEventListener("input", () => applyFilters());
batchElements.previousPage.addEventListener("click", () => { currentPage -= 1; renderTable(); });
batchElements.nextPage.addEventListener("click", () => { currentPage += 1; renderTable(); });

loadDatasets();
