/* Classification Lab: a small, transparent k-nearest-neighbors learning app.
   All records and model calculations live in this file; there is no API. */

const icon = (name) => {
  const paths = {
    grid: '<rect x="3" y="3" width="7" height="7" rx="1.4"/><rect x="14" y="3" width="7" height="7" rx="1.4"/><rect x="3" y="14" width="7" height="7" rx="1.4"/><rect x="14" y="14" width="7" height="7" rx="1.4"/>',
    table: '<path d="M4 5h16M4 12h16M4 19h16M10 5v14m5-14v14"/>',
    sliders: '<path d="M4 7h9m4 0h3M4 17h3m4 0h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    chart: '<path d="M4 19V5m0 14h17"/><path d="m7 15 4-4 3 2 6-7"/>',
    spark: '<path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/>',
    book: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 1 4 17.5z"/><path d="M4 16a2.5 2.5 0 0 1 2.5-2.5H20M8 7h7"/>',
    users: '<path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    search: '<circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4 4"/>',
    trash: '<path d="M4 7h16m-10 4v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3"/>',
    arrow: '<path d="M7 17 17 7M7 7h10v10"/>',
    brain: '<path d="M12 18V5"/><path d="M15.5 6.5a3.5 3.5 0 0 1 2 6.4 3.5 3.5 0 0 1-1.5 6.6H12"/><path d="M8.5 6.5a3.5 3.5 0 0 0-2 6.4A3.5 3.5 0 0 0 8 19.5h4"/><path d="M8.5 6.5A3.5 3.5 0 0 1 12 3v16"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
  };
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`;
};

const STORAGE = { records: "classification-lab-records-v1", k: "classification-lab-k-v1", recent: "classification-lab-recents-v1" };
const OUTCOMES = ["Needs Support", "Pass"];
const FEATURE_KEYS = ["hours", "attendance", "score"];
const POSITIVE = "Needs Support"; // The positive class is fixed throughout all metrics.
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const round = (value, places = 1) => Number(value.toFixed(places));

function seededRandom(seed) {
  let current = seed >>> 0;
  return () => {
    current = (current * 1664525 + 1013904223) >>> 0;
    return current / 4294967296;
  };
}

// Build a stable, balanced sample set that remains clearly synthetic.
function makeSampleRecords() {
  const random = seededRandom(812734);
  const rows = [];
  for (let i = 0; i < 48; i += 1) {
    const hours = round(0.5 + random() * 2.5, 1);
    rows.push({ id: `sample-s-${String(i + 1).padStart(3, "0")}`, hours, attendance: Math.round(60 + random() * 30), score: Math.round(42 + random() * 34), outcome: "Needs Support", source: "sample" });
  }
  for (let i = 0; i < 48; i += 1) {
    const hours = round(1.8 + random() * 3.2, 1);
    rows.push({ id: `sample-o-${String(i + 1).padStart(3, "0")}`, hours, attendance: Math.round(76 + random() * 24), score: Math.round(58 + random() * 40), outcome: "Pass", source: "sample" });
  }
  return rows;
}

function readStored(key, fallback, validate) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key));
    return validate(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

let records = readStored(STORAGE.records, makeSampleRecords(), (items) =>
  Array.isArray(items) && items.every((r) => r && typeof r.id === "string" && FEATURE_KEYS.every((key) => Number.isFinite(r[key])) && OUTCOMES.includes(r.outcome))
);
let kSetting = Number(readStored(STORAGE.k, 5, (v) => Number.isInteger(v) && v >= 1 && v <= 99)) || 5;
let recents = readStored(STORAGE.recent, [], (items) => Array.isArray(items));
let model = null;
let training = false;
let sampleChart = null;
let evalChart = null;
let toastTimer;

const app = document.querySelector("#app");

app.innerHTML = `
  <div class="shell">
    <aside class="sidebar" aria-label="Main navigation">
      <div class="brand"><div class="brand-mark">${icon("brain")}</div><div class="brand-name">Classification Lab<span class="brand-sub">AI &amp; machine learning</span></div></div>
      <div class="mobile-brand"><div class="brand-mark">${icon("brain")}</div><div class="brand-name">Classification Lab<span class="brand-sub">learning studio</span></div></div>
      <div class="nav-caption">Workspace</div>
      <ul class="nav-list">
        <li><a class="nav-link active" href="#dashboard" data-section="dashboard">${icon("grid")}<span>Dashboard</span></a></li>
        <li><a class="nav-link" href="#dataset" data-section="dataset">${icon("table")}<span>Dataset</span></a></li>
        <li><a class="nav-link" href="#train" data-section="train">${icon("sliders")}<span>Train model</span></a></li>
        <li><a class="nav-link" href="#evaluation" data-section="evaluation">${icon("chart")}<span>Evaluation</span></a></li>
        <li><a class="nav-link" href="#prediction" data-section="prediction">${icon("spark")}<span>Prediction</span></a></li>
      </ul>
      <div class="sidebar-note"><span class="note-icon">${icon("book")}</span><strong>Learn by looking closer</strong><p>Follow real calculations on a small, transparent sample. Every vote is explainable.</p></div>
      <div class="sidebar-foot"><span class="foot-dot"></span>Runs locally in your browser</div>
    </aside>
    <main class="main">
      <header class="topbar"><div class="crumb">Learning workspace <span aria-hidden="true"> / </span><b>Classification lab</b></div><div class="topbar-right"><span class="status-pill"><span class="live-dot"></span>Local sample data</span><span class="avatar" aria-label="Learning workspace">CL</span></div></header>
      <div class="page">
        <section id="dashboard" class="hero" aria-labelledby="welcome-title">
          <div class="hero-copy"><span class="eyebrow"><span class="live-dot"></span> A practical introduction to classification</span><h1 id="welcome-title">Make the model make sense.</h1><p>Explore how study habits connect to learning support, then train a simple k-NN model and inspect every prediction it makes.</p></div>
          <div class="hero-action"><button class="button button-white" data-scroll="train">${icon("arrow")} Start a training run</button></div>
        </section>
        <section aria-labelledby="overview-title">
          <div class="section-heading"><div><span class="section-kicker">At a glance</span><h2 id="overview-title">Your learning workspace</h2></div></div>
          <div class="stats-grid">
            <article class="stat-card"><div class="stat-top"><span>Dataset records</span><span class="stat-icon purple">${icon("users")}</span></div><div class="stat-value" id="stat-records">—</div><div class="stat-foot">Synthetic learning examples</div></article>
            <article class="stat-card"><div class="stat-top"><span>Model accuracy</span><span class="stat-icon teal">${icon("check")}</span></div><div class="stat-value" id="stat-accuracy">—</div><div class="stat-foot" id="stat-accuracy-foot">Train the model to evaluate it</div></article>
            <article class="stat-card"><div class="stat-top"><span>Chosen algorithm</span><span class="stat-icon blue">${icon("layers")}</span></div><div class="stat-value" style="font-size:20px">k-NN</div><div class="stat-foot">Euclidean distance · normalized</div></article>
            <article class="stat-card"><div class="stat-top"><span>Target classes</span><span class="stat-icon coral">${icon("book")}</span></div><div class="stat-value">2</div><div class="stat-foot">Needs Support · Pass</div></article>
          </div>
        </section>
        <section class="dashboard-grid" aria-label="Dataset patterns and latest predictions">
          <article class="panel"><div class="panel-head"><div><h3>Study patterns by outcome</h3><p>Average study hours per day in the current dataset</p></div><span class="stat-icon purple">${icon("chart")}</span></div><div class="chart-wrap" id="sample-chart-wrap"><canvas id="sample-chart" aria-label="Bar chart comparing average study time by outcome"></canvas><p id="sample-chart-error" class="chart-error" role="alert"></p></div><div class="chart-legend"><span class="legend-item"><i class="legend-dot" style="background:#d88a6b"></i>Needs Support</span><span class="legend-item"><i class="legend-dot" style="background:#35a59c"></i>Pass</span></div></article>
          <article class="panel"><div class="panel-head"><div><h3>Pattern snapshot</h3><p>Share of records in each outcome class</p></div></div><div class="patterns" id="pattern-bars"></div><div class="pattern-insight" id="pattern-insight">${icon("info")}<span></span></div></article>
        </section>
        <section class="section" aria-labelledby="recent-title"><div class="section-heading"><div><span class="section-kicker">Your activity</span><h2 id="recent-title">Recent predictions</h2><p>New predictions appear here and stay on this device.</p></div><button class="button button-outline button-small" id="clear-recents">${icon("trash")} Clear history</button></div><div class="panel recent-wrap"><table><thead><tr><th>Study / day</th><th>Attendance</th><th>Previous score</th><th>Predicted class</th><th>Vote share</th></tr></thead><tbody id="recent-rows"></tbody></table></div></section>

        <section id="dataset" class="section" aria-labelledby="dataset-title">
          <div class="section-heading"><div><span class="section-kicker">01 / Inspect the examples</span><h2 id="dataset-title">Dataset</h2><p>Search the records, add your own example, or remove one to explore its effect.</p></div></div>
          <div class="dataset-notice">${icon("info")}<div><strong>Educational sample data only.</strong> All included records are synthetic examples generated for learning; they are not real student records and should not be used to make decisions about people.</div></div>
          <div class="dataset-toolbar"><label class="searchbox">${icon("search")}<input id="dataset-search" type="search" placeholder="Search by class or study hours…" aria-label="Search dataset records" data-testid="input-dataset-search"></label><span class="toolbar-count" id="dataset-count"></span><button id="toggle-add-form" class="button button-primary">${icon("plus")} Add a record</button></div>
          <form id="add-record-form" class="panel form-panel" hidden novalidate>
            <div class="field"><label for="new-hours">Study hours / day</label><input id="new-hours" name="hours" type="number" min="0" max="24" step="0.1" placeholder="e.g. 2.5" required><small>Enter a value from 0 to 24.</small></div>
            <div class="field"><label for="new-attendance">Attendance (%)</label><input id="new-attendance" name="attendance" type="number" min="0" max="100" step="1" placeholder="e.g. 82" required><small>Enter a percentage from 0 to 100.</small></div>
            <div class="field"><label for="new-score">Previous test score (%)</label><input id="new-score" name="score" type="number" min="0" max="100" step="1" placeholder="e.g. 68" required><small>Enter a percentage from 0 to 100.</small></div>
            <div class="field"><label for="new-outcome">Sample outcome</label><select id="new-outcome" name="outcome"><option>Needs Support</option><option>Pass</option></select><small>For learning only; a labeled example.</small></div>
            <div class="field-action"><button class="button button-primary" type="submit">${icon("plus")} Save sample record</button><span class="form-message" id="add-record-message" role="status"></span></div>
          </form>
          <div class="panel table-panel"><div class="recent-wrap"><table><thead><tr><th>Record</th><th>Study / day</th><th>Attendance</th><th>Previous score</th><th>Outcome</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody id="dataset-rows"></tbody></table></div></div>
        </section>

        <section id="train" class="section" aria-labelledby="train-title">
          <div class="section-heading"><div><span class="section-kicker">02 / Fit a classifier</span><h2 id="train-title">Train model</h2><p>Fit a reproducible nearest-neighbor classifier to the current sample.</p></div></div>
          <div class="training-layout">
            <article class="panel train-card"><h3>k-nearest neighbors</h3><p>For a new example, k-NN finds the most similar training records and predicts the class that appears most often among them.</p><div class="algorithm-chip">${icon("layers")} Euclidean k-NN <span aria-hidden="true">·</span> three normalized features</div><div class="train-controls"><div class="field train-control"><label for="k-value">Neighbors (k)</label><input id="k-value" type="number" min="1" step="1" value="5" aria-describedby="k-help"><small id="k-help">Must be between 1 and the training set size.</small></div><button id="train-button" class="button button-primary">${icon("spark")} Train model</button></div><div id="training-status" class="training-status">${icon("info")}<span></span></div></article>
            <div class="train-side"><article class="panel split-card"><div class="panel-head"><div><h3>Evaluation split</h3><p>Stratified and deterministic</p></div><span class="stat-icon blue">${icon("layers")}</span></div><div class="split-stats"><div class="split-stat"><strong id="train-count">—</strong><span>training records</span></div><div class="split-stat"><strong id="test-count">—</strong><span>test records</span></div></div><p class="split-copy">Each class is split separately: 75% training, 25% held-out test. Normalization is fitted only on the training records. Test examples never become neighbors.</p></article><article class="panel explain-card"><h3>Why normalize features?</h3><p>Hours and percentages use different scales. We standardize each feature with training-set statistics so no single unit dominates distance just because its numbers are larger.</p></article></div>
          </div>
        </section>

        <section id="evaluation" class="section" aria-labelledby="evaluation-title">
          <div class="section-heading"><div><span class="section-kicker">03 / Check the results</span><h2 id="evaluation-title">Evaluation</h2><p>Every metric below comes from predictions on held-out test examples.</p></div><span class="status-pill" id="evaluation-state">Not trained</span></div>
          <div class="evaluation-top"><article class="panel eval-metric"><span>Accuracy</span><strong id="metric-accuracy">—</strong><small class="hint">Correct / all test examples</small></article><article class="panel eval-metric"><span>Precision · Needs Support</span><strong id="metric-precision">—</strong><small class="hint">True positives / predicted positives</small></article><article class="panel eval-metric"><span>Recall · Needs Support</span><strong id="metric-recall">—</strong><small class="hint">True positives / actual positives</small></article><article class="panel eval-metric"><span>F1 · Needs Support</span><strong id="metric-f1">—</strong><small class="hint">Balance of precision and recall</small></article></div>
          <div class="eval-grid"><article class="panel"><div class="panel-head"><div><h3>Actual vs predicted</h3><p>Held-out test set, grouped by actual class</p></div></div><div class="chart-wrap tall"><canvas id="eval-chart" aria-label="Grouped bar chart of actual and predicted class counts"></canvas><p id="eval-chart-error" class="chart-error" role="alert"></p></div></article><article class="panel"><div class="panel-head"><div><h3>Confusion matrix</h3><p>Positive class: Needs Support</p></div></div><div id="confusion-matrix" class="matrix" aria-live="polite"></div></article></div>
        </section>

        <section id="prediction" class="section" aria-labelledby="prediction-title">
          <div class="section-heading"><div><span class="section-kicker">04 / Try a transparent prediction</span><h2 id="prediction-title">Prediction</h2><p>Enter a sample and inspect the neighbors behind its predicted class.</p></div></div>
          <div class="prediction-layout"><form id="prediction-form" class="panel prediction-card" novalidate><h3>New learning example</h3><p>These inputs are evaluated locally. This demonstration does not assess a real student.</p><div class="prediction-fields"><div class="field"><label for="predict-hours">Study hours / day</label><input id="predict-hours" type="number" min="0" max="24" step="0.1" placeholder="0–24" required></div><div class="field"><label for="predict-attendance">Attendance (%)</label><input id="predict-attendance" type="number" min="0" max="100" step="1" placeholder="0–100" required></div><div class="field"><label for="predict-score">Previous test score (%)</label><input id="predict-score" type="number" min="0" max="100" step="1" placeholder="0–100" required></div></div><div class="field-action" style="margin-top:17px"><button class="button button-primary" type="submit">${icon("spark")} Explain prediction</button><span class="form-message" id="prediction-message" role="status"></span></div><div id="prediction-alert" class="prediction-alert" role="alert"></div></form>
            <article class="panel prediction-result" aria-live="polite"><div class="result-placeholder" id="result-placeholder"><span class="placeholder-symbol">${icon("brain")}</span><h3>Ready when you are</h3><p>Train the model first, then make a prediction to see the neighbor vote and plain-language explanation.</p></div><div id="result-content" class="result-content"><span class="section-kicker">Model prediction</span><h3 id="result-outcome"></h3><span id="result-share" class="result-share"></span><div class="vote-track"><div id="vote-support" class="vote-support"></div><div id="vote-on" class="vote-track-on"></div></div><div class="vote-labels"><span id="support-label"></span><span id="on-label"></span></div><div id="reason-box" class="reason-box"></div></div></article>
          </div>
        </section>
        <footer class="app-footer"><span>Classification Lab · Learn the logic, not just the label.</span><span>Synthetic educational examples · No real student records</span></footer>
      </div>
    </main>
  </div>
  <div class="toast" id="toast" role="status" aria-live="polite">${icon("check")}<span></span></div>
`;

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

function persist() {
  try { localStorage.setItem(STORAGE.records, JSON.stringify(records)); }
  catch { showToast("Changes are active for this visit but could not be saved on this device."); }
}

function showToast(message) {
  const toast = $("#toast");
  $("span", toast).textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function splitStratified(data) {
  const rng = seededRandom(19890517);
  const train = [];
  const test = [];
  OUTCOMES.forEach((outcome) => {
    const group = data.filter((record) => record.outcome === outcome);
    // Fisher–Yates with a fixed seed gives a reproducible split per dataset ordering.
    for (let i = group.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      [group[i], group[j]] = [group[j], group[i]];
    }
    const testCount = group.length > 1 ? Math.max(1, Math.floor(group.length * 0.25)) : 0;
    test.push(...group.slice(0, testCount));
    train.push(...group.slice(testCount));
  });
  return { train, test };
}

function fittedScaler(trainingRows) {
  const stats = {};
  FEATURE_KEYS.forEach((key) => {
    const mean = trainingRows.reduce((sum, row) => sum + row[key], 0) / trainingRows.length;
    const variance = trainingRows.reduce((sum, row) => sum + (row[key] - mean) ** 2, 0) / trainingRows.length;
    stats[key] = { mean, std: Math.sqrt(variance) || 1 };
  });
  return stats;
}

function normalize(row, scaler) {
  return FEATURE_KEYS.map((key) => (row[key] - scaler[key].mean) / scaler[key].std);
}

function classify(query, activeModel, k) {
  const normalizedQuery = normalize(query, activeModel.scaler);
  const nearest = activeModel.train.map((record) => {
    const values = normalize(record, activeModel.scaler);
    const distance = Math.sqrt(values.reduce((sum, value, index) => sum + (value - normalizedQuery[index]) ** 2, 0));
    return { record, distance };
  }).sort((a, b) => a.distance - b.distance || a.record.id.localeCompare(b.record.id)).slice(0, k);
  const votes = Object.fromEntries(OUTCOMES.map((outcome) => [outcome, nearest.filter((item) => item.record.outcome === outcome).length]));
  // Ties are resolved consistently in favor of the positive class.
  const predicted = votes[POSITIVE] >= votes.Pass ? POSITIVE : "Pass";
  return { predicted, votes, nearest };
}

function calculateMetrics(activeModel, k) {
  const matrix = Object.fromEntries(OUTCOMES.map((actual) => [actual, Object.fromEntries(OUTCOMES.map((predicted) => [predicted, 0]))]));
  activeModel.test.forEach((record) => {
    const result = classify(record, activeModel, k);
    matrix[record.outcome][result.predicted] += 1;
  });
  const tp = matrix[POSITIVE][POSITIVE];
  const fp = matrix.Pass[POSITIVE];
  const fn = matrix[POSITIVE].Pass;
  const total = activeModel.test.length;
  const accuracy = total ? (tp + matrix.Pass.Pass) / total : 0;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { matrix, accuracy, precision, recall, f1, total };
}

function renderCharts(metrics = null) {
  if (!window.Chart) {
    $("#sample-chart").hidden = true;
    $("#eval-chart").hidden = true;
    $("#sample-chart-error").textContent = "The chart library could not be loaded. Check your connection or allow the public Chart.js CDN, then refresh to view charts. Dataset and model features remain available.";
    $("#eval-chart-error").textContent = "Chart.js is unavailable. The numeric evaluation and confusion matrix are still shown when a model is trained.";
    $("#sample-chart-error").classList.add("visible");
    $("#eval-chart-error").classList.add("visible");
    return;
  }
  $("#sample-chart").hidden = false;
  $("#eval-chart").hidden = false;
  $("#sample-chart-error").classList.remove("visible");
  $("#eval-chart-error").classList.remove("visible");
  const means = OUTCOMES.map((outcome) => {
    const group = records.filter((record) => record.outcome === outcome);
    return group.length ? group.reduce((sum, record) => sum + record.hours, 0) / group.length : 0;
  });
  if (sampleChart) sampleChart.destroy();
  sampleChart = new Chart($("#sample-chart"), {
    type: "bar",
    data: { labels: OUTCOMES, datasets: [{ label: "Average study hours / day", data: means.map((value) => round(value, 2)), backgroundColor: ["#d88a6b", "#35a59c"], borderRadius: 7, borderSkipped: false, barThickness: 34 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => ` ${context.parsed.y.toFixed(1)} hours / day` } } }, scales: { x: { grid: { display: false }, ticks: { color: "#777b90", font: { family: "DM Sans", size: 10 } }, border: { display: false } }, y: { beginAtZero: true, suggestedMax: 6, grid: { color: "#efeff5" }, ticks: { color: "#9295a7", stepSize: 1, font: { family: "IBM Plex Mono", size: 9 } }, title: { display: true, text: "Hours / day", color: "#9295a7", font: { family: "DM Sans", size: 10 } }, border: { display: false } } } }
  });
  if (evalChart) evalChart.destroy();
  const ctx = $("#eval-chart");
  if (!metrics) {
    evalChart = new Chart(ctx, { type: "bar", data: { labels: OUTCOMES, datasets: [{ label: "Actual", data: [null, null], backgroundColor: "#c9c5e9", borderRadius: 5 }, { label: "Predicted", data: [null, null], backgroundColor: "#71c4b6", borderRadius: 5 }] }, options: chartOptions() });
    return;
  }
  const actualCounts = OUTCOMES.map((outcome) => OUTCOMES.reduce((sum, predicted) => sum + metrics.matrix[outcome][predicted], 0));
  const predictedCounts = OUTCOMES.map((outcome) => OUTCOMES.reduce((sum, actual) => sum + metrics.matrix[actual][outcome], 0));
  evalChart = new Chart(ctx, { type: "bar", data: { labels: OUTCOMES, datasets: [{ label: "Actual", data: actualCounts, backgroundColor: "#c9c5e9", borderRadius: 5 }, { label: "Predicted", data: predictedCounts, backgroundColor: "#71c4b6", borderRadius: 5 }] }, options: chartOptions() });
}

function chartOptions() {
  return { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: "#777b90", usePointStyle: true, pointStyle: "circle", boxWidth: 7, padding: 18, font: { family: "DM Sans", size: 10 } } }, tooltip: { mode: "index", intersect: false } }, scales: { x: { grid: { display: false }, ticks: { color: "#777b90", font: { family: "DM Sans", size: 10 } }, border: { display: false } }, y: { beginAtZero: true, suggestedMax: 12, grid: { color: "#efeff5" }, ticks: { precision: 0, color: "#9295a7", font: { family: "IBM Plex Mono", size: 9 } }, border: { display: false } } } };
}

function renderPatternBars() {
  const counts = Object.fromEntries(OUTCOMES.map((outcome) => [outcome, records.filter((record) => record.outcome === outcome).length]));
  const total = records.length;
  $("#pattern-bars").innerHTML = OUTCOMES.map((outcome, index) => {
    const share = total ? counts[outcome] / total * 100 : 0;
    return `<div class="pattern-row"><span>${escapeHtml(outcome)}</span><div class="track" aria-label="${escapeHtml(outcome)} ${share.toFixed(1)} percent"><div class="fill ${index ? "purple" : ""}" style="width:${share}%"></div></div><strong>${share.toFixed(0)}%</strong></div>`;
  }).join("");
  const supportHours = records.filter((record) => record.outcome === POSITIVE);
  const passHours = records.filter((record) => record.outcome === "Pass");
  const avg = (group) => group.length ? group.reduce((sum, record) => sum + record.hours, 0) / group.length : null;
  const sAvg = avg(supportHours);
  const oAvg = avg(passHours);
  let message = "Add records from both classes to compare their study patterns.";
  if (sAvg !== null && oAvg !== null) {
    const delta = Math.abs(sAvg - oAvg);
    message = delta < 0.15 ? `In this sample, average study time is similar across both classes (${sAvg.toFixed(1)} vs ${oAvg.toFixed(1)} hours/day).` :
      `${sAvg < oAvg ? "Needs Support" : "Pass"} examples average ${delta.toFixed(1)} fewer study hours/day in this synthetic sample. A pattern is not proof of cause.`;
  }
  $("#pattern-insight span").textContent = message;
}

function renderRecent() {
  const body = $("#recent-rows");
  if (!recents.length) {
    body.innerHTML = '<tr><td class="empty-row" colspan="5">No predictions yet. Train a model and try a sample input.</td></tr>';
    $("#clear-recents").disabled = true;
    return;
  }
  $("#clear-recents").disabled = false;
  body.innerHTML = recents.slice(0, 5).map((item) => `<tr><td>${Number(item.hours).toFixed(1)} h</td><td>${escapeHtml(item.attendance)}%</td><td>${escapeHtml(item.score)}%</td><td><span class="class-badge ${item.outcome === POSITIVE ? "support" : "track"}">${escapeHtml(item.outcome)}</span></td><td>${escapeHtml(item.share)}</td></tr>`).join("");
}

function renderDataset() {
  const query = $("#dataset-search").value.trim().toLowerCase();
  const filtered = records.filter((record) => [record.id, record.outcome, record.hours, record.attendance, record.score].join(" ").toLowerCase().includes(query));
  $("#dataset-count").textContent = `${filtered.length} of ${records.length} records`;
  const tbody = $("#dataset-rows");
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-row">${records.length ? "No records match your search. Try another term." : "No sample records remain. Add examples from both classes, then train a model."}</td></tr>`;
    return;
  }
  tbody.innerHTML = filtered.map((record) => `<tr><td><span style="font-family:'IBM Plex Mono',monospace;font-size:10px">${escapeHtml(record.id.startsWith("sample") ? `sample · ${record.id.slice(-3)}` : "added · " + record.id.slice(-4))}</span></td><td>${Number(record.hours).toFixed(1)} h</td><td>${record.attendance}%</td><td>${record.score}%</td><td><span class="class-badge ${record.outcome === POSITIVE ? "support" : "track"}">${escapeHtml(record.outcome)}</span></td><td><button class="row-delete" type="button" data-delete="${escapeHtml(record.id)}" aria-label="Delete record ${escapeHtml(record.id)}" title="Delete record">${icon("trash")}</button></td></tr>`).join("");
}

function updateKBounds() {
  const split = splitStratified(records);
  const input = $("#k-value");
  const max = Math.max(1, split.train.length);
  input.max = String(max);
  kSetting = clamp(Number.isInteger(kSetting) ? kSetting : 5, 1, max);
  input.value = String(kSetting);
  try { localStorage.setItem(STORAGE.k, String(kSetting)); } catch { /* local storage is optional */ }
  $("#train-count").textContent = split.train.length;
  $("#test-count").textContent = split.test.length;
}

function renderMatrix(metrics = null) {
  const holder = $("#confusion-matrix");
  if (!metrics) {
    holder.innerHTML = `<div></div><div class="matrix-label matrix-head">Predicted<br>Needs Support</div><div class="matrix-label matrix-head">Predicted<br>Pass</div><div class="matrix-label matrix-head">Actual<br>Needs Support</div><div class="matrix-cell"><strong>—</strong><small>TP</small></div><div class="matrix-cell"><strong>—</strong><small>FN</small></div><div class="matrix-label matrix-head">Actual<br>Pass</div><div class="matrix-cell"><strong>—</strong><small>FP</small></div><div class="matrix-cell"><strong>—</strong><small>TN</small></div>`;
    return;
  }
  holder.innerHTML = `<div></div><div class="matrix-label matrix-head">Predicted<br>Needs Support</div><div class="matrix-label matrix-head">Predicted<br>Pass</div><div class="matrix-label matrix-head">Actual<br>Needs Support</div><div class="matrix-cell diag"><strong>${metrics.matrix[POSITIVE][POSITIVE]}</strong><small>TP</small></div><div class="matrix-cell off"><strong>${metrics.matrix[POSITIVE].Pass}</strong><small>FN</small></div><div class="matrix-label matrix-head">Actual<br>Pass</div><div class="matrix-cell off"><strong>${metrics.matrix.Pass[POSITIVE]}</strong><small>FP</small></div><div class="matrix-cell diag"><strong>${metrics.matrix.Pass.Pass}</strong><small>TN</small></div>`;
}

function renderEvaluation() {
  const usable = model && model.version === datasetVersion;
  const metricIds = ["accuracy", "precision", "recall", "f1"];
  if (!usable) {
    metricIds.forEach((name) => { $(`#metric-${name}`).textContent = "—"; });
    $("#evaluation-state").textContent = "Not trained";
    renderMatrix();
    renderCharts(null);
    return;
  }
  const metrics = calculateMetrics(model, kSetting);
  metricIds.forEach((name) => { $(`#metric-${name}`).textContent = `${(metrics[name] * 100).toFixed(1)}%`; });
  $("#evaluation-state").textContent = `Held-out · ${metrics.total} examples`;
  renderMatrix(metrics);
  renderCharts(metrics);
  $("#stat-accuracy").innerHTML = `${(metrics.accuracy * 100).toFixed(1)}<span class="metric-unit">%</span>`;
  $("#stat-accuracy-foot").textContent = `Held-out test set · ${metrics.total} examples`;
}

let datasetVersion = 0;

function renderAll() {
  $("#stat-records").textContent = records.length;
  if (!model || model.version !== datasetVersion) {
    $("#stat-accuracy").textContent = "—";
    $("#stat-accuracy-foot").textContent = "Train the model to evaluate it";
  }
  renderDataset();
  renderPatternBars();
  renderRecent();
  updateKBounds();
  renderEvaluation();
}

function invalidateModel() {
  datasetVersion += 1;
  model = null;
  $("#training-status").className = "training-status warn";
  $("#training-status span").textContent = "Dataset changed. Train the model again before using metrics or predictions.";
  $("#result-placeholder").style.display = "block";
  $("#result-content").classList.remove("visible");
  renderAll();
}

function validateNumber(input, min, max, step, label) {
  const value = input.value.trim() === "" ? NaN : Number(input.value);
  const offStep = Number.isFinite(value) && step > 0 && Math.abs(value / step - Math.round(value / step)) > 1e-8;
  if (!Number.isFinite(value) || value < min || value > max || offStep) {
    return { error: `${label} must be ${step === 1 ? "a whole number" : `in ${step} increments`} from ${min} to ${max}.` };
  }
  return { value };
}

function showPredictionError(message) {
  const alert = $("#prediction-alert");
  alert.textContent = message;
  alert.classList.add("visible");
}

function hidePredictionError() {
  $("#prediction-alert").classList.remove("visible");
}

$$(".nav-link").forEach((link) => link.addEventListener("click", (event) => {
  event.preventDefault();
  const section = document.getElementById(link.dataset.section);
  section?.scrollIntoView({ behavior: "smooth", block: "start" });
  $$(".nav-link").forEach((item) => item.classList.toggle("active", item === link));
}));

$$("[data-scroll]").forEach((button) => button.addEventListener("click", () => document.getElementById(button.dataset.scroll)?.scrollIntoView({ behavior: "smooth" })));

const observer = new IntersectionObserver((entries) => {
  const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  $$(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.section === visible.target.id));
}, { rootMargin: "-18% 0px -65% 0px", threshold: [0, 0.2, 0.5] });
["dashboard", "dataset", "train", "evaluation", "prediction"].forEach((id) => observer.observe(document.getElementById(id)));

$("#dataset-search").addEventListener("input", renderDataset);
$("#toggle-add-form").addEventListener("click", () => {
  const form = $("#add-record-form");
  form.hidden = !form.hidden;
  if (!form.hidden) $("#new-hours").focus();
});

$("#add-record-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const fields = [
    [$("#new-hours"), 0, 24, 0.1, "Study hours"],
    [$("#new-attendance"), 0, 100, 1, "Attendance"],
    [$("#new-score"), 0, 100, 1, "Previous test score"],
  ];
  const checks = fields.map(([input, min, max, step, label]) => validateNumber(input, min, max, step, label));
  const message = $("#add-record-message");
  const invalid = checks.find((check) => check.error);
  if (invalid) { message.textContent = invalid.error; message.className = "form-message error"; return; }
  const values = checks.map((check) => check.value);
  records.push({ id: `added-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`, hours: values[0], attendance: values[1], score: values[2], outcome: $("#new-outcome").value, source: "added" });
  persist();
  event.currentTarget.reset();
  message.textContent = "Record added. Train again to update evaluation.";
  message.className = "form-message success";
  invalidateModel();
  showToast("Synthetic sample record added.");
});

$("#dataset-rows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete]");
  if (!button) return;
  const record = records.find((item) => item.id === button.dataset.delete);
  if (!record) return;
  records = records.filter((item) => item.id !== record.id);
  persist();
  invalidateModel();
  showToast("Record removed. Train again to refresh evaluation.");
});

$("#clear-recents").addEventListener("click", () => {
  recents = [];
  try { localStorage.setItem(STORAGE.recent, "[]"); } catch { /* optional */ }
  renderRecent();
  showToast("Prediction history cleared.");
});

$("#k-value").addEventListener("change", () => {
  const input = $("#k-value");
  const max = Math.max(1, model && model.version === datasetVersion ? model.train.length : splitStratified(records).train.length);
  const parsed = Number(input.value);
  if (!Number.isInteger(parsed)) {
    input.value = String(kSetting);
    $("#training-status").className = "training-status warn";
    $("#training-status span").textContent = `Choose a whole-number k between 1 and ${max}.`;
    return;
  }
  kSetting = clamp(parsed, 1, max);
  input.value = String(kSetting);
  try { localStorage.setItem(STORAGE.k, String(kSetting)); } catch { /* optional */ }
  if (model && model.version === datasetVersion) {
    renderEvaluation();
    $("#training-status").className = "training-status good";
    $("#training-status span").textContent = `Evaluation recalculated with k = ${kSetting}. The fitted training split is unchanged.`;
  } else {
    $("#training-status").className = "training-status warn";
    $("#training-status span").textContent = `k set to ${kSetting}. Train the model to calculate evaluation metrics.`;
  }
});

$("#train-button").addEventListener("click", () => {
  if (training) return;
  const split = splitStratified(records);
  if (split.train.length < 1 || split.test.length < 1 || !OUTCOMES.every((outcome) => split.train.some((r) => r.outcome === outcome) && split.test.some((r) => r.outcome === outcome))) {
    $("#training-status").className = "training-status warn";
    $("#training-status span").textContent = "A stratified train/test split needs at least two records in each class. Add examples for any missing class.";
    showToast("Not enough examples in both classes to train.");
    return;
  }
  updateKBounds();
  const trainingVersion = datasetVersion;
  training = true;
  const trainButton = $("#train-button");
  trainButton.disabled = true;
  trainButton.textContent = "Training…";
  $("#training-status").className = "training-status loading";
  $("#training-status span").textContent = "Preparing a stratified split, fitting training-only normalization, and evaluating held-out records…";
  setTimeout(() => {
    training = false;
    trainButton.disabled = false;
    trainButton.innerHTML = `${icon("spark")} Train model`;
    if (trainingVersion !== datasetVersion) {
      model = null;
      $("#training-status").className = "training-status warn";
      $("#training-status span").textContent = "Dataset changed while training. The result was discarded; train again to evaluate the current records.";
      renderAll();
      return;
    }
    const scaler = fittedScaler(split.train);
    model = { train: split.train, test: split.test, scaler, version: trainingVersion };
    const metrics = calculateMetrics(model, kSetting);
    $("#training-status").className = "training-status good";
    $("#training-status span").textContent = `Model trained on ${split.train.length} records and evaluated on ${split.test.length} separate test records. Positive class: Needs Support.`;
    renderAll();
    showToast(`Training complete · held-out accuracy ${(metrics.accuracy * 100).toFixed(1)}%.`);
  }, 380);
});

$("#prediction-form").addEventListener("submit", (event) => {
  event.preventDefault();
  hidePredictionError();
  $("#prediction-message").textContent = "";
  if (!model || model.version !== datasetVersion) {
    showPredictionError("No current trained model is available. Train or retrain the model after any dataset edits, then try again.");
    return;
  }
  const inputs = [
    [$("#predict-hours"), 0, 24, 0.1, "Study hours"],
    [$("#predict-attendance"), 0, 100, 1, "Attendance"],
    [$("#predict-score"), 0, 100, 1, "Previous test score"],
  ];
  const values = inputs.map(([input, min, max, step, label]) => validateNumber(input, min, max, step, label));
  const invalid = values.find((value) => value.error);
  if (invalid) { showPredictionError(invalid.error); return; }
  const query = { hours: values[0].value, attendance: values[1].value, score: values[2].value };
  const result = classify(query, model, kSetting);
  const n = result.nearest.length;
  const supportShare = n ? result.votes[POSITIVE] / n : 0;
  const onShare = n ? result.votes.Pass / n : 0;
  const winner = result.predicted;
  $("#result-placeholder").style.display = "none";
  $("#result-content").classList.add("visible");
  $("#result-outcome").innerHTML = `<span class="class-badge ${winner === POSITIVE ? "support" : "track"}">${escapeHtml(winner)}</span>`;
  $("#result-share").textContent = `${result.votes[winner]} of ${n} nearest neighbors voted this way (${(result.votes[winner] / n * 100).toFixed(1)}%).`;
  $("#vote-support").style.width = `${supportShare * 100}%`;
  $("#vote-on").style.width = `${onShare * 100}%`;
  $("#support-label").textContent = `Needs Support · ${result.votes[POSITIVE]} (${(supportShare * 100).toFixed(1)}%)`;
  $("#on-label").textContent = `Pass · ${result.votes.Pass} (${(onShare * 100).toFixed(1)}%)`;
  const closest = result.nearest[0];
  const similarOutcomes = result.nearest.filter((item) => item.record.outcome === winner);
  $("#reason-box").innerHTML = `<strong>Why this class?</strong> Of the ${n} closest training examples, ${result.votes[winner]} are labeled “${escapeHtml(winner)}.” The closest example is ${closest.distance.toFixed(2)} normalized distance units away and is labeled “${escapeHtml(closest.record.outcome)}.” Similar neighbors are evidence for this model’s vote, not a guarantee about an individual.`;
  recents.unshift({ ...query, outcome: winner, share: `${result.votes[winner]}/${n} (${(result.votes[winner] / n * 100).toFixed(0)}%)`, at: new Date().toISOString() });
  recents = recents.slice(0, 10);
  try { localStorage.setItem(STORAGE.recent, JSON.stringify(recents)); } catch { /* optional */ }
  renderRecent();
  $("#prediction-message").textContent = `Prediction made with ${kSetting} neighbors.`;
  $("#prediction-message").className = "form-message success";
});

// Set the initial split and create empty-but-real evaluation chart axes.
updateKBounds();
$("#k-value").value = String(kSetting);
$("#training-status").className = "training-status warn";
$("#training-status span").textContent = "No model trained yet. Start a training run to calculate held-out metrics and enable predictions.";
renderAll();
