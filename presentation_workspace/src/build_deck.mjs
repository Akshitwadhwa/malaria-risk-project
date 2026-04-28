import fs from "node:fs";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const W = 1920;
const H = 1080;

const ROOT = path.resolve("..");
const WORKSPACE = path.resolve(".");
const DATA = JSON.parse(fs.readFileSync(path.join(WORKSPACE, "scratch", "deck_data.json"), "utf8"));

const C = {
  canvas: "#F7FAF8",
  ink: "#10231F",
  muted: "#5E706A",
  teal: "#0F766E",
  tealDark: "#104E46",
  red: "#C2413D",
  orange: "#E88734",
  green: "#4C9A68",
  navy: "#16324F",
  sand: "#EFE5D0",
  pale: "#E5F3EF",
  line: "#CAD8D4",
  white: "#FFFFFF",
};

const regionColors = {
  Odisha: "#0F766E",
  Mizoram: "#C2413D",
  Tripura: "#E88734",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function fmt(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "";
  const v = Number(n);
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `${Math.round(v / 1000)}K`;
  return `${Math.round(v)}`;
}

function pct(n) {
  return `${Math.round(Number(n) * 100)}%`;
}

function addBg(slide, color = C.canvas) {
  slide.background.fill = { type: "solid", color };
}

function addShape(slide, { x, y, w, h, fill = C.white, line = fill, geometry = "rect", radius = false, rotation = 0 }) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : geometry,
    position: { left: x, top: y, width: w, height: h, rotation },
    fill: { type: "solid", color: fill },
    line: { style: "solid", fill: line, width: 0 },
  });
}

function addText(slide, value, { x, y, w, h, size = 30, color = C.ink, bold = false, align = "left", valign = "top", fill = C.canvas, name = "" }) {
  const s = addShape(slide, { x, y, w, h, fill, line: fill });
  if (name) s.name = name;
  s.text = value;
  s.text.fontSize = size;
  s.text.color = color;
  s.text.bold = bold;
  s.text.alignment = align;
  s.text.verticalAlignment = valign;
  return s;
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, title, { x: 110, y: 70, w: 1360, h: 90, size: 54, bold: true, color: C.ink, name: "slide-title" });
  if (subtitle) addText(slide, subtitle, { x: 112, y: 150, w: 1300, h: 54, size: 24, color: C.muted, name: "slide-subtitle" });
  addShape(slide, { x: 110, y: 218, w: 210, h: 6, fill: C.teal, line: C.teal });
}

function addFooter(slide, label = "Malaria risk modelling project") {
  addShape(slide, { x: 110, y: 1018, w: 1700, h: 1.5, fill: C.line, line: C.line });
  addText(slide, label, { x: 110, y: 1025, w: 1000, h: 30, size: 14, color: C.muted });
}

function addPill(slide, text, x, y, w, color) {
  addShape(slide, { x, y, w, h: 42, fill: color, line: color, radius: true });
  addText(slide, text, { x: x + 8, y: y + 5, w: w - 16, h: 32, size: 18, color: C.white, bold: true, align: "center", valign: "middle", fill: color });
}

function addMetric(slide, label, value, x, y, color) {
  addText(slide, value, { x, y, w: 320, h: 70, size: 54, bold: true, color, fill: C.canvas });
  addText(slide, label, { x, y: y + 72, w: 360, h: 60, size: 21, color: C.muted, fill: C.canvas });
}

function addFlowStep(slide, n, title, sub, x, y, color) {
  addShape(slide, { x, y, w: 238, h: 136, fill: "#FFFFFF", line: C.line, radius: true });
  addShape(slide, { x: x + 18, y: y + 20, w: 42, h: 42, fill: color, line: color, geometry: "ellipse" });
  addText(slide, String(n), { x: x + 18, y: y + 25, w: 42, h: 32, size: 20, color: C.white, bold: true, align: "center", valign: "middle", fill: color });
  addText(slide, title, { x: x + 74, y: y + 22, w: 142, h: 32, size: 23, bold: true, color: C.ink, fill: C.white });
  addText(slide, sub, { x: x + 24, y: y + 70, w: 190, h: 54, size: 16, color: C.muted, fill: C.white });
}

function addArrow(slide, x, y, w = 55) {
  addShape(slide, { x, y: y + 60, w, h: 4, fill: C.teal, line: C.teal });
  addShape(slide, { x: x + w - 12, y: y + 50, w: 24, h: 24, fill: C.teal, line: C.teal, geometry: "triangle", rotation: 90 });
}

function addBullets(slide, bullets, x, y, w, size = 28, color = C.ink, fill = C.canvas) {
  bullets.forEach((b, i) => {
    addShape(slide, { x, y: y + i * 64 + 12, w: 13, h: 13, fill: C.teal, line: C.teal, geometry: "ellipse" });
    addText(slide, b, { x: x + 30, y: y + i * 64, w, h: 56, size, color, fill });
  });
}

function addSimpleTable(slide, rows, columns, { x, y, w, h, headerFill = C.tealDark, rowFill = "#FFFFFF", fontSize = 18 }) {
  const table = slide.tables.add({
    rows: rows.length + 1,
    columns: columns.length,
    left: x,
    top: y,
    width: w,
    height: h,
    values: [columns.map(c => c.label), ...rows.map(r => columns.map(c => r[c.key]))],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  try {
    table.cells.block({ startRow: 0, endRow: 0, startColumn: 0, endColumn: columns.length - 1 }).fill = headerFill;
    table.cells.block({ startRow: 0, endRow: 0, startColumn: 0, endColumn: columns.length - 1 }).textStyle.color = C.white;
    table.cells.block({ startRow: 0, endRow: rows.length, startColumn: 0, endColumn: columns.length - 1 }).textStyle.fontSize = fontSize;
  } catch {}
  return table;
}

function addChart(slide, type, config, frame) {
  const ch = slide.charts.add(type, config);
  ch.frame = frame;
  ch.chartFill = { type: "solid", color: C.canvas };
  ch.plotAreaFill = { type: "solid", color: C.canvas };
  ch.hasLegend = config.hasLegend ?? false;
  return ch;
}

// 1. Cover
{
  const slide = presentation.slides.add();
  addBg(slide, C.tealDark);
  addShape(slide, { x: -140, y: 700, w: 2200, h: 460, fill: "#0B3F39", line: "#0B3F39", rotation: -6 });
  addShape(slide, { x: 1220, y: 80, w: 520, h: 520, fill: "#166B61", line: "#166B61", geometry: "ellipse" });
  addShape(slide, { x: 1335, y: 170, w: 300, h: 300, fill: "#0F766E", line: "#0F766E", geometry: "ellipse" });
  addText(slide, "Malaria Risk\nModelling Project", { x: 110, y: 160, w: 1040, h: 230, size: 82, bold: true, color: C.white, fill: C.tealDark });
  addText(slide, "Geospatial analysis + SIR/SEIR-style reasoning + AI/ML prediction", { x: 116, y: 430, w: 1120, h: 46, size: 30, color: "#CBE9E3", fill: C.tealDark });
  addPill(slide, "India • State + district scale", 116, 520, 330, C.red);
  addPill(slide, "2000-2025 malaria data", 470, 520, 330, C.orange);
  addText(slide, "Odisha • Mizoram • Tripura", { x: 118, y: 900, w: 900, h: 48, size: 30, color: "#DDEFEA", bold: true, fill: "#0B3F39" });
}

// 2. Project question
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "What the project answers", "A regional malaria modelling workflow built from PDF-extracted surveillance data.");
  addText(slide, "Can we move from historical case tables to policy-useful risk signals?", { x: 130, y: 310, w: 1040, h: 150, size: 54, bold: true, color: C.ink });
  addBullets(slide, [
    "Map where malaria burden is concentrated",
    "Compare raw counts with population-adjusted risk",
    "Fit mechanistic and statistical models to the same regions",
    "Use ML to classify next-year high-risk states",
  ], 145, 540, 940, 28);
  addMetric(slide, "selected regions", "3", 1280, 300, C.teal);
  addMetric(slide, "district/state data span", "25y", 1280, 470, C.red);
  addMetric(slide, "model families", "3", 1280, 640, C.orange);
  addFooter(slide);
}

// 3. Workflow
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Analysis pipeline", "The project turns raw tables into mapped patterns, model outputs, and policy interpretation.");
  const y = 380;
  const xs = [90, 355, 620, 885, 1150, 1415];
  addFlowStep(slide, 1, "Data", "PDF tables converted into CSV files", xs[0], y, C.teal);
  addArrow(slide, xs[0] + 238, y);
  addFlowStep(slide, 2, "Clean", "State labels, numeric fields, rates", xs[1], y, C.orange);
  addArrow(slide, xs[1] + 238, y);
  addFlowStep(slide, 3, "Map", "Choropleths and spatial heterogeneity", xs[2], y, C.green);
  addArrow(slide, xs[2] + 238, y);
  addFlowStep(slide, 4, "Model", "SIR + ARIMA/SARIMA curves", xs[3], y, C.red);
  addArrow(slide, xs[3] + 238, y);
  addFlowStep(slide, 5, "Predict", "Random Forest high-risk classifier", xs[4], y, C.navy);
  addArrow(slide, xs[4] + 238, y);
  addFlowStep(slide, 6, "Policy", "Peak timing, magnitude, duration", xs[5], y, C.tealDark);
  addText(slide, "Key design rule: geospatial analysis comes before modelling, so model results are interpreted in spatial context.", { x: 210, y: 720, w: 1500, h: 80, size: 31, bold: true, color: C.ink });
  addFooter(slide);
}

// 4. Dataset and preparation
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Dataset made analysis-ready", "The notebooks separate raw PDF extraction from modelling-ready CSVs.");
  addText(slide, "Main data layers", { x: 120, y: 300, w: 520, h: 50, size: 38, bold: true, color: C.tealDark });
  addBullets(slide, [
    "District malaria cases and deaths: 2000-2024",
    "State-year case rates per 100,000 people",
    "Recent epidemiological indicators: 2024-2025",
    "Lagged features for AI/ML risk prediction",
  ], 140, 380, 720, 25);
  const prepRows = [
    { step: "Clean labels", output: "Himachal/Haryana district issue handled" },
    { step: "Aggregate", output: "District rows to state-year totals" },
    { step: "Normalize", output: "Cases and deaths per 100,000" },
    { step: "Engineer", output: "Lag, rolling, and next-year ML features" },
  ];
  addSimpleTable(slide, prepRows, [{ key: "step", label: "Preparation" }, { key: "output", label: "Output used later" }], { x: 980, y: 315, w: 790, h: 370, fontSize: 18 });
  addText(slide, "Why it matters", { x: 980, y: 735, w: 360, h: 42, size: 32, bold: true, color: C.red });
  addText(slide, "Raw cases show operational burden; per-capita rates reveal intensity. Both are needed for fair regional comparison.", { x: 980, y: 785, w: 760, h: 100, size: 28, color: C.ink });
  addFooter(slide);
}

// 5. Region selection
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Three regions, contrasting risk profiles", "Odisha dominates raw burden; Mizoram dominates intensity after population adjustment.");
  const rows = DATA.burden.map(r => ({
    state: r.state,
    cases: fmt(r.cumulative_cases),
    rate: fmt(r.cumulative_rate_per_100k),
    role: r.state === "Odisha" ? "largest raw burden" : r.state === "Mizoram" ? "highest per-capita intensity" : "intermediate comparator",
  }));
  addSimpleTable(slide, rows, [
    { key: "state", label: "Region" },
    { key: "cases", label: "Cumulative cases" },
    { key: "rate", label: "Cases / 100k" },
    { key: "role", label: "Why included" },
  ], { x: 115, y: 310, w: 980, h: 340, fontSize: 18 });
  addChart(slide, "bar", {
    title: "Cumulative burden contrast",
    categories: DATA.burden.map(r => r.state),
    series: [
      { name: "Cases (millions)", values: DATA.burden.map(r => Number((r.cumulative_cases / 1_000_000).toFixed(2))) },
      { name: "Rate per 100k / 10k", values: DATA.burden.map(r => Number((r.cumulative_rate_per_100k / 10_000).toFixed(2))) },
    ],
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 12 } },
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 90 },
  }, { left: 1160, top: 290, width: 620, height: 430 });
  addText(slide, "Interpretation", { x: 130, y: 730, w: 300, h: 40, size: 32, bold: true, color: C.tealDark });
  addText(slide, "The selected states are comparable as malaria-relevant regions, but contrasting in population size, intensity, and resurgence behaviour.", { x: 130, y: 780, w: 1500, h: 78, size: 29, color: C.ink });
  addFooter(slide);
}

// 6. Geospatial finding
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Geospatial analysis changed the story", "Raw counts and per-capita rates point to different kinds of risk.");
  addText(slide, "Raw burden", { x: 150, y: 300, w: 430, h: 50, size: 42, bold: true, color: C.tealDark });
  addText(slide, "Odisha stands out because of very high cumulative cases and a large population base.", { x: 150, y: 370, w: 560, h: 120, size: 30, color: C.ink });
  addText(slide, "Per-capita intensity", { x: 150, y: 590, w: 520, h: 50, size: 42, bold: true, color: C.red });
  addText(slide, "Mizoram becomes most intense when cases are adjusted for population.", { x: 150, y: 660, w: 600, h: 110, size: 30, color: C.ink });
  addShape(slide, { x: 1010, y: 280, w: 520, h: 520, fill: C.pale, line: C.pale, geometry: "ellipse" });
  addShape(slide, { x: 1120, y: 365, w: 150, h: 190, fill: C.teal, line: C.teal, radius: true, rotation: -10 });
  addShape(slide, { x: 1285, y: 430, w: 90, h: 110, fill: C.red, line: C.red, radius: true, rotation: 15 });
  addShape(slide, { x: 1370, y: 530, w: 70, h: 90, fill: C.orange, line: C.orange, radius: true, rotation: 8 });
  addText(slide, "Spatial heterogeneity", { x: 1010, y: 840, w: 620, h: 55, size: 42, bold: true, color: C.ink, align: "center" });
  addText(slide, "Different regions need different intervention thresholds.", { x: 990, y: 900, w: 660, h: 42, size: 24, color: C.muted, align: "center" });
  addFooter(slide);
}

// 7. Recent epidemiological data
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Recent burden fell, but remains concentrated", "The 2024-2025 epidemiological file adds latest context, not long-term training data.");
  addChart(slide, "bar", {
    title: "Positive malaria cases, 2024 vs 2025",
    categories: DATA.top_recent.map(r => r.state),
    series: [
      { name: "2024", values: DATA.top_recent.map(r => r.positive_2024) },
      { name: "2025", values: DATA.top_recent.map(r => r.positive_2025) },
    ],
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 12 } },
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
    yAxis: { title: { text: "Positive cases" } },
  }, { left: 90, top: 300, width: 1050, height: 520 });
  const recentRows = DATA.recent_selected.map(r => ({
    state: r.state,
    y2024: fmt(r.positive_2024),
    y2025: fmt(r.positive_2025),
    change: `${r.change > 0 ? "+" : ""}${fmt(r.change)}`,
  }));
  addSimpleTable(slide, recentRows, [
    { key: "state", label: "Region" },
    { key: "y2024", label: "2024" },
    { key: "y2025", label: "2025" },
    { key: "change", label: "Change" },
  ], { x: 1220, y: 350, w: 560, h: 260, fontSize: 19 });
  addText(slide, "All three selected regions decreased from 2024 to 2025, but Odisha remains the largest selected-region case load.", { x: 1230, y: 660, w: 520, h: 120, size: 28, color: C.ink });
  addFooter(slide);
}

// 8. SIR
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Mechanistic model: transmission persists", "SIR converts case curves into transmission parameters and an R0 proxy.");
  addChart(slide, "bar", {
    title: "SIR R0 proxy by region",
    categories: DATA.sir_summary.map(r => r.region),
    series: [{ name: "R0 proxy", values: DATA.sir_summary.map(r => r.R0) }],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 95 },
    yAxis: { title: { text: "R0 proxy" } },
    dataLabels: { showValue: true },
  }, { left: 125, top: 300, width: 820, height: 500 });
  addShape(slide, { x: 125, y: 730, w: 820, h: 5, fill: C.red, line: C.red });
  addText(slide, "R0 > 1 for all selected regions", { x: 1040, y: 320, w: 700, h: 70, size: 48, bold: true, color: C.red });
  addBullets(slide, [
    "Mizoram has the highest transmission potential signal",
    "Odisha's decline is strong, but not automatically biological elimination",
    "Tripura shows lower R0, yet still above the endemic threshold",
  ], 1060, 430, 690, 27);
  addFooter(slide);
}

// 9. Forecast disagreement
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Statistical forecasts can be optimistic", "The northeast resurgence is where trend-based models need mechanistic caution.");
  const miz = DATA.ts_series.Mizoram;
  addChart(slide, "line", {
    title: "Mizoram holdout: actual vs ARIMA/SARIMA forecast",
    categories: miz.years.map(String),
    series: [
      { name: "Actual", values: miz.actual },
      { name: "ARIMA", values: miz.arima },
      { name: "SARIMA", values: miz.sarima },
    ],
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 12 } },
    yAxis: { title: { text: "Annual cases" } },
  }, { left: 100, top: 300, width: 980, height: 520 });
  const rows = DATA.ts_summary.map(r => ({ region: r.region, arima: fmt(r.arima_mae), sarima: fmt(r.sarima_mae), winner: r.winner }));
  addSimpleTable(slide, rows, [
    { key: "region", label: "Region" },
    { key: "arima", label: "ARIMA MAE" },
    { key: "sarima", label: "SARIMA MAE" },
    { key: "winner", label: "Best" },
  ], { x: 1170, y: 330, w: 600, h: 280, fontSize: 18 });
  addText(slide, "Policy read", { x: 1180, y: 660, w: 350, h: 48, size: 36, bold: true, color: C.tealDark });
  addText(slide, "Use ARIMA/SARIMA for short-term operational planning, but use SIR R0 when deciding whether it is safe to reduce interventions.", { x: 1180, y: 715, w: 570, h: 130, size: 27, color: C.ink });
  addFooter(slide);
}

// 10. ML
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "AI/ML model: next-year risk screening", "Random Forest predicted whether a state becomes high malaria risk next year.");
  addMetric(slide, "selected model", "RF", 135, 315, C.teal);
  addMetric(slide, "test F1", DATA.ml_metrics.test_f1.toFixed(2), 135, 500, C.red);
  addMetric(slide, "ROC-AUC", DATA.ml_metrics.test_roc_auc.toFixed(2), 135, 685, C.orange);
  addChart(slide, "bar", {
    title: "Top model features",
    categories: DATA.ml_importance.map(r => r.feature.replaceAll("_", " ")),
    series: [{ name: "Importance", values: DATA.ml_importance.map(r => r.importance) }],
    hasLegend: false,
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 80 },
    xAxis: { title: { text: "Importance" } },
  }, { left: 560, top: 300, width: 1180, height: 560 });
  addText(slide, `High risk threshold: ${DATA.ml_metrics.risk_threshold} cases per 100,000`, { x: 560, y: 880, w: 820, h: 40, size: 24, color: C.muted });
  addFooter(slide);
}

// 11. Mandatory policy outputs
{
  const slide = presentation.slides.add();
  addBg(slide);
  addTitle(slide, "Policy outputs extracted from each model", "Peak timing, peak magnitude, and duration translate model curves into planning signals.");
  const compact = DATA.policy_outputs
    .filter(r => (r.model === "SIR" || r.model === "ARIMA" || r.model === "ML classifier"))
    .map(r => ({
      region: r.region,
      model: r.model,
      peak: String(r.peak_timing),
      magnitude: r.unit === "risk probability" ? pct(r.peak_magnitude) : fmt(r.peak_magnitude),
      duration: `${r.duration}y`,
    }));
  addSimpleTable(slide, compact, [
    { key: "region", label: "Region" },
    { key: "model", label: "Model" },
    { key: "peak", label: "Peak timing" },
    { key: "magnitude", label: "Peak magnitude" },
    { key: "duration", label: "Duration" },
  ], { x: 95, y: 290, w: 1240, h: 590, fontSize: 15 });
  addText(slide, "Why peak timing?", { x: 1410, y: 330, w: 380, h: 48, size: 36, bold: true, color: C.red });
  addText(slide, "It is the year where that model reaches its maximum burden or highest high-risk probability.", { x: 1410, y: 390, w: 390, h: 120, size: 27, color: C.ink });
  addText(slide, "That tells health teams when surveillance, testing, treatment, and vector-control pressure is expected to be highest.", { x: 1410, y: 560, w: 390, h: 160, size: 27, color: C.ink });
  addFooter(slide);
}

// 12. Final
{
  const slide = presentation.slides.add();
  addBg(slide, C.tealDark);
  addText(slide, "Main takeaway", { x: 115, y: 120, w: 620, h: 70, size: 58, bold: true, color: C.white, fill: C.tealDark });
  addText(slide, "Malaria risk is spatially uneven, trend-dependent, and still intervention-sensitive.", { x: 115, y: 250, w: 1300, h: 150, size: 58, bold: true, color: "#DDF4ED", fill: C.tealDark });
  addBullets(slide, [
    "Maps show raw burden and per-capita intensity tell different stories",
    "SIR R0 warns that transmission potential persists",
    "Time-series models help forecast case load but can miss resurgence",
    "ML adds scalable high-risk screening across states",
  ], 145, 505, 1120, 30, C.white, C.tealDark);
  addShape(slide, { x: 1360, y: 455, w: 330, h: 330, fill: C.red, line: C.red, geometry: "ellipse" });
  addText(slide, "Do not use one model alone.", { x: 1330, y: 560, w: 390, h: 95, size: 38, bold: true, color: C.white, align: "center", valign: "middle", fill: C.red });
  addText(slide, "Use mapping + SIR + forecasting + ML together.", { x: 118, y: 930, w: 960, h: 46, size: 30, color: "#BFE7DE", bold: true, fill: C.tealDark });
}

// Speaker notes: short presenter guide on each slide.
presentation.slides.items.forEach((slide, idx) => {
  const notes = [
    "Introduce the project as a modelling workflow, not just a data-cleaning task.",
    "Emphasize the central question: converting historical malaria data into risk signals.",
    "Walk left to right: data, cleaning, geospatial analysis, modelling, prediction, policy.",
    "Explain that analysis-ready data is what makes modelling possible.",
    "Use this slide to justify region selection and why raw counts are not enough.",
    "Say that spatial context comes before modelling because model behavior differs by region.",
    "Point out that recent 2025 cases declined, but burden is still concentrated.",
    "Explain R0 simply: above 1 means transmission can persist without control.",
    "Use Mizoram as the example where trend forecasting can miss resurgence.",
    "Explain that ML predicts high-risk status, not biological causes.",
    "Define peak timing as the model maximum and duration as sustained burden/risk.",
    "Close by saying the strongest conclusion comes from combining all model types.",
  ][idx] || "";
  slide.speakerNotes.clear();
  slide.speakerNotes.append(notes);
});

fs.mkdirSync("output", { recursive: true });
for (const file of fs.readdirSync("output")) {
  if (file !== "output.pptx") fs.rmSync(path.join("output", file), { recursive: true, force: true });
}
fs.mkdirSync("scratch/previews", { recursive: true });

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save("output/output.pptx");

for (let i = 0; i < presentation.slides.items.length; i += 1) {
  const blob = await presentation.slides.items[i].export({ format: "png" });
  const buffer = Buffer.from(await blob.arrayBuffer());
  fs.writeFileSync(`scratch/previews/slide-${String(i + 1).padStart(2, "0")}.png`, buffer);
}

console.log(JSON.stringify({
  slides: presentation.slides.items.length,
  pptx: path.join(WORKSPACE, "output", "output.pptx"),
  previews: path.join(WORKSPACE, "scratch", "previews"),
}, null, 2));
