import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire("/tmp/prune-benchmark-workbook/entry.cjs");
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const artifacts = path.join(root, "artifacts");
const outputDir = path.resolve(process.argv[2] || path.join(root, "outputs", "benchmark-quality-20261007"));

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const item = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(item);
    return entry.name === "result.json" ? [item] : [];
  }));
  return nested.flat();
}

async function readJson(file) {
  const stat = await fs.stat(file);
  // Two legacy results contain complete benchmark headers followed by a very
  // large pruning plan. The prefix has every field used in this workbook.
  const text = stat.size > 25_000_000
    ? await fs.open(file, "r").then(async handle => {
        const buffer = Buffer.alloc(1_000_000);
        await handle.read(buffer, 0, buffer.length, 0);
        await handle.close();
        return buffer.toString("utf8");
      })
    : await fs.readFile(file, "utf8");
  if (stat.size <= 25_000_000) return JSON.parse(text);
  const object = (key) => {
    const match = text.match(new RegExp(`"${key}"\\s*:\\s*(\\{[\\s\\S]*?\\n  \\})`));
    return match ? JSON.parse(match[1]) : {};
  };
  const benchmark = object("benchmark");
  const criterion = text.match(/"criterion_name"\s*:\s*"([^"]+)"/);
  return {
    baseline_metrics: object("baseline_metrics"),
    complexity_after: object("complexity_after"),
    complexity_before: object("complexity_before"),
    final_metrics: object("final_metrics"),
    pruning: { benchmark, criterion_name: criterion?.[1] },
    status: "partial-result-read",
  };
}

async function configFor(resultFile) {
  for (const name of ["config.resolved.json", "config.json"]) {
    const file = path.join(path.dirname(resultFile), name);
    try { return await readJson(file); } catch { /* next */ }
  }
  return {};
}

const value = (source, key) => source?.[key] ?? null;
// Historic results store classification accuracy in both conventions: 0.84 and
// 84. Keep a typed fraction so Excel's percent format always displays 84.0%.
const percent = (value) => value === null ? "N/A" : (value > 1 ? value / 100 : value);
const measuredPercent = (value) => value === null ? "NR" : percent(value);
const number = (value) => value === null ? "N/A" : value;
const reduce = (before, after) => before === null || after === null || before === 0
  ? "N/A" : 1 - after / before;
const fmtSource = (file) => path.relative(root, file).replaceAll("\\", "/");

function methodOf(config, result) {
  const pruning = config.pruning || {};
  const resultPruning = result.pruning || {};
  const pruner = pruning.pruner || pruning.method || resultPruning.pruner_name;
  const criterion = pruning.criterion || resultPruning.criterion_name;
  const granularity = pruning.granularity || pruning.structure || resultPruning.granularity_name;
  if (pruner === "block_sparse") return "Block Pruning";
  if (pruner === "depth") return "Layer / Residual-block Pruning";
  if (pruner === "nm") return "N:M Block Pruning";
  if (criterion === "taylor") return "Taylor-based Gradient Pruning";
  if (pruner === "structured") return granularity === "channel" ? "Channel Pruning" : "Filter Pruning (L1)";
  return ({ magnitude: "Magnitude Pruning", snip: "SNIP", grasp: "GraSP", synflow: "SynFlow", lamp: "LAMP" })[criterion] || criterion || "Unknown";
}

function structureOf(config, result) {
  const pruning = config.pruning || {};
  const resultPruning = result.pruning || {};
  const pruner = pruning.pruner || pruning.method || resultPruning.pruner_name;
  if (pruner === "structured" || pruner === "depth") return "Structured";
  if (pruner === "block_sparse" || pruner === "nm") return "Block / patterned";
  return "Unstructured";
}

function normalizeRecord(file, config, result) {
  const model = config.model?.name || result.pruning?.model_name || "Unknown";
  const pruning = config.pruning || {};
  const before = result.complexity_before || {};
  const after = result.complexity_after || {};
  const base = result.baseline_metrics || {};
  const final = result.final_metrics || {};
  const benchmark = result.pruning?.benchmark || {};
  const amount = pruning.amount ?? pruning.target_ratio ?? result.requested_sparsity ?? null;
  const actualSparsity = result.actual_sparsity ?? (after.sparsity_pct == null ? null : after.sparsity_pct / 100);
  const recovery = config.recovery?.enabled;
  return {
    source: fmtSource(file), run: path.basename(path.dirname(file)), model,
    method: methodOf(config, result), structure: structureOf(config, result),
    requestedRatio: amount, actualSparsity, fineTuning: recovery === true ? "Configured" : recovery === false ? "No" : "N/A",
    baselineAccuracy: value(base, "accuracy"), postAccuracy: value(final, "accuracy"),
    precision: value(final, "precision"), recall: value(final, "recall"), f1: value(final, "f1"),
    map50: value(final, "map50"), map5095: final.map50_95 ?? final.map5095 ?? final.map ?? null,
    paramsBefore: value(before, "params"), paramsAfter: value(after, "params"),
    flopsBefore: value(before, "flops"), flopsAfter: value(after, "flops"),
    inferenceMs: value(benchmark, "inference_latency_ms"), nmsMs: model.startsWith("yolo") ? value(benchmark, "nms_latency_ms") : null,
    totalMs: value(benchmark, "total_latency_ms"), fps: value(benchmark, "fps"),
    status: result.status || "complete",
    qualityReevaluated: Boolean(result.quality_reevaluation),
  };
}

function fingerprint(record) {
  return [record.model, record.method, record.structure, record.requestedRatio, record.actualSparsity,
    record.baselineAccuracy, record.postAccuracy, record.precision, record.recall, record.f1, record.map50, record.map5095,
    record.paramsAfter, record.flopsAfter,
    record.inferenceMs, record.totalMs, record.fps].join("|");
}

const files = await walk(artifacts);
const raw = [];
for (const file of files) {
  try {
    const result = await readJson(file);
    const config = await configFor(file);
    const model = config.model?.name || result.pruning?.model_name;
    if (["lenet5", "lenet5_emnist_onnx", "resnet18", "torchvision_resnet18", "yolov5", "yolov5s", "rtdetr", "rt-detr"].includes(model)) {
      raw.push(normalizeRecord(file, config, result));
    }
  } catch (error) {
    console.error(`Skipped invalid result ${file}: ${error.message}`);
  }
}
raw.sort((a, b) => a.model.localeCompare(b.model) || a.method.localeCompare(b.method) || (a.requestedRatio ?? 9) - (b.requestedRatio ?? 9) || a.run.localeCompare(b.run));
const unique = [];
const duplicates = new Map();
for (const record of raw) {
  const id = fingerprint(record);
  if (!duplicates.has(id)) { duplicates.set(id, [record]); unique.push(record); }
  else duplicates.get(id).push(record);
}
for (const record of unique) record.repeatCount = duplicates.get(fingerprint(record)).length;

const wb = Workbook.create();
const overview = wb.worksheets.add("Overview");
const quality = wb.worksheets.add("Quality Benchmark");
const runtime = wb.worksheets.add("Complexity & Runtime");
const rawSheet = wb.worksheets.add("Raw runs");
const notes = wb.worksheets.add("Notes");

const navy = "#163A4D", teal = "#0F766E", lightBlue = "#E8F1F5", gray = "#F5F7FA", amber = "#FFF2CC";
function title(sheet, text, end) {
  sheet.getRange(`A2:${end}2`).merge();
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format = { font: { name: "Arial", size: 15, bold: true, color: navy } };
  sheet.getRange(`A3:${end}3`).format.borders = { bottom: { style: "medium", color: teal } };
}
function header(sheet, range) {
  sheet.getRange(range).format = { fill: navy, font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
}
function dataStyle(sheet, range) {
  sheet.getRange(range).format = { font: { name: "Arial", size: 10, color: "#172033" }, verticalAlignment: "center" };
}
for (const sheet of [overview, quality, runtime, rawSheet, notes]) sheet.showGridLines = false;
overview.tabColor = navy; quality.tabColor = teal; runtime.tabColor = teal;

title(overview, "Pruning benchmark: classification and detection", "H");
overview.getRange("A5:H5").values = [["Scope", "Saved runs", "Benchmark rows", "LeNet5 rows", "ResNet18 rows", "Fine-tuning", "Quality metrics", "Runtime metrics"]];
header(overview, "A5:H5");
const lenetCount = unique.filter(row => row.model.startsWith("lenet")).length;
const resnetCount = unique.filter(row => row.model.includes("resnet")).length;
overview.getRange("A6:H6").values = [["Saved result.json files", raw.length, unique.length, lenetCount, resnetCount, "Configuration only", "P / R / F1 / mAP / accuracy", "Per source run"]];
dataStyle(overview, "A6:H6");
overview.getRange("A8:H8").values = [["Reading guide", "", "", "", "", "", "", ""]];
overview.getRange("A8:H8").merge();
overview.getRange("A8").format = { fill: lightBlue, font: { name: "Arial", size: 10, bold: true, color: navy } };
overview.getRange("A9:A12").values = [["Quality"], ["Runtime"], ["Fine-tuning"], ["Comparability"]];
for (const [row, text] of [
  [9, "Classification: accuracy and macro P/R/F1 over observed classes. Detection: macro P/R/F1 and mAP. IoU mAP is N/A for classification. NR means the saved run has no measurement."],
  [10, "Parameters and GFLOPs are before → after. Unstructured masks can leave dense parameter count unchanged and do not prove a speedup."],
  [11, "All rows state whether recovery was configured. Saved runs do not provide a verified after-fine-tuning metric."],
  [12, "Only compare rows that share model, dataset, hardware and protocol. This workbook does not rank across LeNet5 and ResNet18."],
]) { overview.getRange(`B${row}:H${row}`).merge(); overview.getRange(`B${row}`).values = [[text]]; }
overview.getRange("A9:H12").format.wrapText = true;
overview.getRange("A9:A12").format.font = { name: "Arial", size: 10, bold: true, color: navy };
overview.getRange("A9:H12").format = { font: { name: "Arial", size: 10 }, verticalAlignment: "center", wrapText: true };
overview.getRange("A14:H14").merge();
overview.getRange("A14").values = [["N/A = not applicable; NR = quality metric not recorded. Raw runs retains every source result. Reevaluated quality metrics come from validation inference on restored models. Remaining NR values need the original data/model."]];
overview.getRange("A14").format = { fill: amber, font: { name: "Arial", size: 10, italic: true }, wrapText: true };

const qualityHeaders = ["Model Name", "Pruning Method", "Structure", "Requested ratio", "Actual sparsity", "Precision (P)", "Recall (R)", "F1", "mAP@0.5", "mAP@0.5:0.95", "Accuracy baseline", "Accuracy post-prune", "Fine-tuning", "Repeated runs", "Source run"];
const classificationModel = row => row.model.startsWith("lenet") || row.model.includes("resnet");
const qualityMetrics = row => [measuredPercent(row.precision), measuredPercent(row.recall), measuredPercent(row.f1),
  classificationModel(row) ? "N/A" : measuredPercent(row.map50),
  classificationModel(row) ? "N/A" : measuredPercent(row.map5095)];
const qualityRows = unique.map(row => [row.model, row.method, row.structure, number(row.requestedRatio), number(row.actualSparsity), ...qualityMetrics(row), percent(row.baselineAccuracy), percent(row.postAccuracy), row.fineTuning, row.repeatCount, row.run]);
title(quality, "1. Quality Benchmark", "O");
quality.getRangeByIndexes(4, 0, 1, qualityHeaders.length).values = [qualityHeaders];
quality.getRangeByIndexes(5, 0, qualityRows.length, qualityHeaders.length).values = qualityRows;
header(quality, "A5:O5"); dataStyle(quality, `A6:O${5 + qualityRows.length}`);
quality.getRange(`D6:L${5 + qualityRows.length}`).format.numberFormat = "0.0%";
quality.getRange(`D6:L${5 + qualityRows.length}`).format.horizontalAlignment = "center";
quality.getRange(`F6:H${5 + qualityRows.length}`).format.numberFormat = "0.00%";
quality.getRange(`M6:N${5 + qualityRows.length}`).format.horizontalAlignment = "center";
quality.freezePanes.freezeRows(5);

const runtimeHeaders = ["Model Name", "Pruning Method", "Structure", "Requested ratio", "Actual sparsity", "Parameters before", "Parameters after", "Parameter reduction", "GFLOPs before", "GFLOPs after", "GFLOPs reduction", "Inference Speed (ms)", "NMS Speed (ms)", "Total Latency (ms)", "FPS", "Fine-tuning", "Repeated runs", "Source run"];
const runtimeRows = unique.map(row => [row.model, row.method, row.structure, number(row.requestedRatio), number(row.actualSparsity), number(row.paramsBefore), number(row.paramsAfter), reduce(row.paramsBefore, row.paramsAfter), row.flopsBefore === null ? "N/A" : row.flopsBefore / 1e9, row.flopsAfter === null ? "N/A" : row.flopsAfter / 1e9, reduce(row.flopsBefore, row.flopsAfter), number(row.inferenceMs), number(row.nmsMs), number(row.totalMs), number(row.fps), row.fineTuning, row.repeatCount, row.run]);
title(runtime, "2. Complexity & Runtime Benchmark", "R");
runtime.getRangeByIndexes(4, 0, 1, runtimeHeaders.length).values = [runtimeHeaders];
runtime.getRangeByIndexes(5, 0, runtimeRows.length, runtimeHeaders.length).values = runtimeRows;
header(runtime, "A5:R5"); dataStyle(runtime, `A6:R${5 + runtimeRows.length}`);
runtime.getRange(`D6:E${5 + runtimeRows.length}`).format.numberFormat = "0.0%";
runtime.getRange(`H6:H${5 + runtimeRows.length}`).format.numberFormat = "0.0%";
runtime.getRange(`I6:J${5 + runtimeRows.length}`).format.numberFormat = "0.000000";
runtime.getRange(`K6:K${5 + runtimeRows.length}`).format.numberFormat = "0.0%";
runtime.getRange(`L6:O${5 + runtimeRows.length}`).format.numberFormat = "0.000";
runtime.freezePanes.freezeRows(5);

const rawHeaders = ["Model", "Method", "Structure", "Requested ratio", "Actual sparsity", "Baseline accuracy", "Post-prune accuracy", "Parameters before", "Parameters after", "GFLOPs before", "GFLOPs after", "Inference ms", "Total ms", "FPS", "Fine-tuning", "Status", "Source result.json", "Precision (P)", "Recall (R)", "F1", "mAP@0.5", "mAP@0.5:0.95"];
const rawRows = raw.map(row => [row.model, row.method, row.structure, number(row.requestedRatio), number(row.actualSparsity), percent(row.baselineAccuracy), percent(row.postAccuracy), number(row.paramsBefore), number(row.paramsAfter), row.flopsBefore === null ? "N/A" : row.flopsBefore / 1e9, row.flopsAfter === null ? "N/A" : row.flopsAfter / 1e9, number(row.inferenceMs), number(row.totalMs), number(row.fps), row.fineTuning, row.status, row.source, ...qualityMetrics(row)]);
title(rawSheet, "Raw saved runs (all source runs retained)", "V");
rawSheet.getRangeByIndexes(4, 0, 1, rawHeaders.length).values = [rawHeaders];
rawSheet.getRangeByIndexes(5, 0, rawRows.length, rawHeaders.length).values = rawRows;
header(rawSheet, "A5:V5"); dataStyle(rawSheet, `A6:V${5 + rawRows.length}`);
rawSheet.getRange(`R6:V${5 + rawRows.length}`).format.numberFormat = "0.0%";
rawSheet.getRange(`R6:V${5 + rawRows.length}`).format.horizontalAlignment = "center";
rawSheet.getRange(`R6:T${5 + rawRows.length}`).format.numberFormat = "0.00%";
rawSheet.getRange(`D6:G${5 + rawRows.length}`).format.numberFormat = "0.0%";
rawSheet.getRange(`J6:K${5 + rawRows.length}`).format.numberFormat = "0.000000";
rawSheet.getRange(`L6:N${5 + rawRows.length}`).format.numberFormat = "0.000";
rawSheet.freezePanes.freezeRows(5);

title(notes, "Notes and metric definitions", "D");
notes.getRange("A5:D5").values = [["Item", "Definition", "Treatment in this workbook", "Source"]];
header(notes, "A5:D5");
notes.getRange("A6:D14").values = [
  ["Population", "LeNet5, ResNet18, YOLO and RT-DETR", "Completed saved results, including historic and smoke runs; inspect status/source before comparing.", "Saved artifacts"],
  ["Quality metrics", "Precision / Recall / F1 / mAP", "Macro P/R/F1 for classification; IoU mAP is N/A. Missing historical quality measurements are NR. Macro F1 averages per-class F1. Detection P/R protocol follows its evaluator; unmeasured F1 remains NR.", "final_metrics"],
  ["Fine-tuning", "Recovery configuration", "Configured / No is not evidence of an observed after-FT score.", "config.resolved.json"],
  ["Parameters", "Dense parameter count", "Unstructured masks may keep this unchanged.", "complexity_before / complexity_after"],
  ["GFLOPs", "FLOPs divided by 1e9", "Only shown when both saved values exist. Compare within a matched protocol.", "complexity_before / complexity_after"],
  ["Latency", "Saved benchmark timing", "NMS is N/A for classification. Hardware may differ across historic runs.", "pruning.benchmark"],
  ["Deduplication", "Identical headline metrics", "The two benchmark sheets consolidate identical rows. Raw runs keeps every result.json.", "Workbook processing"],
  ["Quality reevaluation", "Validation inference on restored models", `${raw.filter(row => row.qualityReevaluated).length} runs have measured P/R/F1. Classification uses verified saved-plan replay; YOLO uses retained checkpoints. Missing datasets remain NR. Old runtime measurements are preserved.`, "quality_reevaluation.json"],
  ["Large legacy result", "N:M ResNet18 result", "Read only the complete header portion; giant pruning-plan payload is intentionally omitted.", "result.json prefix"],
];
dataStyle(notes, "A6:D14"); notes.getRange("A6:D14").format.wrapText = true;

for (const sheet of [overview, quality, runtime, rawSheet, notes]) {
  const used = sheet.getUsedRange();
  used.format.autofitColumns();
  used.format.autofitRows();
}
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("B:B").format.columnWidth = 36;
overview.getRange("C:H").format.columnWidth = 18;
overview.getRange("A9:H12").format.rowHeight = 32;
overview.getRange("A6:H6").format.wrapText = true;
overview.getRange("B6:E6").format.horizontalAlignment = "center";
quality.getRange("A:A").format.columnWidth = 22; quality.getRange("B:B").format.columnWidth = 26; quality.getRange("O:O").format.columnWidth = 42;
quality.getRange("C:N").format.columnWidth = 15; quality.getRange("A5:O5").format.rowHeight = 34;
quality.getRange(`O6:O${5 + qualityRows.length}`).format.wrapText = true;
runtime.getRange("A:A").format.columnWidth = 22; runtime.getRange("B:B").format.columnWidth = 26; runtime.getRange("R:R").format.columnWidth = 42;
runtime.getRange("C:Q").format.columnWidth = 15; runtime.getRange("A5:R5").format.rowHeight = 34;
runtime.getRange(`R6:R${5 + runtimeRows.length}`).format.wrapText = true;
rawSheet.getRange("Q:Q").format.columnWidth = 70;
rawSheet.getRange("A:P").format.columnWidth = 16; rawSheet.getRange("A5:Q5").format.rowHeight = 34;
rawSheet.getRange("R:V").format.columnWidth = 16;
rawSheet.getRange(`Q6:Q${5 + rawRows.length}`).format.wrapText = true;
notes.getRange("A:A").format.columnWidth = 22; notes.getRange("B:B").format.columnWidth = 38; notes.getRange("C:C").format.columnWidth = 52; notes.getRange("D:D").format.columnWidth = 32;
for (const sheet of [overview, quality, runtime, rawSheet, notes]) sheet.getUsedRange().format.autofitRows();

wb.recalculate();
for (const check of [
  await wb.inspect({ kind: "table", range: "Quality Benchmark!A2:O20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 15 }),
  await wb.inspect({ kind: "table", range: "Complexity & Runtime!A2:R20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 18 }),
  await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 30 }, summary: "formula error scan" }),
]) console.log(check.ndjson);

await fs.mkdir(outputDir, { recursive: true });
for (const [sheetName, range, file] of [
  ["Overview", "A1:H14", "overview_preview.png"],
  ["Quality Benchmark", "A1:O30", "quality_preview.png"],
  ["Complexity & Runtime", "A1:R30", "runtime_preview.png"],
  ["Raw runs", "Q1:V15", "raw_runs_preview.png"],
  ["Notes", "A1:D15", "notes_preview.png"],
]) {
  const preview = await wb.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(outputDir, file), new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(outputDir, "pruning_benchmark.xlsx"));
const csvCell = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
const csvRows = [[...qualityHeaders.slice(0, -1), "Source result.json"], ...qualityRows.map((row, index) => [...row.slice(0, -1), unique[index].source])];
await fs.writeFile(path.join(outputDir, "quality_benchmark.csv"), csvRows.map(row => row.map(csvCell).join(",")).join("\n") + "\n");
await fs.writeFile(path.join(outputDir, "benchmark_rows.json"), JSON.stringify({ rawRuns: raw, consolidatedRows: unique }, null, 2));
console.log(JSON.stringify({ rawRuns: raw.length, consolidatedRows: unique.length, outputDir }));
