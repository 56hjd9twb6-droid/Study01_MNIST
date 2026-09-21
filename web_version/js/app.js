/**
 * 화면 연결부: 캔버스에 그리기 -> 전처리 -> 추론 -> 결과 표시.
 * 외부 라이브러리 없이 동작한다.
 */
import { MLP } from "./model.js";
import { imageDataToInk, normalize, CANVAS as GRID } from "./preprocess.js";

const PAD_SIZE = 280;       // 그리기 캔버스 한 변(픽셀)
const STROKE_WIDTH = 20;    // MNIST의 획 두께(28px에서 약 2px)에 맞춘 값
const PREDICT_DELAY = 120;  // 그리기를 멈춘 뒤 추론까지 기다리는 시간(ms)

const pad = document.getElementById("pad");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const digitEl = document.getElementById("digit");
const confidenceEl = document.getElementById("confidence");
const confidenceLabel = document.getElementById("confidence-label");
const barsEl = document.getElementById("bars");

const ctx = pad.getContext("2d", { willReadFrequently: true });
const previewCtx = preview.getContext("2d");

let model = null;
let drawing = false;
let hasInk = false;
let predictTimer = null;

/* ---------- 확률 막대 ---------- */

const bars = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML =
    `<div class="label">${d}</div>` +
    `<div class="bar-track"><div class="bar-fill"></div></div>` +
    `<div class="pct">—</div>`;
  barsEl.appendChild(row);
  bars.push({ row, fill: row.querySelector(".bar-fill"), pct: row.querySelector(".pct") });
}

function renderProbs(probs, best) {
  bars.forEach((bar, d) => {
    const p = probs ? probs[d] : 0;
    bar.fill.style.width = `${(p * 100).toFixed(1)}%`;
    bar.pct.textContent = probs ? `${(p * 100).toFixed(1)}%` : "—";
    bar.row.classList.toggle("top", probs != null && d === best);
  });
}

/* ---------- 그리기 캔버스 ---------- */

function resetPad() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, PAD_SIZE, PAD_SIZE);
  ctx.lineWidth = STROKE_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#101014";
}

function clearAll() {
  resetPad();
  hasInk = false;
  digitEl.textContent = "–";
  digitEl.classList.add("empty");
  confidenceEl.textContent = "—";
  confidenceLabel.textContent = "아직 그린 내용이 없습니다";
  renderProbs(null, -1);
  previewCtx.clearRect(0, 0, GRID, GRID);
  previewCtx.fillStyle = "#000";
  previewCtx.fillRect(0, 0, GRID, GRID);
}

/** 포인터 위치를 캔버스 내부 좌표로 바꾼다(CSS 크기와 실제 해상도가 다를 수 있다). */
function positionOf(event) {
  const rect = pad.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * PAD_SIZE,
    y: ((event.clientY - rect.top) / rect.height) * PAD_SIZE,
  };
}

function startStroke(event) {
  if (!model) return;
  drawing = true;
  hasInk = true;
  pad.setPointerCapture(event.pointerId);
  const { x, y } = positionOf(event);
  ctx.beginPath();
  ctx.moveTo(x, y);
  // 점 하나만 찍어도 자국이 남도록 한다
  ctx.lineTo(x + 0.01, y + 0.01);
  ctx.stroke();
  schedulePredict();
}

function continueStroke(event) {
  if (!drawing) return;
  const { x, y } = positionOf(event);
  ctx.lineTo(x, y);
  ctx.stroke();
  schedulePredict();
}

function endStroke(event) {
  if (!drawing) return;
  drawing = false;
  if (pad.hasPointerCapture?.(event.pointerId)) pad.releasePointerCapture(event.pointerId);
  schedulePredict();
}

pad.addEventListener("pointerdown", startStroke);
pad.addEventListener("pointermove", continueStroke);
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);
pad.addEventListener("pointerleave", endStroke);
document.getElementById("clear").addEventListener("click", clearAll);

/* ---------- 추론 ---------- */

function schedulePredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(predict, PREDICT_DELAY);
}

function drawPreview(vec) {
  const img = previewCtx.createImageData(GRID, GRID);
  for (let i = 0; i < GRID * GRID; i++) {
    const v = Math.round(Math.max(0, Math.min(1, vec[i])) * 255);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  previewCtx.putImageData(img, 0, 0);
}

function predict() {
  if (!model || !hasInk) return;

  const ink = imageDataToInk(ctx.getImageData(0, 0, PAD_SIZE, PAD_SIZE));
  const vec = normalize(ink, PAD_SIZE, PAD_SIZE);
  if (!vec) {
    clearAll();
    return;
  }

  drawPreview(vec);
  const { digit, confidence, probs } = model.predict(vec);

  digitEl.textContent = String(digit);
  digitEl.classList.remove("empty");
  confidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;
  confidenceLabel.textContent =
    confidence > 0.9 ? "확실합니다"
    : confidence > 0.6 ? "아마도 이 숫자입니다"
    : "헷갈립니다 — 더 크게 그려 보세요";
  renderProbs(probs, digit);
}

/* ---------- 시작 ---------- */

function setStatus(text, state) {
  statusText.textContent = text;
  statusEl.className = `status${state ? ` ${state}` : ""}`;
}

clearAll();

try {
  model = await MLP.load("./model/weights.json");
  setStatus(
    `준비 완료 · ${model.sizes.join(" → ")} · 파라미터 ${model.paramCount.toLocaleString()}개`,
    "ready",
  );
} catch (err) {
  setStatus(`모델을 불러오지 못했습니다: ${err.message}`, "error");
  console.error(err);
}
