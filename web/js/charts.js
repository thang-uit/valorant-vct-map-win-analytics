import { getLanguage, t } from "./i18n.js";
import { percent } from "./formatters.js";

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

export function drawProbabilityChart({ canvas, frame, tooltip, points, onPoint }) {
  if (!canvas || !frame || !points?.length) return;
  const ratio = window.devicePixelRatio || 1;
  const width = frame.clientWidth;
  const height = frame.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const padding = { left: 10, right: 12, top: 14, bottom: 14 };
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const plotHeight = Math.max(1, height - padding.top - padding.bottom);
  const x = (index) => padding.left + (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth);
  const y = (value) => padding.top + (1 - value) * plotHeight;
  const accent = css("--accent");
  const surface = css("--surface");
  const fill = context.createLinearGradient(0, padding.top, 0, height);
  fill.addColorStop(0, `${accent}28`);
  fill.addColorStop(1, `${accent}00`);
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(x(index), y(point.probability)) : context.moveTo(x(index), y(point.probability)));
  context.lineTo(x(points.length - 1), height - padding.bottom);
  context.lineTo(x(0), height - padding.bottom);
  context.closePath();
  context.fillStyle = fill;
  context.fill();
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(x(index), y(point.probability)) : context.moveTo(x(index), y(point.probability)));
  context.lineWidth = 2.5;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.strokeStyle = accent;
  context.stroke();
  points.forEach((point, index) => {
    context.beginPath();
    context.arc(x(index), y(point.probability), 3.6, 0, Math.PI * 2);
    context.fillStyle = surface;
    context.fill();
    context.lineWidth = 1.8;
    context.strokeStyle = accent;
    context.stroke();
  });
  const pointAt = (event) => {
    const bounds = canvas.getBoundingClientRect();
    const cursor = Math.max(0, Math.min(points.length - 1, Math.round(((event.clientX - bounds.left) / bounds.width) * (points.length - 1))));
    const selected = points[cursor];
    tooltip.style.display = "block";
    tooltip.style.left = `${x(cursor)}px`;
    tooltip.style.top = `${y(selected.probability)}px`;
    tooltip.textContent = `${t("round")} ${selected.round}  ·  ${percent(selected.probability)}  ·  ${selected.score_A}:${selected.score_B}`;
    onPoint?.(cursor);
  };
  canvas.onpointermove = pointAt;
  canvas.onpointerdown = pointAt;
  canvas.onpointerleave = () => { tooltip.style.display = "none"; };
}

export function drawCalibrationChart({ canvas, curves }) {
  if (!canvas || !curves) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const padding = { left: 39, right: 18, top: 18, bottom: 32 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (value) => padding.left + value * plotWidth;
  const y = (value) => padding.top + (1 - value) * plotHeight;
  const line = css("--line");
  const muted = css("--muted");
  const accent = css("--accent");
  context.font = "10px SFMono-Regular, Consolas, monospace";
  context.fillStyle = muted;
  context.strokeStyle = line;
  context.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach((value) => {
    context.beginPath(); context.moveTo(x(0), y(value)); context.lineTo(x(1), y(value)); context.stroke();
    context.fillText(`${Math.round(value * 100)}%`, 2, y(value) + 3);
  });
  context.beginPath(); context.setLineDash([4, 4]); context.moveTo(x(0), y(0)); context.lineTo(x(1), y(1)); context.stroke(); context.setLineDash([]);
  const colors = [accent, css("--cool"), muted];
  Object.entries(curves).forEach(([name, curve], curveIndex) => {
    if (!curve.predicted_probability?.length) return;
    context.beginPath();
    curve.predicted_probability.forEach((value, index) => index ? context.lineTo(x(value), y(curve.observed_frequency[index])) : context.moveTo(x(value), y(curve.observed_frequency[index])));
    context.strokeStyle = colors[curveIndex % colors.length]; context.lineWidth = curveIndex === 0 ? 2.5 : 1.6; context.stroke();
    curve.predicted_probability.forEach((value, index) => { context.beginPath(); context.arc(x(value), y(curve.observed_frequency[index]), 2.8, 0, Math.PI * 2); context.fillStyle = colors[curveIndex % colors.length]; context.fill(); });
  });
  context.fillStyle = muted;
  context.fillText(getLanguage() === "vi" ? "Xác suất dự đoán" : "Predicted probability", Math.max(padding.left, width / 2 - 44), height - 7);
}
