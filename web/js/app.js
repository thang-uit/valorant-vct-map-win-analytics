import { loadDemoData } from "./data-service.js";
import { drawCalibrationChart, drawProbabilityChart } from "./charts.js";
import { applyTranslations, getLanguage, setLanguage, t } from "./i18n.js";
import { createStore } from "./state.js";
import { renderCalibrationLegend, renderCheckpoints, renderDictionary, renderMap, renderMapChoices, renderModels, renderSummary, setActiveMap } from "./ui.js";

const $ = (selector) => document.querySelector(selector);
const store = createStore({ data: null, mapIndex: 0, pointIndex: 0, filter: "", playing: false });
let stopPlayback = null;

function applyTheme(theme) { document.documentElement.dataset.theme = theme; localStorage.setItem("round-state-theme", theme); $("#theme-toggle").setAttribute("aria-label", theme === "dark" ? "Dùng giao diện sáng" : "Dùng giao diện tối"); }

function preferredTheme() { return localStorage.getItem("round-state-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); }

function drawCurrentProbability(points) { drawProbabilityChart({ canvas: $("#probability-chart"), frame: $("#probability-chart-frame"), tooltip: $("#chart-tooltip"), points, onPoint: (pointIndex) => store.setState({ pointIndex }) }); }

function renderAll() {
  const state = store.getState(); if (!state.data) return;
  applyTranslations(); renderSummary(state.data); renderModels(state.data); renderCheckpoints(state.data); renderCalibrationLegend(state.data); renderDictionary(); renderMapChoices(state.data, state.filter, selectMap); setActiveMap(state.mapIndex); renderMap(state.data, state.mapIndex, state.pointIndex, drawCurrentProbability); drawCalibrationChart({ canvas: $("#calibration-chart"), curves: state.data.calibration_curves }); updatePlaybackButton();
}

function selectMap(index) { const data = store.getState().data; const map = data?.maps[index]; if (!map) return; store.setState({ mapIndex: index, pointIndex: 0, playing: false }); stopPlayback?.(); renderAll(); document.querySelector("#explorer")?.scrollIntoView({ behavior: "smooth", block: "start" }); }

function updatePoint(index) { const state = store.getState(); const length = state.data?.maps[state.mapIndex]?.trajectory?.length || 1; store.setState({ pointIndex: Math.max(0, Math.min(index, length - 1)) }); renderMap(state.data, state.mapIndex, store.getState().pointIndex, drawCurrentProbability); }

function updatePlaybackButton() { const state = store.getState(); $("#play-icon").textContent = state.playing ? "Ⅱ" : "▶"; $("#play-label").textContent = state.playing ? t("pause") : t("play"); $("#play-toggle").setAttribute("aria-label", state.playing ? t("pause") : t("playRound")); }

function togglePlayback() {
  const state = store.getState(); if (state.playing) { store.setState({ playing: false }); stopPlayback?.(); stopPlayback = null; updatePlaybackButton(); return; }
  const trajectory = state.data.maps[state.mapIndex].trajectory; let index = state.pointIndex >= trajectory.length - 1 ? 0 : state.pointIndex; store.setState({ playing: true }); updatePlaybackButton(); const timer = window.setInterval(() => { index += 1; if (index >= trajectory.length) { window.clearInterval(timer); stopPlayback = null; store.setState({ playing: false }); updatePlaybackButton(); return; } updatePoint(index); }, 500); stopPlayback = () => window.clearInterval(timer);
}

function bindEvents() {
  $("#theme-toggle").addEventListener("click", () => { applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); const state = store.getState(); if (state.data) { drawCurrentProbability(state.data.maps[state.mapIndex].trajectory); drawCalibrationChart({ canvas: $("#calibration-chart"), curves: state.data.calibration_curves }); } });
  document.querySelectorAll(".language-button").forEach((button) => button.addEventListener("click", () => { setLanguage(button.dataset.language); renderAll(); }));
  $("#map-filter").addEventListener("input", (event) => { store.setState({ filter: event.target.value }); const state = store.getState(); renderMapChoices(state.data, state.filter, selectMap); setActiveMap(state.mapIndex); });
  $("#map-select").addEventListener("change", (event) => selectMap(Number(event.target.value)));
  $("#round-slider").addEventListener("input", (event) => updatePoint(Number(event.target.value)));
  $("#play-toggle").addEventListener("click", togglePlayback);
  $("#retry-button").addEventListener("click", init);
  const resizeObserver = new ResizeObserver(() => { const state = store.getState(); if (!state.data) return; drawCurrentProbability(state.data.maps[state.mapIndex].trajectory); drawCalibrationChart({ canvas: $("#calibration-chart"), curves: state.data.calibration_curves }); });
  resizeObserver.observe($("#probability-chart-frame")); resizeObserver.observe($("#calibration-chart"));
  const revealObserver = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) { entry.target.classList.add("is-visible"); revealObserver.unobserve(entry.target); } }), { threshold: .12 });
  document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));
}

async function init() {
  $("#loading-state").hidden = false; $("#error-state").hidden = true; $("#app-content").hidden = true; applyTranslations();
  try { const data = await loadDemoData(); store.setState({ data, mapIndex: 0, pointIndex: 0 }); $("#loading-state").hidden = true; $("#app-content").hidden = false; renderAll(); } catch (error) { $("#loading-state").hidden = true; $("#error-state").hidden = false; $("#error-message").textContent = `${error.message}. ${t("sourceError")}`; console.error(error); }
}

applyTheme(preferredTheme()); bindEvents(); init();
