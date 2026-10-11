import { t } from "./i18n.js";
import { buyLabel, credits, metric, modelLabel, number, percent } from "./formatters.js";

const $ = (selector) => document.querySelector(selector);

export function renderSummary(data) {
  $("#stat-maps").textContent = number(data.data_summary.maps.test);
  $("#stat-samples").textContent = number(data.data_summary.prefix_samples);
  $("#stat-rows").textContent = number(data.data_summary.round_rows);
  $("#stat-model").textContent = data.main_model.replace(" + mean", "");
  $("#hero-model").textContent = modelLabel(data.main_model);
  $("#hero-split").textContent = `VCT ${data.source.seasons_used.at(-1)}`;
  $("#hero-scope").textContent = `${number(data.data_summary.features.length)} ${t("states") === "states" ? "features" : "đặc trưng"}`;
  $("#top-source-link").href = data.source.url;
  $("#source-link").href = data.source.url;
}

export function renderMapChoices(data, filter, onSelect) {
  const maps = data.maps.filter((map) => `${map.map} ${map.team_A} ${map.team_B}`.toLowerCase().includes(filter.trim().toLowerCase()));
  const list = $("#map-list");
  const select = $("#map-select");
  list.replaceChildren(); select.replaceChildren();
  $("#map-count").textContent = `${maps.length} / ${data.maps.length}`;
  if (!maps.length) { const empty = document.createElement("p"); empty.className = "map-empty"; empty.textContent = t("noMap"); list.append(empty); return []; }
  maps.forEach((map) => {
    const index = data.maps.indexOf(map);
    const option = document.createElement("option"); option.value = String(index); option.textContent = `${map.map} / ${map.team_A} vs ${map.team_B}`; select.append(option);
    const button = document.createElement("button"); button.type = "button"; button.className = "map-item"; button.dataset.index = String(index); button.innerHTML = `<strong></strong><span></span>`; button.querySelector("strong").textContent = map.map; button.querySelector("span").textContent = `${map.team_A} / ${map.team_B}`; button.addEventListener("click", () => onSelect(index)); list.append(button);
  });
  return maps;
}

export function setActiveMap(index) {
  document.querySelectorAll(".map-item").forEach((item) => item.classList.toggle("active", Number(item.dataset.index) === index));
  const select = $("#map-select"); if (select) select.value = String(index);
}

export function renderMap(data, mapIndex, pointIndex, drawChart) {
  const map = data.maps[mapIndex]; const trajectory = map.trajectory || []; const point = trajectory[pointIndex] || trajectory.at(-1);
  if (!point) { $("#match-map").textContent = "MAP / —"; return; }
  $("#match-map").textContent = `MAP / ${map.map}`; $("#team-a").textContent = map.team_A; $("#team-b").textContent = map.team_B; $("#context-team-a").textContent = map.team_A; $("#context-team-b").textContent = map.team_B; $("#final-a").textContent = map.final_score_A; $("#final-b").textContent = map.final_score_B; $("#probability-team").textContent = map.team_A; $("#current-probability").textContent = (point.probability * 100).toFixed(1); $("#current-round").textContent = `${t("round")} ${point.round}`; $("#current-score").textContent = `${point.score_A} : ${point.score_B}`; $("#round-title").textContent = `${t("round")} ${point.round} / ${point.score_A} : ${point.score_B}`; $("#round-slider-value").textContent = `${t("round")} ${point.round}`; $("#buy-a").textContent = buyLabel(point.buy_A); $("#buy-b").textContent = buyLabel(point.buy_B); $("#loadout-a").textContent = credits(point.loadout_A); $("#loadout-b").textContent = credits(point.loadout_B); $("#credits-a").textContent = credits(point.credits_A); $("#credits-b").textContent = credits(point.credits_B); $("#chart-end").textContent = `${t("round").toUpperCase()} ${trajectory.at(-1).round}`;
  const change = $("#change-pill"); change.classList.remove("positive", "negative"); if (point.probability_change == null) { change.textContent = t("openingState"); } else { const direction = point.probability_change >= 0 ? "positive" : "negative"; change.classList.add(direction); change.textContent = `${point.probability_change >= 0 ? "+" : ""}${(point.probability_change * 100).toFixed(1)} pts / ${t(point.probability_change >= 0 ? "positive" : "negative")}`; }
  renderAgents("#agents-a", map.agents_A); renderAgents("#agents-b", map.agents_B);
  const slider = $("#round-slider"); slider.max = String(Math.max(0, trajectory.length - 1)); slider.value = String(pointIndex); slider.setAttribute("aria-valuetext", `${t("round")} ${point.round}`);
  drawChart(trajectory);
}

function renderAgents(selector, agents) { const container = $(selector); container.replaceChildren(); agents.forEach((agent) => { const chip = document.createElement("span"); chip.className = "agent-chip"; chip.textContent = agent; container.append(chip); }); }

export function renderModels(data) {
  const sorted = [...data.models].sort((a, b) => a.test.log_loss - b.test.log_loss); const table = $("#model-table"); table.replaceChildren();
  sorted.forEach((model, index) => { const row = document.createElement("tr"); if (!index) row.className = "best"; [modelLabel(model.model), metric(model.test.log_loss), metric(model.test.brier), metric(model.test.roc_auc), percent(model.test.accuracy)].forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }); table.append(row); });
  const best = sorted[0]; $("#best-model-value").textContent = metric(best.test.log_loss); $("#best-model-detail").textContent = `${modelLabel(best.model)} ${t("bestNote")}`; $("#model-note").textContent = `${modelLabel(best.model)} ${t("bestNote")} ${t("modelsDescription")}`;
}

export function renderCheckpoints(data) {
  const model = data.models.find((item) => item.model === "GRU + mean"); const list = $("#checkpoint-list"); list.replaceChildren(); if (!model?.test) return;
  data.checkpoint_metrics.forEach((checkpoint) => { const metrics = checkpoint.models["GRU + mean"]; if (!metrics) return; const item = document.createElement("div"); item.className = "checkpoint-item"; item.innerHTML = `<label></label><div class="checkpoint-bar"><span></span></div><strong class="checkpoint-value"></strong>`; item.querySelector("label").textContent = `${t("checkpointRound")} ${checkpoint.round}`; item.querySelector("span").style.transform = `scaleX(${metrics.accuracy})`; item.querySelector("strong").textContent = percent(metrics.accuracy); list.append(item); });
}

export function renderCalibrationLegend(data) {
  const legend = $("#calibration-legend"); legend.replaceChildren(); ["GRU + mean", "XGBoost", "Logistic Regression"].forEach((name, index) => { const item = document.createElement("div"); item.className = "calibration-legend-item"; const line = document.createElement("i"); if (index) line.className = "neutral"; item.append(line, document.createTextNode(modelLabel(name))); legend.append(item); });
}

export function renderDictionary() {
  const items = [["dictScore", "dictScoreDesc", "score_A / score_B"], ["dictEconomy", "dictEconomyDesc", "loadout / credits"], ["dictBuy", "dictBuyDesc", "buy_level"], ["dictAgents", "dictAgentsDesc", "agents_A / agents_B"]]; const grid = $("#dictionary-grid"); grid.replaceChildren(); items.forEach(([title, description, field]) => { const item = document.createElement("article"); item.className = "dictionary-item"; item.innerHTML = `<strong></strong><p></p><span></span>`; item.querySelector("strong").textContent = t(title); item.querySelector("p").textContent = t(description); item.querySelector("span").textContent = field; grid.append(item); });
}
