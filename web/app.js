(() => {
  "use strict";

  const state = { data: null, selectedMap: 0, selectedRound: 0, points: [] };
  const $ = (selector) => document.querySelector(selector);

  const formatNumber = (value) => new Intl.NumberFormat("en-US").format(value);
  const formatCredits = (value) => `${new Intl.NumberFormat("en-US").format(Math.round(value))} cr`;
  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  const shortModelName = (name) => name.replace(" + agent attention", " + attention");

  function setText(selector, value) {
    const element = $(selector);
    if (element) element.textContent = value;
  }

  function renderSummary() {
    const { data } = state;
    setText("#data-note", data.data_note);
    setText("#stat-maps", formatNumber(data.data_summary.maps.test));
    setText("#stat-samples", formatNumber(data.data_summary.prefix_samples));
    setText("#map-count", `${data.maps.length} / ${data.data_summary.maps.test}`);
    $("#source-link").href = data.source.url;
  }

  function renderMapChoices() {
    const list = $("#map-list");
    const select = $("#map-select");
    list.replaceChildren();
    select.replaceChildren();
    state.data.maps.forEach((map, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${map.map} / ${map.team_A} vs ${map.team_B}`;
      select.append(option);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "map-item";
      button.dataset.index = index;
      button.innerHTML = `<strong>${map.map}</strong><span>${map.team_A} / ${map.team_B}</span>`;
      button.addEventListener("click", () => selectMap(index));
      list.append(button);
    });
    select.addEventListener("change", (event) => selectMap(Number(event.target.value)));
  }

  function selectMap(index) {
    state.selectedMap = index;
    state.selectedRound = state.data.maps[index].trajectory.length - 1;
    $("#map-select").value = String(index);
    document.querySelectorAll(".map-item").forEach((item) => {
      item.classList.toggle("active", Number(item.dataset.index) === index);
    });
    renderMap();
  }

  function renderAgents(selector, agents) {
    const container = $(selector);
    container.replaceChildren();
    agents.forEach((agent) => {
      const chip = document.createElement("span");
      chip.className = "agent-chip";
      chip.textContent = agent;
      container.append(chip);
    });
  }

  function renderMap() {
    const map = state.data.maps[state.selectedMap];
    const trajectory = map.trajectory;
    const point = trajectory[state.selectedRound];
    state.points = trajectory;
    setText("#match-map", `MAP / ${map.map}`);
    setText("#team-a", map.team_A);
    setText("#team-b", map.team_B);
    setText("#context-team-a", map.team_A);
    setText("#context-team-b", map.team_B);
    setText("#final-a", map.final_score_A);
    setText("#final-b", map.final_score_B);
    setText("#probability-team", map.team_A);
    setText("#current-probability", (point.probability * 100).toFixed(1));
    setText("#current-round", `Round ${point.round}`);
    setText("#current-score", `${point.score_A} : ${point.score_B}`);
    setText("#round-title", `Round ${point.round} / ${point.score_A} : ${point.score_B}`);
    setText("#change-pill", point.probability_change == null ? "OPENING STATE" : `${point.probability_change >= 0 ? "+" : ""}${(point.probability_change * 100).toFixed(1)} pts`);
    setText("#buy-a", point.buy_A.replace(": ", " / "));
    setText("#buy-b", point.buy_B.replace(": ", " / "));
    setText("#loadout-a", formatCredits(point.loadout_A));
    setText("#loadout-b", formatCredits(point.loadout_B));
    setText("#credits-a", formatCredits(point.credits_A));
    setText("#credits-b", formatCredits(point.credits_B));
    setText("#chart-end", `ROUND ${trajectory[trajectory.length - 1].round}`);
    renderAgents("#agents-a", map.agents_A);
    renderAgents("#agents-b", map.agents_B);
    drawChart();
  }

  function drawChart() {
    const canvas = $("#probability-chart");
    const frame = $(".chart-frame");
    const ratio = window.devicePixelRatio || 1;
    const width = frame.clientWidth;
    const height = frame.clientHeight;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const padding = { left: 11, right: 11, top: 10, bottom: 10 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const points = state.points;
    const x = (index) => padding.left + (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth);
    const y = (value) => padding.top + (1 - value) * plotHeight;
    context.lineWidth = 2;
    context.strokeStyle = "#ef4b32";
    context.beginPath();
    points.forEach((point, index) => index === 0 ? context.moveTo(x(index), y(point.probability)) : context.lineTo(x(index), y(point.probability)));
    context.stroke();
    context.fillStyle = "#fbfaf7";
    context.strokeStyle = "#ef4b32";
    points.forEach((point, index) => {
      context.beginPath();
      context.arc(x(index), y(point.probability), 3.3, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    });
    canvas.onmousemove = (event) => {
      const box = canvas.getBoundingClientRect();
      const cursorIndex = Math.max(0, Math.min(points.length - 1, Math.round(((event.clientX - box.left) / box.width) * (points.length - 1))));
      const selected = points[cursorIndex];
      const tooltip = $("#chart-tooltip");
      tooltip.style.display = "block";
      tooltip.style.left = `${x(cursorIndex)}px`;
      tooltip.style.top = `${y(selected.probability)}px`;
      tooltip.textContent = `R${selected.round}  ${formatPercent(selected.probability)}  /  ${selected.score_A}:${selected.score_B}`;
    };
    canvas.onmouseleave = () => { $("#chart-tooltip").style.display = "none"; };
  }

  function renderModels() {
    const table = $("#model-table");
    const sorted = [...state.data.models].sort((a, b) => a.test.log_loss - b.test.log_loss);
    table.replaceChildren();
    sorted.forEach((model, index) => {
      const row = document.createElement("tr");
      if (index === 0) row.className = "best";
      [shortModelName(model.model), model.test.log_loss.toFixed(4), model.test.brier.toFixed(4), model.test.roc_auc.toFixed(4), formatPercent(model.test.accuracy)].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      table.append(row);
    });
    const best = sorted[0];
    setText("#model-note", `${shortModelName(best.model)} có Log Loss thấp nhất trên tập kiểm tra 2024 (${best.test.log_loss.toFixed(4)}). GRU có attention theo đội hình được giữ lại để minh họa đúng đề tài, dù kết quả chưa vượt GRU gộp trung bình trong lần chạy này.`);
  }

  async function init() {
    try {
      const response = await fetch("data/demo.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.data = await response.json();
      renderSummary();
      renderMapChoices();
      renderModels();
      selectMap(0);
      new ResizeObserver(drawChart).observe($(".chart-frame"));
    } catch (error) {
      setText("#data-note", "Không tải được demo.json. Hãy chạy scripts/train.py trước khi mở web.");
      console.error(error);
    }
  }

  window.addEventListener("resize", drawChart);
  init();
})();
