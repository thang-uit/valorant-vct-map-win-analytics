const translations = {
  vi: {
    skipToContent: "Bỏ qua đến nội dung",
    navOverview: "Tổng quan", navExplorer: "Map explorer", navModels: "Mô hình", navMethod: "Phương pháp", source: "Nguồn dữ liệu", theme: "Giao diện",
    loading: "Đang tải dữ liệu phân tích...", errorKicker: "Không thể tải dữ liệu", errorTitle: "Dashboard chưa sẵn sàng", retry: "Thử lại",
    heroEyebrow: "VCT / POST-MATCH LAB", heroTitle: "Đọc từng round.<br /><em>Hiểu cả map.</em>", heroDescription: "Một phòng phân tích sau trận để xem điểm số, nền kinh tế và đội hình đặc vụ đã làm thay đổi xác suất thắng map như thế nào.", exploreCta: "Mở map explorer <span aria-hidden=\"true\">↘</span>", readMethod: "Xem phương pháp", experimentReady: "EXPERIMENT READY", signalModel: "Mô hình chính", signalSplit: "Tập kiểm tra", signalScope: "Phạm vi",
    overviewEyebrow: "THE DATASET", overviewTitle: "Một mùa giải, nhìn từ trạng thái của từng round.", statMaps: "MAP KIỂM TRA", statMapsDetail: "holdout VCT 2024", statStates: "TRẠNG THÁI ROUND", statStatesDetail: "sau round 3 trở đi", statRows: "DÒNG DỮ LIỆU", statRowsDetail: "economy + score + agents", statModel: "MÔ HÌNH CHÍNH", statModelDetail: "ghi nhớ chuỗi thời gian",
    explorerEyebrow: "MAP EXPLORER", explorerTitle: "Xem lại một map theo từng điểm ngoặt.", explorerDescription: "Chọn map, kéo đến round cần xem hoặc bật phát tự động để theo dõi xác suất thay đổi theo diễn biến trận.", searchMaps: "Tìm map hoặc đội", searchPlaceholder: "Tìm theo map, đội...", selectMap: "Chọn map", availableMaps: "MAP TRONG DEMO", mapRailNote: "24 map đại diện được đưa vào payload demo để trang tải nhanh.", probabilityLabel: "XÁC SUẤT THẮNG MAP", probabilityCaption: "ước tính sau round hiện tại", currentState: "TRẠNG THÁI HIỆN TẠI", chartLegend: "Xác suất Team A thắng map", chartHint: "Di chuột hoặc chạm vào đường biểu diễn", chartStart: "ROUND 03", roundEstimate: "ROUND-BY-ROUND ESTIMATE", playRound: "Phát diễn biến", play: "Phát diễn biến", pause: "Tạm dừng", roundTimeline: "DÒNG THỜI GIAN ROUND", roundContext: "ROUND CONTEXT", teamA: "Đội A", teamB: "Đội B", remainingCredits: "Tiền còn lại",
    modelsEyebrow: "MODEL CHECK", modelsTitle: "Mô hình học sâu phải được đặt cạnh baseline.", modelsDescription: "Kết quả trên tập kiểm tra 2024. Log Loss và Brier Score càng thấp càng tốt.", bestModel: "BEST LOG LOSS", comparisonTable: "SO SÁNH MÔ HÌNH", testHoldout: "2024 HOLDOUT", modelTableCaption: "So sánh các mô hình trên tập kiểm tra", modelColumn: "Mô hình", accuracy: "Độ chính xác", checkpoints: "ĐỘ CHÍNH XÁC THEO MỐC ROUND", checkpointNote: "GRU + mean", bestNote: "có Log Loss thấp nhất trên tập kiểm tra 2024.",
    calibrationEyebrow: "PROBABILITY QUALITY", calibrationTitle: "Xác suất tốt phải đáng tin, không chỉ đúng.", calibrationDescription: "Calibration curve so sánh xác suất mô hình dự đoán với tần suất thắng quan sát được.", perfectCalibration: "Hiệu chỉnh lý tưởng", modelCalibration: "Mô hình",
    methodEyebrow: "WHAT THIS IS", methodTitle: "Một công cụ xem lại trận đấu,<br /><em>không phải lời tiên tri live.</em>", methodParagraphOne: "Mỗi điểm trên biểu đồ được tạo sau khi một round kết thúc. Mô hình nhận chuỗi lịch sử round, tỷ số, trạng thái kinh tế, map và đội hình năm đặc vụ, rồi ước tính Team A có thắng cả map hay không.", methodParagraphTwo: "Trang này là mô phỏng sau trận. Nó không theo dõi trận live, không khuyến nghị mua súng hay skill, và không khẳng định một đặc vụ là nguyên nhân của chiến thắng. Giá trị thực tế là giúp analyst tìm ra những thời điểm đáng mở lại trong VOD.", openSource: "Mở nguồn dữ liệu ↗", dictionaryEyebrow: "FIELD GUIDE", dictionaryTitle: "Các tín hiệu mô hình đọc sau mỗi round.", footerText: "Phân tích sau trận · VCT 2023–2024", viewGithub: "Xem mã nguồn ↗",
    noMap: "Không tìm thấy map phù hợp.", noHistory: "Map này chưa có dữ liệu diễn biến.", openingState: "TRẠNG THÁI MỞ ĐẦU", positive: "tăng", negative: "giảm", round: "Round", maps: "map", states: "trạng thái", sourceError: "Hãy chạy scripts/train.py trước khi mở web.", checkpointRound: "Sau round", current: "hiện tại", unknown: "Chưa có dữ liệu",
    dictScore: "Tỷ số hiện tại", dictScoreDesc: "Số round Team A và Team B đã thắng tính đến thời điểm quan sát.", dictEconomy: "Nền kinh tế", dictEconomyDesc: "Giá trị loadout và số credits còn lại của hai đội.", dictBuy: "Mức mua", dictBuyDesc: "Mã hóa trạng thái Eco, Semi-eco, Semi-buy và Full buy.", dictAgents: "Đội hình đặc vụ", dictAgentsDesc: "Năm đặc vụ của mỗi đội, được đưa vào model bằng embedding.",
  },
  en: {
    skipToContent: "Skip to content", navOverview: "Overview", navExplorer: "Map explorer", navModels: "Models", navMethod: "Method", source: "Data source", theme: "Appearance",
    loading: "Loading analysis data...", errorKicker: "Data unavailable", errorTitle: "Dashboard is not ready", retry: "Try again",
    heroEyebrow: "VCT / POST-MATCH LAB", heroTitle: "Read the round.<br /><em>Understand the map.</em>", heroDescription: "A post-match room for tracing how score, economy and agent composition changed the estimated chance of winning a map.", exploreCta: "Open map explorer <span aria-hidden=\"true\">↘</span>", readMethod: "Read the method", experimentReady: "EXPERIMENT READY", signalModel: "Main model", signalSplit: "Test split", signalScope: "Scope",
    overviewEyebrow: "THE DATASET", overviewTitle: "One season, read through the state of every round.", statMaps: "TEST MAPS", statMapsDetail: "VCT 2024 holdout", statStates: "ROUND STATES", statStatesDetail: "after round 3 onward", statRows: "DATA ROWS", statRowsDetail: "economy + score + agents", statModel: "MAIN MODEL", statModelDetail: "sequence memory",
    explorerEyebrow: "MAP EXPLORER", explorerTitle: "Revisit a map, turning point by turning point.", explorerDescription: "Choose a map, move to a round, or play the sequence to see how the estimate changes with the match.", searchMaps: "Find a map or team", searchPlaceholder: "Search map, team...", selectMap: "Select a map", availableMaps: "MAPS IN DEMO", mapRailNote: "24 representative maps are included in the demo payload to keep the page fast.", probabilityLabel: "MAP-WIN PROBABILITY", probabilityCaption: "estimated after the current round", currentState: "CURRENT STATE", chartLegend: "Team A map-win probability", chartHint: "Hover or touch the line", chartStart: "ROUND 03", roundEstimate: "ROUND-BY-ROUND ESTIMATE", playRound: "Play sequence", play: "Play sequence", pause: "Pause", roundTimeline: "ROUND TIMELINE", roundContext: "ROUND CONTEXT", teamA: "Team A", teamB: "Team B", remainingCredits: "Remaining credits",
    modelsEyebrow: "MODEL CHECK", modelsTitle: "A deep model earns its place beside the baselines.", modelsDescription: "Results on the 2024 test holdout. Lower Log Loss and Brier Score are better.", bestModel: "BEST LOG LOSS", comparisonTable: "MODEL COMPARISON", testHoldout: "2024 HOLDOUT", modelTableCaption: "Model comparison on the test set", modelColumn: "Model", accuracy: "Accuracy", checkpoints: "ACCURACY AT ROUND CHECKPOINTS", checkpointNote: "GRU + mean", bestNote: "has the lowest Log Loss on the 2024 test holdout.",
    calibrationEyebrow: "PROBABILITY QUALITY", calibrationTitle: "Good probabilities need to be trusted, not only correct.", calibrationDescription: "The calibration curve compares predicted probability with the observed win frequency.", perfectCalibration: "Perfect calibration", modelCalibration: "Model",
    methodEyebrow: "WHAT THIS IS", methodTitle: "A film-room instrument,<br /><em>not a live oracle.</em>", methodParagraphOne: "Each chart point is generated after a round ends. The model receives round history, score, economy state, map and five-agent compositions, then estimates whether Team A will win the whole map.", methodParagraphTwo: "This page is a post-match simulation. It does not watch a live game, recommend purchases or skills, or claim that an agent caused a win. Its practical use is helping an analyst find turning points worth revisiting in a VOD.", openSource: "Open data source ↗", dictionaryEyebrow: "FIELD GUIDE", dictionaryTitle: "The signals the model reads after each round.", footerText: "Post-match analysis · VCT 2023–2024", viewGithub: "View source ↗",
    noMap: "No matching map found.", noHistory: "This map has no trajectory data.", openingState: "OPENING STATE", positive: "up", negative: "down", round: "Round", maps: "maps", states: "states", sourceError: "Run scripts/train.py before opening the web app.", checkpointRound: "After round", current: "current", unknown: "No data",
    dictScore: "Current score", dictScoreDesc: "Rounds won by Team A and Team B at the observed point in the map.", dictEconomy: "Economy state", dictEconomyDesc: "Loadout value and remaining credits for both teams.", dictBuy: "Buy level", dictBuyDesc: "The Eco, Semi-eco, Semi-buy and Full buy categories encoded for the model.", dictAgents: "Agent composition", dictAgentsDesc: "Five agents for each team, represented through learned embeddings.",
  },
};

let currentLanguage = localStorage.getItem("round-state-language") || "vi";

export function t(key) {
  return translations[currentLanguage]?.[key] ?? translations.vi[key] ?? key;
}

export function getLanguage() { return currentLanguage; }

export function setLanguage(language) {
  currentLanguage = translations[language] ? language : "vi";
  localStorage.setItem("round-state-language", currentLanguage);
  return currentLanguage;
}

export function applyTranslations() {
  document.documentElement.lang = currentLanguage;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const value = t(element.dataset.i18n);
    if (element.dataset.i18n.includes("Title") || element.dataset.i18n === "methodTitle" || element.dataset.i18n === "heroTitle" || element.dataset.i18n === "exploreCta") element.innerHTML = value;
    else element.textContent = value;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => { element.placeholder = t(element.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => { element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel)); });
  document.querySelectorAll(".language-button").forEach((button) => {
    const active = button.dataset.language === currentLanguage;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}
