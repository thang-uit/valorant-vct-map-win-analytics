const DATA_URL = "data/demo.json";

function isObject(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }

function validateData(data) {
  if (!isObject(data) || !Array.isArray(data.maps) || !Array.isArray(data.models) || !isObject(data.data_summary)) throw new Error("Payload demo không đúng cấu trúc.");
  if (!data.maps.length || !data.models.length) throw new Error("Payload demo không có dữ liệu map hoặc mô hình.");
  return data;
}

export async function loadDemoData() {
  const response = await fetch(DATA_URL, { cache: "no-cache" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return validateData(await response.json());
}
