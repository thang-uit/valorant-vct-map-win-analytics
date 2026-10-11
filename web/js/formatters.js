import { getLanguage, t } from "./i18n.js";

export function number(value) {
  return new Intl.NumberFormat(getLanguage() === "vi" ? "vi-VN" : "en-US").format(value);
}

export function credits(value) {
  return `${number(Math.round(value))} cr`;
}

export function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function metric(value, digits = 4) {
  return Number(value).toFixed(digits);
}

export function buyLabel(value) {
  if (!value) return t("unknown");
  const [level, range] = value.split(": ");
  const levels = getLanguage() === "vi"
    ? { Eco: "Eco", "Semi-eco": "Semi-eco", "Semi-buy": "Semi-buy", "Full buy": "Full buy" }
    : { Eco: "Eco", "Semi-eco": "Semi-eco", "Semi-buy": "Semi-buy", "Full buy": "Full buy" };
  return `${levels[level] || level}${range ? ` / ${range}` : ""}`;
}

export function modelLabel(value) {
  return value.replace(" + agent attention", getLanguage() === "vi" ? " + attention Agent" : " + agent attention");
}
