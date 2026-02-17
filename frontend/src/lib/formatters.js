export const DASH = "—";

export function isNum(x) {
  return typeof x === "number" && !Number.isNaN(x);
}

export function formatPct(x, digits = 2) {
  if (!isNum(x)) return DASH;
  return `${(x * 100).toFixed(digits)}%`;
}

export function formatNum(x, digits = 2) {
  if (!isNum(x)) return DASH;
  return x.toFixed(digits);
}

export function formatMoney(x, digits = 2) {
  if (!isNum(x)) return DASH;
  return x.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPrice(x) {
  if (!isNum(x)) return DASH;

  const ax = Math.abs(x);

  // Adaptive precision for prices
  let digits = 2;
  if (ax > 0 && ax < 0.1) digits = 6;
  else if (ax < 1) digits = 4;

  return formatMoney(x, digits);
}

/**
 * Returns object with numeric fields = (scen - base).
 * Non-numerics become null (so blocks can render "—").
 */
export function diffObjects(base, scen) {
  if (!base || !scen) return null;
  const out = {};
  const keys = new Set([...Object.keys(base), ...Object.keys(scen)]);
  keys.forEach((k) => {
    const a = base[k];
    const b = scen[k];
    out[k] = isNum(a) && isNum(b) ? b - a : null;
  });
  return out;
}

