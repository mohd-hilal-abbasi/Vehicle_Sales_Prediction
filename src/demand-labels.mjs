/**
 * Derive demand labels from auction CSV fields.
 *
 * The dataset has no listing-date / days-on-lot column, so we build a
 * proxy demand score from market strength (sale vs MMR), condition,
 * mileage intensity, age, and model popularity — then map that to
 * days-to-sale and HIGH / MEDIUM / LOW classes for inventory planning.
 */

export const DEMAND_CLASSES = ['HIGH', 'MEDIUM', 'LOW'];

/** Days thresholds aligned with score tertiles (see labelDemand). */
export const DAYS_HIGH_MAX = 21;
export const DAYS_MEDIUM_MAX = 40;

/**
 * Attach demandScore, daysToSale, and demandClass to cleaned rows.
 */
export function labelDemand(rows) {
  const modelCounts = new Map();
  for (const row of rows) {
    const key = modelKey(row);
    modelCounts.set(key, (modelCounts.get(key) || 0) + 1);
  }

  const maxCount = Math.max(...modelCounts.values(), 1);

  const scored = rows.map((row) => {
    const popularity = (modelCounts.get(modelKey(row)) || 1) / maxCount;
    const market = marketStrength(row.sellingprice, row.mmr);
    const conditionScore = clamp(row.condition / 50, 0, 1);
    const mileageScore = mileageIntensityScore(row.odometer, row.vehicleAge);
    const ageScore = clamp(1 - row.vehicleAge / 12, 0, 1);

    const demandScore = clamp(
      0.3 * market +
        0.2 * conditionScore +
        0.2 * mileageScore +
        0.15 * popularity +
        0.15 * ageScore,
      0,
      1,
    );

    // High demand → ~7 days; low demand → ~70 days
    const daysToSale = Math.round(7 + (1 - demandScore) * 63);

    return {
      ...row,
      demandScore,
      daysToSale,
      popularity,
    };
  });

  // Balance classes by score tertiles (dealer-friendly mix)
  const sortedScores = scored.map((r) => r.demandScore).sort((a, b) => a - b);
  const lowCut = quantile(sortedScores, 1 / 3);
  const highCut = quantile(sortedScores, 2 / 3);

  return scored.map((row) => {
    let demandClass = 'MEDIUM';
    if (row.demandScore >= highCut) demandClass = 'HIGH';
    else if (row.demandScore < lowCut) demandClass = 'LOW';

    return { ...row, demandClass };
  });
}

export function classFromDays(days) {
  if (days <= DAYS_HIGH_MAX) return 'HIGH';
  if (days <= DAYS_MEDIUM_MAX) return 'MEDIUM';
  return 'LOW';
}

function marketStrength(sellingprice, mmr) {
  if (!Number.isFinite(mmr) || mmr <= 0) return 0.5;
  const ratio = sellingprice / mmr;
  return clamp((ratio - 0.85) / 0.2, 0, 1);
}

function mileageIntensityScore(odometer, vehicleAge) {
  const years = Math.max(vehicleAge, 0.5);
  const milesPerYear = odometer / years;
  return clamp(1 - (milesPerYear - 8_000) / 20_000, 0, 1);
}

function modelKey(row) {
  return `${row.make}|${row.model}`.toLowerCase();
}

function quantile(sorted, q) {
  if (sorted.length === 0) return 0;
  const idx = (sorted.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
