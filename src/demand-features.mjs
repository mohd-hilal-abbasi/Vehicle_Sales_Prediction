import { labelDemand } from './demand-labels.mjs';

/**
 * Feature pipeline for demand / days-to-sale prediction.
 * Prediction-time inputs: year, make, model, state, odometer, condition.
 * Sale price / MMR are used only when labeling training data.
 */
export function fitDemandPipeline(rawRows, { topMakes = 20, topModels = 25, topStates = 15 } = {}) {
  const base = rawRows
    .map(cleanDemandRow)
    .filter((row) => row && Number.isFinite(row.sellingprice) && row.sellingprice > 0);
  const labeled = labelDemand(base);

  const makeCounts = countField(labeled, 'make');
  const modelCounts = countField(labeled, 'model');
  const stateCounts = countField(labeled, 'state');
  const bodyCounts = countField(labeled, 'body');

  const topMakeList = topKeys(makeCounts, topMakes);
  const topModelList = topKeys(modelCounts, topModels);
  const topStateList = topKeys(stateCounts, topStates);
  const topBodyList = topKeys(bodyCounts, 8);

  const stats = {
    year: meanStd(labeled.map((r) => r.year)),
    condition: meanStd(labeled.map((r) => r.condition)),
    odometer: meanStd(labeled.map((r) => r.odometer)),
    vehicle_age: meanStd(labeled.map((r) => r.vehicleAge)),
    miles_per_year: meanStd(labeled.map((r) => r.milesPerYear)),
  };

  const featureNames = [
    'bias',
    'year',
    'condition',
    'odometer',
    'vehicle_age',
    'miles_per_year',
    ...topMakeList.map((m) => `make_${slug(m)}`),
    ...topModelList.map((m) => `model_${slug(m)}`),
    ...topStateList.map((s) => `state_${slug(s)}`),
    ...topBodyList.map((b) => `body_${slug(b)}`),
    'transmission_automatic',
  ];

  return {
    featureNames,
    topMakeList,
    topModelList,
    topStateList,
    topBodyList,
    stats,
    labeled,
    transform(rowsToTransform, { alreadyLabeled = false } = {}) {
      const source = alreadyLabeled
        ? rowsToTransform
        : labelDemand(rowsToTransform.map(cleanDemandRow).filter(Boolean));

      const X = [];
      const yDays = [];
      const yClass = [];
      const rows = [];

      for (const row of source) {
        X.push(vectorize(row, { topMakeList, topModelList, topStateList, topBodyList, stats }));
        yDays.push(row.daysToSale);
        yClass.push(row.demandClass);
        rows.push(row);
      }

      return { X, yDays, yClass, rows };
    },
  };
}

export function vectorize(row, { topMakeList, topModelList, topStateList, topBodyList, stats }) {
  return [
    1,
    z(row.year, stats.year),
    z(row.condition, stats.condition),
    z(row.odometer, stats.odometer),
    z(row.vehicleAge, stats.vehicle_age),
    z(row.milesPerYear, stats.miles_per_year),
    ...topMakeList.map((m) => (row.make === m ? 1 : 0)),
    ...topModelList.map((m) => (row.model === m ? 1 : 0)),
    ...topStateList.map((s) => (row.state === s ? 1 : 0)),
    ...topBodyList.map((b) => (row.body === b ? 1 : 0)),
    row.transmission === 'automatic' ? 1 : 0,
  ];
}

export function cleanDemandRow(row) {
  const year = Number(row.year);
  const condition = Number(row.condition);
  const odometer = Number(row.odometer);
  const sellingprice = Number(row.sellingprice);
  const mmr = Number(row.mmr);
  const saleYear = parseSaleYear(row.saledate) ?? year;

  if (![year, condition, odometer].every(Number.isFinite)) return null;
  if (odometer < 0 || year < 1980 || year > 2030) return null;

  // Training rows need a sale price; prediction rows may omit it
  const hasSale = Number.isFinite(sellingprice) && sellingprice > 0;

  const vehicleAge = Math.max(0, saleYear - year);
  const milesPerYear = odometer / Math.max(vehicleAge, 0.5);

  return {
    year,
    condition,
    odometer,
    sellingprice: hasSale ? sellingprice : null,
    mmr: Number.isFinite(mmr) && mmr > 0 ? mmr : null,
    vehicleAge,
    milesPerYear,
    make: (row.make || 'unknown').toLowerCase(),
    model: (row.model || 'unknown').toLowerCase(),
    body: (row.body || 'unknown').toLowerCase(),
    state: (row.state || 'unknown').toLowerCase(),
    transmission: (row.transmission || 'unknown').toLowerCase(),
  };
}

function parseSaleYear(saledate) {
  if (!saledate) return null;
  const match = String(saledate).match(/\b(19|20)\d{2}\b/);
  return match ? Number(match[0]) : null;
}

function countField(rows, field) {
  const counts = new Map();
  for (const row of rows) {
    counts.set(row[field], (counts.get(row[field]) || 0) + 1);
  }
  return counts;
}

function topKeys(counts, n) {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k);
}

function meanStd(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return { mean, std: Math.sqrt(variance) || 1 };
}

function z(value, { mean, std }) {
  return (value - mean) / std;
}

function slug(value) {
  return String(value).replace(/[^a-z0-9]+/gi, '_').toLowerCase();
}
