const NUMERIC = ['year', 'condition', 'odometer'];
const CATEGORICAL = ['make', 'body', 'transmission'];

/**
 * Fit encoders on training rows and turn rows into numeric feature matrices.
 */
export function fitFeaturePipeline(rows, topMakes = 15, topBodies = 8) {
  const clean = rows.map(cleanRow).filter(Boolean);

  const makeCounts = countField(clean, 'make');
  const bodyCounts = countField(clean, 'body');

  const topMakeList = topKeys(makeCounts, topMakes);
  const topBodyList = topKeys(bodyCounts, topBodies);

  const featureNames = [
    'bias',
    'year',
    'condition',
    'odometer',
    'vehicle_age',
    ...topMakeList.map((m) => `make_${slug(m)}`),
    ...topBodyList.map((b) => `body_${slug(b)}`),
    'transmission_automatic',
  ];

  const stats = {
    year: meanStd(clean.map((r) => r.year)),
    condition: meanStd(clean.map((r) => r.condition)),
    odometer: meanStd(clean.map((r) => r.odometer)),
    vehicle_age: meanStd(clean.map((r) => r.vehicleAge)),
  };

  return {
    featureNames,
    topMakeList,
    topBodyList,
    stats,
    transform(rowsToTransform) {
      const kept = [];
      const X = [];
      const y = [];

      for (const raw of rowsToTransform) {
        const row = cleanRow(raw);
        if (!row) continue;

        const features = [
          1,
          z(row.year, stats.year),
          z(row.condition, stats.condition),
          z(row.odometer, stats.odometer),
          z(row.vehicleAge, stats.vehicle_age),
          ...topMakeList.map((m) => (row.make === m ? 1 : 0)),
          ...topBodyList.map((b) => (row.body === b ? 1 : 0)),
          row.transmission === 'automatic' ? 1 : 0,
        ];

        kept.push(row);
        X.push(features);
        y.push(row.sellingprice);
      }

      return { X, y, rows: kept };
    },
  };
}

function cleanRow(row) {
  const year = Number(row.year);
  const condition = Number(row.condition);
  const odometer = Number(row.odometer);
  const sellingprice = Number(row.sellingprice);
  const saleYear = parseSaleYear(row.saledate) ?? year;

  if (![year, condition, odometer, sellingprice].every(Number.isFinite)) return null;
  if (sellingprice <= 0 || odometer < 0 || year < 1980 || year > 2030) return null;

  return {
    year,
    condition,
    odometer,
    sellingprice,
    vehicleAge: Math.max(0, saleYear - year),
    make: (row.make || 'unknown').toLowerCase(),
    body: (row.body || 'unknown').toLowerCase(),
    transmission: (row.transmission || 'unknown').toLowerCase(),
    model: row.model || '',
    trim: row.trim || '',
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

export { NUMERIC, CATEGORICAL };
