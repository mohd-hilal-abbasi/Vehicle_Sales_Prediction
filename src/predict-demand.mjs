import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanDemandRow, vectorize } from './demand-features.mjs';
import { DAYS_HIGH_MAX, DAYS_MEDIUM_MAX } from './demand-labels.mjs';
import { predictSoftmax } from './demand-model.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const args = parseArgs(process.argv.slice(2));
const modelPath = path.resolve(root, args.model);
const model = JSON.parse(await readFile(modelPath, 'utf8'));

const saleYear = args.saleYear ? Number(args.saleYear) : new Date().getFullYear();
const raw = {
  year: args.year,
  make: args.make,
  model: args.modelName,
  state: args.state,
  body: args.body,
  transmission: args.transmission,
  condition: args.condition,
  odometer: args.odometer,
  saledate: String(saleYear),
};

const vehicle = cleanDemandRow(raw);
if (!vehicle) {
  console.error('Invalid vehicle inputs. Check --year, --condition, and --odometer.');
  process.exit(1);
}

const features = vectorize(vehicle, model);
const daysRaw = features.reduce((sum, value, i) => sum + value * model.daysTheta[i], 0);
const daysToSale = Math.max(1, Math.round(daysRaw));
const [{ demandClass: classified, probabilities }] = predictSoftmax([features], model.classifier);

// Prefer the classifier for inventory class; keep days as the planning number.
// If they disagree, nudge the displayed days into the predicted class band.
const daysToSaleAligned = alignDaysToClass(daysToSale, classified);

printReport({
  vehicle,
  demandClass: classified,
  daysToSale: daysToSaleAligned,
  probabilities,
  metrics: model.metrics?.test,
});

function alignDaysToClass(days, demandClass) {
  if (demandClass === 'HIGH') {
    if (days <= DAYS_HIGH_MAX) return days;
    // Classifier is more confident than the days model — use a typical fast-turn figure
    return Math.round((7 + DAYS_HIGH_MAX) / 2);
  }
  if (demandClass === 'MEDIUM') {
    return Math.min(Math.max(days, DAYS_HIGH_MAX + 1), DAYS_MEDIUM_MAX);
  }
  return Math.max(days, DAYS_MEDIUM_MAX + 1);
}

function printReport({ vehicle, demandClass, daysToSale, probabilities, metrics }) {
  const displayName = `${title(vehicle.make)} ${title(vehicle.model)}`;

  console.log(`Vehicle: ${displayName}`);
  console.log(`State: ${vehicle.state.toUpperCase()}`);
  console.log(`Year: ${vehicle.year}`);
  console.log(`Mileage: ${vehicle.odometer.toLocaleString()}`);
  console.log('');
  console.log(`Predicted demand: ${demandClass}`);
  console.log(`Expected days-to-sale: ${daysToSale} days`);
  console.log('');
  console.log('Class probabilities');
  for (const [name, p] of Object.entries(probabilities)) {
    console.log(`  ${name.padEnd(6)} ${(p * 100).toFixed(1)}%`);
  }

  if (metrics?.classifierAccuracy) {
    console.log('');
    console.log(
      `(holdout class accuracy ${(metrics.classifierAccuracy.accuracy * 100).toFixed(1)}%, ` +
        `days MAE ${metrics.days.mae.toFixed(1)}d)`,
    );
  }
}

function title(value) {
  return String(value)
    .split(/\s+/)
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(' ');
}

function parseArgs(argv) {
  const defaults = {
    model: 'models/demand-model.json',
    year: '2020',
    make: 'toyota',
    modelName: 'camry',
    state: 'ca',
    body: 'sedan',
    transmission: 'automatic',
    condition: '40',
    odometer: '45000',
    saleYear: null,
  };

  const aliases = {
    'model-name': 'modelName',
    'car-model': 'modelName',
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;

    const key = arg.slice(2);
    if (key === 'help') {
      printHelp();
      process.exit(0);
    }

    const mapped = aliases[key] || key;
    if (!(mapped in defaults)) {
      console.error(`Unknown option: ${arg}`);
      process.exit(1);
    }
    defaults[mapped] = argv[++i];
  }

  return defaults;
}

function printHelp() {
  console.log(`Usage:
  node src/predict-demand.mjs [options]

Example:
  node src/predict-demand.mjs --make toyota --model-name camry --state ca --year 2020 --odometer 45000

Options:
  --model path              Trained demand model (default models/demand-model.json)
  --make toyota
  --model-name camry
  --state ca
  --year 2020
  --odometer 45000
  --condition 40
  --body sedan
  --transmission automatic`);
}
