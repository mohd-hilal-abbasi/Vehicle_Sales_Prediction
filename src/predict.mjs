import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const args = parseArgs(process.argv.slice(2));
const modelPath = path.resolve(root, args.model);
const model = JSON.parse(await readFile(modelPath, 'utf8'));

const vehicle = {
  year: Number(args.year),
  condition: Number(args.condition),
  odometer: Number(args.odometer),
  make: String(args.make).toLowerCase(),
  body: String(args.body).toLowerCase(),
  transmission: String(args.transmission).toLowerCase(),
  vehicleAge: Math.max(0, (args.saleYear ? Number(args.saleYear) : new Date().getFullYear()) - Number(args.year)),
};

if (![vehicle.year, vehicle.condition, vehicle.odometer].every(Number.isFinite)) {
  console.error('Provide --year, --condition, and --odometer as numbers.');
  process.exit(1);
}

const features = buildFeatures(vehicle, model);
const price = features.reduce((sum, value, i) => sum + value * model.theta[i], 0);

console.log('Input');
console.log(`  year           ${vehicle.year}`);
console.log(`  make           ${vehicle.make}`);
console.log(`  body           ${vehicle.body}`);
console.log(`  transmission   ${vehicle.transmission}`);
console.log(`  condition      ${vehicle.condition}`);
console.log(`  odometer       ${vehicle.odometer.toLocaleString()} mi`);
console.log(`  vehicle age    ${vehicle.vehicleAge} yr`);
console.log('');
console.log(`Predicted selling price: $${Math.max(0, Math.round(price)).toLocaleString()}`);

if (model.metrics?.test) {
  const { mae, rmse, r2 } = model.metrics.test;
  console.log(`(model test MAE $${mae.toFixed(0)}, RMSE $${rmse.toFixed(0)}, R² ${r2.toFixed(3)})`);
}

function buildFeatures(vehicle, model) {
  const { stats, topMakeList, topBodyList } = model;
  return [
    1,
    z(vehicle.year, stats.year),
    z(vehicle.condition, stats.condition),
    z(vehicle.odometer, stats.odometer),
    z(vehicle.vehicleAge, stats.vehicle_age),
    ...topMakeList.map((m) => (vehicle.make === m ? 1 : 0)),
    ...topBodyList.map((b) => (vehicle.body === b ? 1 : 0)),
    vehicle.transmission === 'automatic' ? 1 : 0,
  ];
}

function z(value, { mean, std }) {
  return (value - mean) / std;
}

function parseArgs(argv) {
  const defaults = {
    model: 'models/model.json',
    year: '2014',
    make: 'kia',
    body: 'suv',
    transmission: 'automatic',
    condition: '35',
    odometer: '45000',
    saleYear: null,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (!(key in defaults) && key !== 'help') {
        console.error(`Unknown option: ${arg}`);
        process.exit(1);
      }
      if (key === 'help') {
        printHelp();
        process.exit(0);
      }
      defaults[key] = argv[++i];
    }
  }
  return defaults;
}

function printHelp() {
  console.log(`Usage:
  node src/predict.mjs [options]

Options:
  --model path           Trained model JSON (default models/model.json)
  --year 2014
  --make kia
  --body suv
  --transmission automatic
  --condition 35
  --odometer 45000
  --saleYear 2015`);
}
