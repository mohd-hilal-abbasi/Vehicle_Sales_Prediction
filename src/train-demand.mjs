import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadCsv } from './csv.mjs';
import { fitDemandPipeline } from './demand-features.mjs';
import { evaluateDemand, trainDemandModels } from './demand-model.mjs';
import { DEMAND_CLASSES } from './demand-labels.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const args = parseArgs(process.argv.slice(2));
const dataPath = path.resolve(root, args.data);
const sampleSize = Number(args.sample);
const testRatio = Number(args.test);
const modelPath = path.resolve(root, args.out);

console.log(`Loading sample of ${sampleSize} rows from ${dataPath} ...`);
const { rows, totalSeen } = await loadCsv(dataPath, { sampleSize, seed: 42 });
console.log(`Read ${totalSeen.toLocaleString()} rows; kept ${rows.length.toLocaleString()} for demand training.`);

const pipeline = fitDemandPipeline(rows);
const { X, yDays, yClass } = pipeline.transform(pipeline.labeled, { alreadyLabeled: true });

if (X.length < 100) {
  console.error('Not enough valid rows after cleaning. Check the CSV.');
  process.exit(1);
}

const classCounts = Object.fromEntries(DEMAND_CLASSES.map((c) => [c, 0]));
for (const label of yClass) classCounts[label] += 1;

console.log('Label mix (proxy demand from MMR / condition / mileage / popularity):');
for (const c of DEMAND_CLASSES) {
  const pct = ((classCounts[c] / yClass.length) * 100).toFixed(1);
  console.log(`  ${c.padEnd(6)} ${String(classCounts[c]).padStart(6)}  (${pct}%)`);
}

const split = Math.floor(X.length * (1 - testRatio));
const train = {
  X: X.slice(0, split),
  yDays: yDays.slice(0, split),
  yClass: yClass.slice(0, split),
};
const test = {
  X: X.slice(split),
  yDays: yDays.slice(split),
  yClass: yClass.slice(split),
};

console.log(`\nTraining on ${train.X.length} rows, testing on ${test.X.length} ...`);
const models = trainDemandModels(train.X, train.yDays, train.yClass);

const trainEval = evaluateDemand(train.X, train.yDays, train.yClass, models);
const testEval = evaluateDemand(test.X, test.yDays, test.yClass, models);

printEval('Train', trainEval);
printEval('Test ', testEval);

const model = {
  createdAt: new Date().toISOString(),
  task: 'vehicle-demand-prediction',
  algorithm: {
    daysToSale: 'multiple-linear-regression',
    demandClass: 'softmax-logistic-regression',
  },
  labelNote:
    'Days-to-sale and demand class are proxy labels derived from sale-vs-MMR, condition, mileage intensity, age, and model popularity (dataset has no true days-on-lot).',
  sampleSize: rows.length,
  featureNames: pipeline.featureNames,
  topMakeList: pipeline.topMakeList,
  topModelList: pipeline.topModelList,
  topStateList: pipeline.topStateList,
  topBodyList: pipeline.topBodyList,
  stats: pipeline.stats,
  daysTheta: models.daysTheta,
  classifier: models.classifier,
  classCounts,
  metrics: { train: trainEval, test: testEval },
};

await mkdir(path.dirname(modelPath), { recursive: true });
await writeFile(modelPath, JSON.stringify(model, null, 2));
console.log(`\nSaved demand model → ${modelPath}`);

function printEval(label, ev) {
  console.log(
    `${label}  days MAE ${ev.days.mae.toFixed(1).padStart(5)}d  R² ${ev.days.r2.toFixed(3)}  ` +
      `class acc ${pct(ev.classifierAccuracy.accuracy)}  (from-days ${pct(ev.classFromDaysAccuracy.accuracy)})`,
  );
}

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

function parseArgs(argv) {
  const defaults = {
    data: 'fake data.csv',
    sample: '25000',
    test: '0.2',
    out: 'models/demand-model.json',
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--data') defaults.data = argv[++i];
    else if (arg === '--sample') defaults.sample = argv[++i];
    else if (arg === '--test') defaults.test = argv[++i];
    else if (arg === '--out') defaults.out = argv[++i];
    else if (arg === '--help' || arg === '-h') {
      console.log(
        'Usage: node src/train-demand.mjs [--data path] [--sample N] [--test 0.2] [--out models/demand-model.json]',
      );
      process.exit(0);
    }
  }
  return defaults;
}
