import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadCsv } from './csv.mjs';
import { fitFeaturePipeline } from './features.mjs';
import { metrics, predict, trainLinearRegression } from './regression.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const args = parseArgs(process.argv.slice(2));
const dataPath = path.resolve(root, args.data);
const sampleSize = Number(args.sample);
const testRatio = Number(args.test);
const modelPath = path.resolve(root, args.out);

console.log(`Loading sample of ${sampleSize} rows from ${dataPath} ...`);
const { rows, totalSeen } = await loadCsv(dataPath, { sampleSize, seed: 42 });
console.log(`Read ${totalSeen.toLocaleString()} rows; kept ${rows.length.toLocaleString()} for training.`);

const pipeline = fitFeaturePipeline(rows);
const { X, y } = pipeline.transform(rows);

if (X.length < 100) {
  console.error('Not enough valid rows after cleaning. Check the CSV.');
  process.exit(1);
}

const split = Math.floor(X.length * (1 - testRatio));
const Xtrain = X.slice(0, split);
const ytrain = y.slice(0, split);
const Xtest = X.slice(split);
const ytest = y.slice(split);

console.log(`Training on ${Xtrain.length} rows, testing on ${Xtest.length} ...`);
const theta = trainLinearRegression(Xtrain, ytrain);

const trainMetrics = metrics(ytrain, predict(Xtrain, theta));
const testMetrics = metrics(ytest, predict(Xtest, theta));

printMetrics('Train', trainMetrics);
printMetrics('Test ', testMetrics);

const model = {
  createdAt: new Date().toISOString(),
  algorithm: 'multiple-linear-regression',
  sampleSize: rows.length,
  featureNames: pipeline.featureNames,
  topMakeList: pipeline.topMakeList,
  topBodyList: pipeline.topBodyList,
  stats: pipeline.stats,
  theta,
  metrics: { train: trainMetrics, test: testMetrics },
};

await mkdir(path.dirname(modelPath), { recursive: true });
await writeFile(modelPath, JSON.stringify(model, null, 2));
console.log(`\nSaved model → ${modelPath}`);

function printMetrics(label, m) {
  console.log(
    `${label}  MAE $${m.mae.toFixed(0).padStart(6)}  RMSE $${m.rmse.toFixed(0).padStart(6)}  R² ${m.r2.toFixed(3)}`,
  );
}

function parseArgs(argv) {
  const defaults = {
    data: 'fake data.csv',
    sample: '25000',
    test: '0.2',
    out: 'models/model.json',
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--data') defaults.data = argv[++i];
    else if (arg === '--sample') defaults.sample = argv[++i];
    else if (arg === '--test') defaults.test = argv[++i];
    else if (arg === '--out') defaults.out = argv[++i];
    else if (arg === '--help' || arg === '-h') {
      console.log(`Usage: node src/train.mjs [--data path] [--sample N] [--test 0.2] [--out models/model.json]`);
      process.exit(0);
    }
  }
  return defaults;
}
