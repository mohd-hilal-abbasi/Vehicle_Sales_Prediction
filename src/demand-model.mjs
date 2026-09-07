import { metrics, predict, trainLinearRegression } from './regression.mjs';
import { classFromDays, DEMAND_CLASSES } from './demand-labels.mjs';

/**
 * Softmax multiclass logistic regression (one weight vector per class).
 * Trained with mini-batch SGD — enough for this tabular baseline.
 */
export function trainSoftmaxClassifier(X, yLabels, { epochs = 40, lr = 0.15, batchSize = 256, seed = 7 } = {}) {
  const classes = DEMAND_CLASSES;
  const classIndex = Object.fromEntries(classes.map((c, i) => [c, i]));
  const n = X.length;
  const m = X[0].length;
  const k = classes.length;

  const W = Array.from({ length: k }, () => Array(m).fill(0));
  const rng = mulberry32(seed);
  const order = Array.from({ length: n }, (_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffle(order, rng);
    for (let start = 0; start < n; start += batchSize) {
      const end = Math.min(start + batchSize, n);
      const grad = Array.from({ length: k }, () => Array(m).fill(0));

      for (let t = start; t < end; t++) {
        const i = order[t];
        const xi = X[i];
        const probs = softmax(scores(xi, W));
        const yi = classIndex[yLabels[i]];

        for (let c = 0; c < k; c++) {
          const err = probs[c] - (c === yi ? 1 : 0);
          for (let j = 0; j < m; j++) grad[c][j] += err * xi[j];
        }
      }

      const scale = lr / (end - start);
      for (let c = 0; c < k; c++) {
        for (let j = 0; j < m; j++) W[c][j] -= scale * grad[c][j];
      }
    }
  }

  return { W, classes };
}

export function predictSoftmax(X, model) {
  return X.map((xi) => {
    const probs = softmax(scores(xi, model.W));
    let best = 0;
    for (let c = 1; c < probs.length; c++) {
      if (probs[c] > probs[best]) best = c;
    }
    return {
      demandClass: model.classes[best],
      probabilities: Object.fromEntries(model.classes.map((name, i) => [name, probs[i]])),
    };
  });
}

export function classificationAccuracy(yTrue, yPred) {
  let correct = 0;
  const confusion = Object.fromEntries(
    DEMAND_CLASSES.map((a) => [a, Object.fromEntries(DEMAND_CLASSES.map((b) => [b, 0]))]),
  );

  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct += 1;
    confusion[yTrue[i]][yPred[i]] += 1;
  }

  return { accuracy: correct / yTrue.length, confusion };
}

export function trainDemandModels(X, yDays, yClass) {
  const daysTheta = trainLinearRegression(X, yDays);
  const classifier = trainSoftmaxClassifier(X, yClass);
  return { daysTheta, classifier };
}

export function evaluateDemand(X, yDays, yClass, { daysTheta, classifier }) {
  const daysPred = predict(X, daysTheta).map((d) => Math.max(1, Math.round(d)));
  const classFromReg = daysPred.map(classFromDays);
  const soft = predictSoftmax(X, classifier);
  const classPred = soft.map((s) => s.demandClass);

  return {
    days: metrics(yDays, daysPred),
    classFromDaysAccuracy: classificationAccuracy(yClass, classFromReg),
    classifierAccuracy: classificationAccuracy(yClass, classPred),
  };
}

function scores(xi, W) {
  return W.map((wc) => {
    let s = 0;
    for (let j = 0; j < xi.length; j++) s += wc[j] * xi[j];
    return s;
  });
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
