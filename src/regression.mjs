/**
 * Multiple linear regression via normal equations:
 *   theta = (XᵀX)⁻¹ Xᵀy
 */

export function trainLinearRegression(X, y) {
  const n = X.length;
  const m = X[0].length;

  const xtx = zeros(m, m);
  const xty = Array(m).fill(0);

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    const yi = y[i];
    for (let j = 0; j < m; j++) {
      xty[j] += xi[j] * yi;
      for (let k = j; k < m; k++) {
        xtx[j][k] += xi[j] * xi[k];
      }
    }
  }

  // Symmetry
  for (let j = 0; j < m; j++) {
    for (let k = 0; k < j; k++) {
      xtx[j][k] = xtx[k][j];
    }
  }

  // Ridge regularization for numerical stability
  const lambda = 1e-3;
  for (let j = 1; j < m; j++) xtx[j][j] += lambda;

  const theta = solveLinearSystem(xtx, xty);
  return theta;
}

export function predict(X, theta) {
  return X.map((row) => dot(row, theta));
}

export function metrics(yTrue, yPred) {
  const n = yTrue.length;
  let mae = 0;
  let mse = 0;
  let ssRes = 0;
  let ssTot = 0;
  const mean = yTrue.reduce((a, b) => a + b, 0) / n;

  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    mae += Math.abs(err);
    mse += err * err;
    ssRes += err * err;
    ssTot += (yTrue[i] - mean) ** 2;
  }

  return {
    mae: mae / n,
    rmse: Math.sqrt(mse / n),
    r2: 1 - ssRes / (ssTot || 1),
  };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

/** Gaussian elimination with partial pivoting. */
function solveLinearSystem(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];

    const diag = M[col][col];
    if (Math.abs(diag) < 1e-12) {
      throw new Error('Singular matrix — try a larger sample or fewer features');
    }

    for (let j = col; j <= n; j++) M[col][j] /= diag;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col];
      for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
    }
  }

  return M.map((row) => row[n]);
}
