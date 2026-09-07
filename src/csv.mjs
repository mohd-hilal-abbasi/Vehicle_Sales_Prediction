import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

/**
 * Stream-parse a CSV file and optionally keep a random sample of rows.
 * Assumes a header row and no quoted commas in values (matches this dataset).
 */
export async function loadCsv(filePath, { sampleSize = 20_000, seed = 42 } = {}) {
  const rl = createInterface({
    input: createReadStream(filePath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });

  let headers = null;
  const reservoir = [];
  let seen = 0;
  const rng = mulberry32(seed);

  for await (const line of rl) {
    if (!line.trim()) continue;

    if (!headers) {
      headers = splitCsvLine(line);
      continue;
    }

    const values = splitCsvLine(line);
    if (values.length !== headers.length) continue;

    const row = Object.fromEntries(headers.map((h, i) => [h.trim(), values[i]?.trim() ?? '']));
    seen += 1;

    if (reservoir.length < sampleSize) {
      reservoir.push(row);
    } else {
      const j = Math.floor(rng() * seen);
      if (j < sampleSize) reservoir[j] = row;
    }
  }

  return { headers, rows: reservoir, totalSeen: seen };
}

function splitCsvLine(line) {
  return line.split(',');
}

/** Deterministic PRNG for reproducible sampling. */
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
