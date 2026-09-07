# Vehicle Sales Prediction

Simple Node.js (24+) machine learning project for used-vehicle **selling price** and **demand / days-to-sale** prediction from auction-style CSV data.

Uses plain ES modules (`.mjs`), no ML libraries.

## Requirements

- Node.js **24+**

## Quick start

### Price prediction

```bash
npm run train
npm run predict
```

### Demand prediction (inventory planning)

```bash
npm run train:demand
npm run predict:demand
```

Example:

```bash
node src/predict-demand.mjs --make toyota --model-name camry --state ca --year 2020 --odometer 45000
```

Output shape:

```text
Vehicle: Toyota Camry
State: CA
Year: 2020
Mileage: 45,000

Predicted demand: HIGH
Expected days-to-sale: 14 days
```

## Data

Keep the training CSV as `fake data.csv` in the project root.

Expected columns:

`year,make,model,trim,body,transmission,vin,state,condition,odometer,color,interior,seller,mmr,sellingprice,saledate`

Trainers randomly sample rows (default **25,000**) so runs stay fast on the large file.

## 1. Selling price model

| Feature | Notes |
|--------|--------|
| year, condition, odometer | Standardized numeric |
| vehicle age | Sale year − model year |
| make / body | Top categories one-hot |
| transmission | Automatic flag |

**Target:** `sellingprice`

```bash
node src/train.mjs --sample 25000
node src/predict.mjs --year 2015 --make kia --body suv --condition 40 --odometer 30000
```

## 2. Demand model ⭐

Classifies inventory velocity for dealer planning:

| Class | Meaning |
|-------|---------|
| **HIGH** | Likely to sell quickly (≤ 21 days) |
| **MEDIUM** | Typical turn (22–40 days) |
| **LOW** | Slow mover (41+ days) |

Also predicts **expected days-to-sale**.

### How labels are built

The CSV has no true days-on-lot field. Training labels are a **proxy** from:

- sale price vs MMR (market strength)
- condition
- mileage intensity (miles / age)
- vehicle age
- model popularity in the sample

Classes are assigned by **score tertiles** (~⅓ HIGH / MEDIUM / LOW) so the classifier stays balanced for inventory planning.

At prediction time the model only needs listing-style inputs: make, model, state, year, mileage, condition — not final sale price.

### Demand features

year, condition, odometer, age, miles/year, top makes / models / states / bodies, transmission

### Demand scripts

```bash
npm run train:demand
node src/predict-demand.mjs --make toyota --model-name camry --state ca --year 2020 --odometer 45000 --condition 40
```

| Command | What it does |
|---------|----------------|
| `npm run train:demand` | Train days + HIGH/MEDIUM/LOW models → `models/demand-model.json` |
| `npm run predict:demand` | Predict demand for a vehicle |
| `npm run demo:demand` | Train then predict the Camry example |

## Project layout

```
package.json
fake data.csv
src/
  csv.mjs              # stream + reservoir sample
  features.mjs         # price features
  regression.mjs       # linear regression + metrics
  train.mjs / predict.mjs
  demand-labels.mjs    # proxy HIGH/MEDIUM/LOW + days
  demand-features.mjs  # demand feature pipeline
  demand-model.mjs     # days regression + softmax classifier
  train-demand.mjs / predict-demand.mjs
models/
  model.json           # price model
  demand-model.json    # demand model
```

## Train options (both trainers)

```text
--data path          CSV path (default: fake data.csv)
--sample N           Rows to sample (default: 25000)
--test 0.2           Hold-out ratio
--out path           Output model JSON
```

## Notes

- Baseline models for learning / demos — not a production pricing or allocation engine.
- Invalid rows are dropped during cleaning.
- Demand classes are proxy labels; replace with real days-on-lot when you have them.
