# Explicit numeric algorithms and tabular experiments

A finished collection of explicit Bernoulli likelihood classification, centered least squares, Lloyd clustering, bounded RGB palette reduction, small Python functions, table selection/summary and portable tabular-model comparisons. Reference/filled source notebooks and assisted changes are described in `ATTRIBUTION.md`; shared material is not claimed as solely authored work. Numeric outputs and verification are included, with raw data and fitted binaries kept local.

## Run

Use Python 3.12 and install `requirements.txt`:

```powershell
python -m pytest -q -p no:cacheprovider
python experiment.py --data-dir local/data --mnist-dir local/mnist --output local/metrics.json
```

The data directory requires `bank.csv` with target `deposit`, `UniversalBank.csv` with target `Personal Loan`, `titanic-train.csv` with actual passenger predictors and `Survived`, UCI `car.data`, Wine Reviews `winemag-data-130k-v2.json`, and `demographics.csv` with City/Age/Gender/Family Income. MNIST uses the standard locally downloaded torchvision directory. There are no hidden Colab/Drive paths. Missing inputs fail explicitly. Source links are in `ATTRIBUTION.md`. The original 418-row Titanic CSV was only a prediction/submission table; this run uses the actual retained 891-row training table instead.

## Fresh measurements

Tabular pipelines deduplicate before a stratified 60/20/20 split, exclude conflicting identical-predictor labels, and fit imputation, encoding and scaling on training rows only. Unknown categories are supported at inference. Validation macro F1 chooses decision-tree depth, random-forest depth and KNN neighbor count independently; test data selects no settings. Post-contact `duration` is excluded from contact-response prediction, customer IDs/ZIP from loan prediction, and passenger names/tickets/IDs/cabin from Titanic prediction.

| Dataset | Decision tree | Random forest | KNN |
|---|---:|---:|---:|
| bank_contact_response | 68.16% | 72.55% | 70.04% |
| bank_personal_loan | 98.50% | 98.90% | 96.19% |
| titanic_survival | 75.33% | 78.00% | 82.67% |
| car_acceptability | 97.69% | 97.11% | 87.57% |
| breast_cell_benchmark | 91.23% | 92.98% | 92.11% |

These are one-seed row-split measurements. Bank-source provenance, entity/time separation and operational validity are not independently certified. UCI Car Evaluation encodes a predefined hierarchical decision model. Breast-cell results are an offline benchmark, not a clinical tool or medical guidance. High loan-table accuracy does not establish a valid financial decision system.

The explicit Bernoulli classifier corrects an empty test array formed by slicing already-truncated training data. Both official splits are thresholded consistently at 127. A balanced 6,000-digit training subset is split into 4,800 training/1,200 validation examples; smoothing is chosen on validation only, followed by 1,000 balanced official test examples. Fresh test accuracy is 85.70%. Every joint log likelihood and prediction matches an independent library Bernoulli model to numerical tolerance. Arbitrary integer/string class labels decode correctly, and long feature vectors stay in log space. This is a subset benchmark, not full MNIST performance or real handwriting validation.

Centered least squares fixes the erroneous extra factors in the retained `sum(x*y)` formula and rejects constant/non-finite inputs. Fresh offline regression uses the bundled diabetes table's BMI predictor because the historical X/R CSVs were unavailable. Validation chooses polynomial degree 1 among degrees 1–5. Test R² is 0.243; the manual line's test predictions agree with the independent library fit. This is a deliberately limited mathematical comparison, not a clinical predictor.

Explicit Lloyd clustering uses deterministic distance-weighted initialization, bounds distance-matrix batches and retains empty centroids to avoid NaNs. Iris supplies a fresh three-cluster comparison. The RGB function validates input, caps image size, samples at most 20,000 pixels for fitting and returns a palette-reduced RGB array. Synthetic pixel fixtures verify implementation behavior; no claim is made about the missing historical picture's perceptual quality. No picture is committed.

The Wine Reviews table contains 129,971 rows; numeric summaries report 19,540 Italian rows and 49 Australian/New Zealand rows scoring at least 95. Location selections and missing-count summaries are checked on actual data. The local 12-row demographic table contributes aggregates only, with valid observation counts reported for missing values. No individual raw record is exported.

## Notebooks and limits

Four notebooks follow setup → implementation/experiment → results and have executed code cells with outputs stripped. Eight checks cover independent Bernoulli likelihood parity, long-vector stability, centered regression, monotone clustering objective, library/centroid agreement, bounded color palettes, Boolean/list edge cases and training-only pipeline fitting with conflict removal. Full fresh data runs cover five tabular datasets, official MNIST subsets, regression, clustering and 129,971-row table operations.

This is a curated finished implementation collection. Original templates and duplicates remain local. Missing Social Network Ads, Drug200, HR/credit samples, historical regression/image data and historical plotting/cleaning-source tables were not reproduced. The documented real datasets and mathematical fixtures support the scope above; metrics are not invented for unavailable inputs. `metrics.json` includes split counts, selected settings, confusion matrices and limitations.
