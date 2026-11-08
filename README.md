# Software Defect Prediction - PROMISE Dataset

> BITS Pilani · Machine Learning · Assignment 2 - an end-to-end classification project with five models and a Streamlit app ("Defect Radar").

## 1. Problem Statement

Software defect prediction using the PROMISE dataset.

Given the static code metrics (McCabe and Halstead) of a software module, predict whether the module is
**defective (1)** or **non-defective (0)**. Early identification of defect-prone modules lets teams focus code
reviews and testing where they matter most. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian
Naive Bayes, SVM) are trained on the same preprocessed data and compared using Accuracy, AUC, Precision, Recall,
F1 and the Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

| Item | Details |
|---|---|
| **Source** | PROMISE repository (NASA Metrics Data Program) - Kaggle collection [`fracpete/promisedata`](https://www.kaggle.com/datasets/fracpete/promisedata). The Kaggle download requires an API token, so `src/download_data.py` fetches the same PROMISE ARFF files **JM1, KC1, KC2 and PC1** from the public OpenML mirror and merges them. These four projects share an identical set of 21 metrics. |
| **Number of records** | 14,625 raw rows (JM1 10,885 · KC1 2,109 · KC2 522 · PC1 1,109) → **11,193** after removing 3,250 duplicates and 182 invalid records |
| **Number of features** | **21** numeric static code metrics: `loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t, lOCode, lOComment, lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount` |
| **Target column** | `defects` - converted to binary: `0` = Non Defective, `1` = Defective (the source files use `true/false` and `yes/no`) |
| **Class distribution (raw)** | Non Defective 12,009 (82.1 %) · Defective 2,616 (17.9 %) |
| **Class distribution (cleaned)** | Non Defective 8,755 (78.2 %) · Defective 2,438 (21.8 %) |
| **Train / test split** | Stratified 80 / 20 → 8,954 train rows and 2,239 test rows (1,751 non-defective / 488 defective) |

### Preprocessing pipeline (`src/preprocessing.py`)

1. **Load** the merged CSV (`data/promise_defects.csv`).
2. **Missing values** - 25 missing cells (JM1) are imputed with the *training-set median* (fitted after the split, so there is no leakage).
3. **Duplicates** - 3,250 exact duplicate rows removed *before* the split (otherwise identical modules could land in both train and test).
4. **Invalid records** - 182 rows violating metric consistency rules are dropped: `loc < 1`, `v(g) > loc`, `lOCode > loc`, Halstead length `n ≠ total_Op + total_Opnd`, Halstead level `l > 1`, negative values, `ev(g)/iv(g) > v(g)`, unique tokens above total tokens.
5. **Class imbalance** - random over-sampling of the minority class on the **training partition only** (8,954 → 14,008 rows); the test set keeps its natural distribution.
6. **Features / target separation** and **stratified** `train_test_split` (`test_size=0.2`).
7. **Transform and scale** - all metrics are heavily right-skewed, so `log1p` is applied first, then `StandardScaler` (fitted on the training data only) and applied identically to every model.
8. **Reproducibility** - fixed seed `42` for the split, resampling, cross-validation and all stochastic estimators.

## 3. GitHub Repository Link

https://github.com/2025NS04011-Muthu/Muthu_MachineLearning_Assignment-2.git

## 4. Models Used

All five models are trained on the same scaled, balanced training set and evaluated on the same 2,239-row hold-out set (`test_data.csv`).

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6717 | **0.7084** | 0.3562 | 0.6270 | **0.4543** | **0.2642** |
| Decision Tree | 0.6632 | 0.6813 | 0.3432 | 0.5963 | 0.4356 | 0.2368 |
| KNN | 0.6396 | 0.6889 | 0.3323 | **0.6475** | 0.4392 | 0.2379 |
| Naive Bayes | **0.7021** | 0.6713 | **0.3638** | 0.4898 | 0.4175 | 0.2276 |
| SVM | 0.6695 | 0.6975 | 0.3489 | 0.5963 | 0.4402 | 0.2444 |

*Bold = best value in the column. Hyper-parameters were chosen by 3-fold stratified cross-validation (ROC-AUC) on the **training** data only (over-sampling is done inside each fold; the hold-out set is never used): Logistic Regression `C=10`; Decision Tree `max_depth=4, min_samples_leaf=20`; KNN `n_neighbors=101, weights="uniform"`; GaussianNB `var_smoothing=0.1`; SVM `RBF, C=1, gamma=0.01, probability=True`. Re-run the search with `python -m src.train_models --tune`.*

## 5. Model Observations

| Model | Observation |
|---|---|
| **Logistic Regression** | Best overall. After the `log1p` + standardisation the metric/defect relationship is close to linear, so a linear boundary generalises best: highest AUC (0.708), F1 (0.454) and MCC (0.264), with recall 0.63. It is also the cheapest and most interpretable model (largest weights: Halstead volume `v`, length `n`, total operands). |
| **Decision Tree** | Catches 60 % of defects (recall 0.60) but with low precision (0.34) and AUC 0.68. A shallow tree (depth 4) is needed - deeper trees overfit and lose AUC. `loc` dominates (Gini importance ≈ 0.83), followed by Halstead difficulty `d`, intelligence `i` and effort `e`. |
| **KNN** | Highest recall (0.65) but low precision (0.33), so many clean modules are flagged. It only becomes competitive with a large neighbourhood (k = 101): with k = 7 its AUC was 0.64, because small neighbourhoods are very noisy on this data. |
| **Naive Bayes** | Highest accuracy (0.70) and precision (0.36) but the lowest recall (0.49), AUC (0.67) and MCC (0.23): the Gaussian, feature-independence assumption does not hold for the strongly correlated Halstead metrics. |
| **SVM** | Second best (AUC 0.698, MCC 0.244, F1 0.440). The RBF kernel adds little over the linear boundary for this data, while training is by far the slowest (probability calibration) and the model file is large. |

> Absolute scores are modest for every model (AUC ≈ 0.67-0.71, which is typical for the de-duplicated NASA/PROMISE
> data). Static code metrics only weakly separate defective modules, and removing the ~22 % duplicate rows takes away
> the artificial optimism of leaked copies. Accuracy alone is misleading (a "never defective" model would score 78 %),
> which is why MCC, F1 and AUC are used for the ranking.

## 6. Overall Best Model

**Winner: Logistic Regression**

| Criterion | Rank 1 | Rank 2 |
|---|---|---|
| MCC | **Logistic Regression (0.2642)** | SVM (0.2444) |
| F1 | **Logistic Regression (0.4543)** | SVM (0.4402) |
| AUC | **Logistic Regression (0.7084)** | SVM (0.6975) |

Logistic Regression ranks first on all three criteria. **Why it performed best:** after the `log1p` transform and
standardisation, the relationship between size/complexity metrics and defect-proneness is largely monotonic and close
to linear, which a regularised linear model captures without overfitting. Higher-capacity models (Decision Tree,
KNN) are more sensitive to noisy labels and to the over-sampled minority rows, Naive Bayes violates its independence
assumption on correlated metrics, and the RBF-SVM reaches slightly lower quality at a much higher training and
storage cost. Logistic Regression is also the simplest, fastest and most explainable option.

## 7. Run Instructions

```bash
pip install -r requirements.txt

streamlit run app.py
```

Optional - rebuild everything from scratch (the repository already ships the dataset, trained models and `test_data.csv`):

```bash
python -m src.download_data      # download + merge the PROMISE projects -> data/promise_defects.csv
python -m src.train_models       # preprocess, train 5 models, save model/*.pkl, metrics.csv, test_data.csv
python -m src.train_models --tune  # same, but re-run the cross-validated hyper-parameter search first
python -m src.generate_test_data # regenerate only test_data.csv (deterministic)
```

`notebooks/training.ipynb` walks through the same pipeline step by step with plots.

### Project structure

```
├── app.py                  # Streamlit application ("Defect Radar")
├── requirements.txt
├── README.md
├── test_data.csv           # stratified hold-out set (2,239 rows, 21 metrics + defects)
├── .streamlit/config.toml  # dark theme
├── data/promise_defects.csv
├── model/                  # logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, svm.pkl,
│                           # scaler.pkl, imputer.pkl, encoders.pkl, metrics.csv, metadata.json
├── notebooks/training.ipynb
└── src/
    ├── preprocessing.py    # load, clean, split, impute, scale, balance, schema validation
    ├── train_models.py     # build/train/save models and artifacts
    ├── evaluate.py         # metrics, confusion matrix, report, ROC, ranking
    ├── predict.py          # artifact loading (auto-retrain fallback) and inference
    ├── visuals.py          # seaborn / plotly charts used by the app
    ├── download_data.py    # dataset download + merge
    ├── generate_test_data.py
    └── utils.py            # paths, constants, logging, seeding, JSON helpers
```

### Using the app

1. **Upload & Validate** - upload `test_data.csv` (or click *Load bundled test_data.csv*). The 21 metric columns are mandatory; `defects` is optional (without it only predictions are produced).
2. **Evaluate Model** - choose one of the five models: prediction counts, the six metrics, seaborn confusion-matrix heatmap, ROC curve, classification report, prediction table and a CSV download.
3. **Compare Models** - leaderboard (ranked by MCC → F1 → AUC), grouped bar chart, radar chart and ROC overlay for all five models.
4. **Insights** - Decision Tree feature importance, Logistic Regression weights and the data-cleaning log.

## 8. Streamlit Deployment Instructions

1. Push the project (including `model/`, `data/` and `test_data.csv`) to a public GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select the repository and branch, and set **Main file path** to `app.py`.
5. (Optional) In *Advanced settings* choose Python 3.11 or newer.
6. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically; no code changes are required.

If the pickled models were created with a different scikit-learn version than the one installed on the server, the app
detects this on start-up and transparently retrains the five models from the bundled `data/promise_defects.csv`
(about 30 seconds), so deployment never fails because of version drift.

## 9. Screenshots Section

**Local execution**

<!-- Add a screenshot of the app running locally (e.g. http://localhost:8501) -->
`<LOCAL_EXECUTION_SCREENSHOT>`

**Streamlit Cloud deployment**

<!-- Add a screenshot of the deployed app (https://<your-app>.streamlit.app) -->
`<STREAMLIT_CLOUD_SCREENSHOT>`