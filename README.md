# NumCompute-Stream

A modular NumPy-based scientific computing and streaming machine learning toolkit.

This repository extends the original **NumCompute** package into **NumCompute-Stream**, a decision tree and ensemble-based streaming machine learning framework developed for Assignment 2.2.

The project is implemented using plain Python, NumPy, and matplotlib. External machine learning or data processing libraries such as scikit-learn, pandas, PyTorch, and TensorFlow are not used.

---

## Overview

NumCompute-Stream supports chunk-wise machine learning workflows where data arrives gradually over time. The framework includes custom modules for:

- CSV loading and chunked data handling
- Streaming preprocessing
- Streaming statistics
- Streaming metrics
- ML-style pipelines with `partial_fit()`
- Decision tree classification
- Random Forest / Bagging-style ensemble classification
- Stream training and logging
- Benchmarking
- Matplotlib visualisation

The main focus is clean modular design, numerical stability, reusable APIs, and systematic testing.

---

## Features

### Original NumCompute features

- Data loading and preprocessing
- Sorting, searching, and top-k operations
- Ranking with tie handling
- Descriptive statistics and quantiles
- Evaluation metrics such as accuracy, precision, recall, F1, MSE, ROC-AUC
- Finite-difference gradients and Jacobians
- ML-style pipeline abstraction
- Vectorisation benchmarking

### NumCompute-Stream features

- `DecisionTreeClassifier` implemented from scratch using NumPy
- `EnsembleClassifier` using multiple decision trees with bootstrap sampling and majority voting
- `StreamTrainer` for chunk-wise training, scoring, and logging
- Streaming-compatible `partial_fit()` support
- Streaming preprocessing with running updates
- Streaming metrics with `update()`, `result()`, and `reset()`
- Streaming statistics using chunk-based updates
- Built-in visualisation module using matplotlib
- Demo notebook using a real diabetes classification dataset
- Benchmark script comparing tree and ensemble models under streaming conditions

---

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
│
├── numcompute_stream/
│   ├── __init__.py
│   ├── benchmarking.py
│   ├── ensemble.py
│   ├── io.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── rank.py
│   ├── sort_search.py
│   ├── stats.py
│   ├── stream.py
│   ├── tree.py
│   ├── utils.py
│   └── visualise.py
│
├── tests/
│   ├── test_*.py
│   └── test_*_stream.py
│
├── demo/
│   ├── stream_demo.ipynb
│   └── data/
│       └── diabetes.csv
│
├── benchmarks/
│   ├── stream_benchmark.py
│   └── outputs/
│       ├── tree_cumulative_accuracy.png
│       ├── ensemble_cumulative_accuracy.png
│       └── tree_vs_ensemble_accuracy.png
│
├── README.md
└── pyproject.toml
```

---

## Requirements

- Python 3.10+
- NumPy
- matplotlib
- pytest

No external machine learning or data processing libraries are required.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ammargoes100/NumCompute-Assignment-2.1.git
cd NumCompute
```

If you are using a different repository name locally, simply open the project root folder.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy matplotlib pytest
pip install -e .
```

---

## Running Tests

Run all tests:

```bash
pytest
```

Current test status:

```text
533 passed
```

The tests cover both the original NumCompute modules and the new streaming framework modules.

---

## Running the Demo Notebook

Open:

```text
demo/stream_demo.ipynb
```

The demo notebook shows the full streaming workflow:

1. Load `diabetes.csv` using the custom `load_csv()` function.
2. Split the dataset into chunks.
3. Train `DecisionTreeClassifier` incrementally using `partial_fit()`.
4. Train `EnsembleClassifier` incrementally using `partial_fit()`.
5. Log per-chunk and cumulative accuracy with `StreamTrainer`.
6. Visualise streaming accuracy using `visualise.py`.
7. Compare the single tree and ensemble model.
8. Plot predictions against ground truth for the latest chunk.

---

## Running the Streaming Benchmark

Run from the project root:

```bash
python benchmarks/stream_benchmark.py
```

Example benchmark output:

```text
NumCompute-Stream Benchmark
===========================
Dataset: demo/data/diabetes.csv
Dataset shape: (768, 9)
Chunk size: 64
Number of chunks: 12

Decision Tree
-------------
Final cumulative accuracy: 0.8359
Mean chunk accuracy:       0.8359
Total fit time:           2.615610 seconds
Total predict time:       0.000855 seconds
Final memory estimate:    111789 bytes

Ensemble Classifier
-------------------
Final cumulative accuracy: 0.8424
Mean chunk accuracy:       0.8424
Total fit time:           7.000176 seconds
Total predict time:       0.014496 seconds
Final memory estimate:    111617 bytes
```

The benchmark shows that the ensemble model achieved slightly higher cumulative accuracy than the single decision tree, while requiring more training and prediction time because multiple trees are fitted and evaluated.

Benchmark plots are saved in:

```text
benchmarks/outputs/
```

---

## Example Usage

```python
from numcompute_stream.io import load_csv
from numcompute_stream.benchmarking import make_stream_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer

data, columns = load_csv("demo/data/diabetes.csv", skip_header=True)

X = data[:, :-1]
y = data[:, -1].astype(int)

chunks = list(make_stream_chunks(X, y, chunk_size=64))

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=5,
    criterion="gini",
    random_state=42,
)

trainer = StreamTrainer(model)
trainer.fit_stream(chunks)

print(trainer.get_history("cumulative_accuracy"))
```

---

## Main Streaming Components

### `DecisionTreeClassifier`

A depth-limited decision tree classifier implemented from scratch using NumPy.

Supports:

- `fit(X, y)`
- `partial_fit(X_chunk, y_chunk)`
- `predict(X)`
- Gini impurity
- Entropy impurity
- Maximum depth
- Minimum samples split
- Feature subsampling through `max_features`

The streaming implementation stores chunks seen so far and rebuilds the depth-limited tree after each update. This keeps the implementation simple, deterministic, and suitable for the educational streaming setting.

### `EnsembleClassifier`

A Random Forest / Bagging-style ensemble classifier built from multiple `DecisionTreeClassifier` models.

Supports:

- `n_estimators`
- bootstrap sampling
- majority voting
- `fit(X, y)`
- `partial_fit(X_chunk, y_chunk)`
- `predict(X)`

The ensemble design is inspired by common ensemble learning interfaces, but the implementation is written from scratch using NumPy only.

### `StreamTrainer`

A controller class for chunk-wise learning.

Logs:

- chunk index
- number of samples
- chunk score
- cumulative accuracy
- fit time
- prediction time
- approximate memory footprint

### `visualise.py`

Provides reusable matplotlib plotting functions:

```python
plot_metric_over_time(metric_values, title, ylabel)
compare_models(metric1, metric2, labels)
plot_predictions_vs_ground_truth(y_true, y_pred)
```

Each function returns matplotlib `fig, ax` objects and supports saving plots to file.

---

## Design Notes

The project follows a modular architecture:

- `io.py` handles custom CSV loading.
- `preprocessing.py` handles scaling, encoding, and imputation.
- `pipeline.py` chains transformers and models.
- `tree.py` implements the base decision tree learner.
- `ensemble.py` implements tree-based ensembling.
- `stream.py` manages chunk-wise learning and logging.
- `metrics.py` and `stats.py` provide streaming evaluation utilities.
- `visualise.py` provides plotting helpers.
- `benchmarking.py` supports performance comparison.

This design allows single trees and ensemble models to share a consistent API.

---

## Limitations

The current streaming tree and ensemble models accumulate previously seen chunks and rebuild after each update. This approach is simple and reliable for small educational datasets, but it is not as memory-efficient as specialised online tree algorithms such as Hoeffding trees.

Boosting was not implemented because it requires additional sample-weight management and more complex streaming update logic. The implemented ensemble focuses on a Random Forest / Bagging-style approach.

---

## Author

Muhammad Ammar

Assignment 2.2: Programming Task 2  
NumCompute-Stream: A Modularised Ensemble Tree-based Streaming Machine Learning Framework
