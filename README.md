# NumCompute

A modular, high-performance scientific computing toolkit built using Python and NumPy.

This project is designed to replicate core functionalities of ML libraries like scikit-learn, focusing on vectorised computation, numerical stability, and clean software design.

---

## Features

- Data loading and preprocessing (scaling, encoding, imputation)
- Sorting, searching, and top-k operations
- Ranking with tie handling
- Statistical computations (mean, variance, quantiles)
- Evaluation metrics (accuracy, precision, recall, F1, MSE, ROC-AUC)
- Finite-difference gradients (optional Jacobians)
- ML-style pipeline abstraction
- Benchmarking: NumPy vs pure Python performance

---

## Project Structure
```
NumCompute/
├── numcompute/
│   ├── __init__.py
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
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_stats.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_optim.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   ├── test_utils.py
│   ├── test_benchmarking.py
│   └── test_run.py
├── demo/
│   └── quickstart.ipynb
├── dataset/
│   └── test_student_performance_data.csv
├── README.md
└── pyproject.toml
```
---------

## Requirements

- Python 3.10+
- NumPy only (no pandas, sklearn, pytorch)
- pytest for testing

---------

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/saminzk/NumCompute.git
cd NumCompute
```
### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pytest
pip install -e .
```
---------

# How to Run
Run a simple script:
```bash
python test_run.py
```

Run all unit tests
```bash
pytest
```

Run Demo Notebook
```bash
jupyter notebook demo/quickstart.ipynb
```





