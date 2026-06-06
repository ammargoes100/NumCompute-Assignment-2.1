"""
io.py 
Descriptions: 
- CSV reading and dtype handling for NumCompute.
- Load CSV into NumPy arrays (float64, missing values become NaN).
- Supports chunked/streaming reading for large files.
"""

import numpy as np
import os

# Common missing value markers
MISSING_VALUE_MARKERS = ["", "na", "n/a", "null", "none", "nan", "?", "missing"]
DEFAULT_FILL_VALUE = np.nan


def load_csv(file_path, delimiter=",", fill_value=DEFAULT_FILL_VALUE, skip_header=True):
    """
    Load entire CSV into a 2D float64 array. Non‑numeric becomes NaN.

    Parameters
    ----------
    file_path : str
    delimiter : str, default ","
    fill_value : float, default np.nan
    skip_header : bool, default True

    Returns
    -------
    data : np.ndarray, shape (n_rows, n_cols), dtype float64
    column_names : list of str or None
    """
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")
    if not file_path.strip():
        raise ValueError("file_path cannot be empty")
    
    if os.path.getsize(file_path) == 0:
        return np.empty((0, 0)), None

    skip = 1 if skip_header else 0
    data = np.genfromtxt(
        file_path,
        delimiter=delimiter,
        skip_header=skip,
        filling_values=fill_value,
        dtype=np.float64,
        missing_values=MISSING_VALUE_MARKERS,
        autostrip=True,
        loose=True,
        invalid_raise=False
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    column_names = None
    if skip_header:
        with open(file_path) as f:
            column_names = f.readline().strip().split(delimiter)

    return data, column_names


def load_csv_fill_missing(file_path, delimiter=",", fill_value=0.0, skip_header=True):
    """Same as load_csv but fills missing with given value (default 0.0)."""
    return load_csv(file_path, delimiter, fill_value, skip_header)


def load_csv_skip_missing_rows(file_path, delimiter=",", skip_header=True):
    """Load CSV and drop any row with a missing value."""
    data, cols = load_csv(file_path, delimiter, np.nan, skip_header)
    clean = ~np.isnan(data).any(axis=1)
    data = data[clean]
    if data.size == 0:
        raise ValueError("No valid rows after removing missing values.")
    return data, cols


def load_csv_chunked(file_path, chunk_size=100, delimiter=",",
                     fill_value=DEFAULT_FILL_VALUE, skip_header=True):
    """
    Stream CSV in chunks. Yields one 2D float64 array per chunk.
    Does NOT return column names – read separately if needed.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    with open(file_path) as f:
        # skip header line if requested
        if skip_header:
            f.readline()

        chunk_lines = []
        for line in f:
            line = line.rstrip('\n')
            chunk_lines.append(line)

            if len(chunk_lines) == chunk_size:
                arr = np.genfromtxt(
                    chunk_lines,
                    delimiter=delimiter,
                    filling_values=fill_value,
                    dtype=np.float64,
                    missing_values=MISSING_VALUE_MARKERS,
                    autostrip=True,
                    loose=True,
                    invalid_raise=False
                )
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                yield arr
                chunk_lines = []

        # leftover rows
        if chunk_lines:
            arr = np.genfromtxt(
                chunk_lines,
                delimiter=delimiter,
                filling_values=fill_value,
                dtype=np.float64,
                missing_values=MISSING_VALUE_MARKERS,
                autostrip=True,
                loose=True,
                invalid_raise=False
            )
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            yield arr