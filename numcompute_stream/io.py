"""
CSV input utilities for NumCompute-Stream.

This module builds on the CSV loading tools from the original NumCompute
package. The full-file loader and chunked reader are retained because they are
useful for both normal batch workflows and simple streaming simulations.
"""

import os
import numpy as np


MISSING_VALUE_MARKERS = ["", "na", "n/a", "null", "none", "nan", "?", "missing"]
DEFAULT_FILL_VALUE = np.nan

def _validate_file_path(file_path):
    """
    Validate a CSV file path before reading.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")

    if not file_path.strip():
        raise ValueError("file_path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")


def _validate_delimiter(delimiter):
    """
    Validate the delimiter used for CSV parsing.
    """
    if not isinstance(delimiter, str) or delimiter == "":
        raise ValueError("delimiter must be a non-empty string")

def load_csv(file_path, delimiter=",", fill_value=DEFAULT_FILL_VALUE, skip_header=True):
    """
    Load an entire CSV file into a 2D float64 NumPy array.

    Non-numeric values and recognised missing markers are converted to the
    selected fill value.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    delimiter : str, default=","
        Column separator.
    fill_value : float, default=np.nan
        Value used for missing entries.
    skip_header : bool, default=True
        Whether to treat the first row as column names.

    Returns
    -------
    data : np.ndarray of shape (n_rows, n_cols)
        Loaded numeric data.
    column_names : list of str or None
        Header names if skip_header=True, otherwise None.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")

    if not file_path.strip():
        raise ValueError("file_path cannot be empty")

    if os.path.getsize(file_path) == 0:
        return np.empty((0, 0), dtype=np.float64), None

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
        invalid_raise=False,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    column_names = None

    if skip_header:
        with open(file_path, encoding="utf-8") as file:
            column_names = file.readline().strip().split(delimiter)

    return data, column_names


def load_csv_fill_missing(file_path, delimiter=",", fill_value=0.0, skip_header=True):
    """
    Load a CSV file and replace missing values with the chosen fill value.
    """
    return load_csv(
        file_path=file_path,
        delimiter=delimiter,
        fill_value=fill_value,
        skip_header=skip_header,
    )


def load_csv_skip_missing_rows(file_path, delimiter=",", skip_header=True):
    """
    Load a CSV file and remove rows containing missing values.
    """
    data, columns = load_csv(
        file_path=file_path,
        delimiter=delimiter,
        fill_value=np.nan,
        skip_header=skip_header,
    )

    clean_rows = ~np.isnan(data).any(axis=1)
    data = data[clean_rows]

    if data.size == 0:
        raise ValueError("No valid rows after removing missing values.")

    return data, columns


def load_csv_chunked(
    file_path,
    chunk_size=100,
    delimiter=",",
    fill_value=DEFAULT_FILL_VALUE,
    skip_header=True,
):
    """
    Load a CSV file chunk by chunk.

    Each yielded chunk is a 2D float64 NumPy array. This is useful when the
    full file should not be loaded at once or when simulating a stream of data.
    """
    _validate_file_path(file_path)
    _validate_delimiter(delimiter)

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if os.path.getsize(file_path) == 0:
        return

    with open(file_path, encoding="utf-8") as file:
        if skip_header:
            file.readline()

        chunk_lines = []

        for line in file:
            line = line.rstrip("\n")
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
                    invalid_raise=False,
                )

                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                yield arr
                chunk_lines = []

        if chunk_lines:
            arr = np.genfromtxt(
                chunk_lines,
                delimiter=delimiter,
                filling_values=fill_value,
                dtype=np.float64,
                missing_values=MISSING_VALUE_MARKERS,
                autostrip=True,
                loose=True,
                invalid_raise=False,
            )

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            yield arr
def read_csv_header(file_path, delimiter=","):
    """
    Read only the header row from a CSV file.

    This is useful when chunked reading is used but column names are still
    needed separately.
    """
    _validate_file_path(file_path)
    _validate_delimiter(delimiter)
    with open(file_path, encoding="utf-8") as file:
        first_line = file.readline().strip()

    if first_line == "":
        return None

    return first_line.split(delimiter)


def split_features_target(data, target_col=-1):
    """
    Split a numeric array into features X and target y.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_columns)
        Full numeric data.
    target_col : int, default=-1
        Index of the target column.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    if data.shape[1] < 2:
        raise ValueError("data must contain at least one feature column and one target column")

    n_cols = data.shape[1]

    if target_col < 0:
        target_col = n_cols + target_col

    if target_col < 0 or target_col >= n_cols:
        raise IndexError("target_col is out of range")

    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)

    return X, y


def load_csv_xy(file_path, target_col=-1, delimiter=",", fill_value=DEFAULT_FILL_VALUE, skip_header=True):
    """
    Load a full CSV file and split it into X features and y target.
    """
    data, columns = load_csv(
        file_path=file_path,
        delimiter=delimiter,
        fill_value=fill_value,
        skip_header=skip_header,
    )

    X, y = split_features_target(data, target_col=target_col)

    return X, y, columns


def load_csv_xy_chunked(
    file_path,
    target_col=-1,
    chunk_size=100,
    delimiter=",",
    fill_value=DEFAULT_FILL_VALUE,
    skip_header=True,
):
    """
    Load a CSV file in chunks and yield X_chunk, y_chunk pairs.

    This is the most convenient reader for streaming model training.
    """
    for chunk in load_csv_chunked(
        file_path=file_path,
        chunk_size=chunk_size,
        delimiter=delimiter,
        fill_value=fill_value,
        skip_header=skip_header,
    ):
        yield split_features_target(chunk, target_col=target_col)