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
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

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