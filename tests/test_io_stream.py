"""
Tests for retained CSV input utilities.
"""

import numpy as np
import pytest

from numcompute_stream.io import (
    load_csv,
    load_csv_fill_missing,
    load_csv_skip_missing_rows,
    load_csv_chunked,
)


@pytest.fixture
def numeric_csv(tmp_path):
    csv_file = tmp_path / "numeric.csv"
    csv_file.write_text(
        "age,score,grade\n"
        "25,90.0,1\n"
        "30,85.5,2\n"
        "35,78.0,3\n"
    )
    return str(csv_file)


@pytest.fixture
def missing_csv(tmp_path):
    csv_file = tmp_path / "missing.csv"
    csv_file.write_text(
        "age,score,grade\n"
        "25,90.0,1\n"
        "30,,2\n"
        "35,78.0,\n"
        "40,88.0,3\n"
    )
    return str(csv_file)


@pytest.fixture
def all_missing_csv(tmp_path):
    csv_file = tmp_path / "all_missing.csv"
    csv_file.write_text(
        "age,score\n"
        "25,\n"
        ",85\n"
        "35,\n"
    )
    return str(csv_file)


@pytest.fixture
def large_csv(tmp_path):
    csv_file = tmp_path / "large.csv"
    rows = ["age,score"]

    for i in range(250):
        rows.append(f"{20 + i},{50.0 + i}")

    csv_file.write_text("\n".join(rows) + "\n")
    return str(csv_file)


@pytest.fixture
def empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    return str(csv_file)


def test_load_csv_returns_numpy_array(numeric_csv):
    data, _ = load_csv(numeric_csv)

    assert isinstance(data, np.ndarray)


def test_load_csv_returns_2d_array(numeric_csv):
    data, _ = load_csv(numeric_csv)

    assert data.ndim == 2


def test_load_csv_dtype_is_float64(numeric_csv):
    data, _ = load_csv(numeric_csv)

    assert data.dtype == np.float64


def test_load_csv_returns_column_names(numeric_csv):
    data, column_names = load_csv(numeric_csv)

    assert column_names == ["age", "score", "grade"]
    assert len(column_names) == data.shape[1]


def test_load_csv_missing_values_become_nan(missing_csv):
    data, _ = load_csv(missing_csv)

    assert np.isnan(data).any()


def test_load_csv_fill_missing_removes_nans(missing_csv):
    data, _ = load_csv_fill_missing(missing_csv, fill_value=0.0)

    assert not np.isnan(data).any()


def test_load_csv_skip_missing_rows_removes_nan_rows(missing_csv):
    data, _ = load_csv_skip_missing_rows(missing_csv)

    assert not np.isnan(data).any()
    assert data.shape[0] == 2


def test_load_csv_skip_missing_rows_raises_when_no_valid_rows(all_missing_csv):
    with pytest.raises(ValueError):
        load_csv_skip_missing_rows(all_missing_csv)


def test_load_csv_empty_file_returns_empty_array(empty_csv):
    data, columns = load_csv(empty_csv, skip_header=False)

    assert data.shape == (0, 0)
    assert columns is None


def test_load_csv_chunked_total_rows_match_full_file(large_csv):
    full_data, _ = load_csv(large_csv)

    total_rows = sum(
        chunk.shape[0]
        for chunk in load_csv_chunked(large_csv, chunk_size=50)
    )

    assert total_rows == full_data.shape[0]


def test_load_csv_chunked_reconstructs_missing_file(missing_csv):
    full_data, _ = load_csv(missing_csv)

    chunks = list(load_csv_chunked(missing_csv, chunk_size=2))
    reconstructed = np.vstack(chunks)

    assert reconstructed.shape == full_data.shape
    assert np.allclose(reconstructed, full_data, equal_nan=True)