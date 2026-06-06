"""
Unit tests for numcompute/io.py
Independent test suite - contains all necessary fixtures.
"""

import pytest
import numpy as np
import os
from numcompute.io import (
    load_csv,
    load_csv_fill_missing,
    load_csv_skip_missing_rows,
    load_csv_chunked,
)


# Fixtures

@pytest.fixture
def numeric_csv(tmp_path):
    """A clean numeric CSV with no missing values."""
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
    """A numeric CSV with some missing values."""
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
    """A CSV where every row has at least one missing value."""
    csv_file = tmp_path / "all_missing.csv"
    csv_file.write_text(
        "age,score\n"
        "25,\n"
        ",85\n"
        "35,\n"
    )
    return str(csv_file)

@pytest.fixture
def tab_csv(tmp_path):
    """A tab-separated CSV."""
    csv_file = tmp_path / "tab.tsv"
    csv_file.write_text(
        "age\tscore\tgrade\n"
        "25\t90.0\t1\n"
        "30\t85.5\t2\n"
        "35\t78.0\t3\n"
    )
    return str(csv_file)

@pytest.fixture
def single_row_csv(tmp_path):
    """A CSV with only one data row."""
    csv_file = tmp_path / "single_row.csv"
    csv_file.write_text(
        "age,score\n"
        "25,90.0\n"
    )
    return str(csv_file)

@pytest.fixture
def single_col_csv(tmp_path):
    """A CSV with only one column."""
    csv_file = tmp_path / "single_col.csv"
    csv_file.write_text(
        "age\n"
        "25\n"
        "30\n"
        "35\n"
    )
    return str(csv_file)

@pytest.fixture
def no_header_csv(tmp_path):
    """A CSV with no header row."""
    csv_file = tmp_path / "no_header.csv"
    csv_file.write_text(
        "25,90.0,1\n"
        "30,85.5,2\n"
        "35,78.0,3\n"
    )
    return str(csv_file)

@pytest.fixture
def large_csv(tmp_path):
    """A CSV with many rows for chunking tests."""
    csv_file = tmp_path / "large.csv"
    rows = ["age,score"]
    for i in range(250):
        rows.append(f"{20 + i},{50.0 + i}")
    csv_file.write_text("\n".join(rows) + "\n")
    return str(csv_file)

@pytest.fixture
def empty_csv(tmp_path):
    """An empty CSV file (no rows, no header)."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    return str(csv_file)


# Tests for load_csv()

class TestLoadCsv:
    def test_returns_numpy_array(self, numeric_csv):
        data, _ = load_csv(numeric_csv)
        assert isinstance(data, np.ndarray)

    def test_always_returns_2d(self, numeric_csv):
        data, _ = load_csv(numeric_csv)
        assert data.ndim == 2

    def test_single_row_is_2d(self, single_row_csv):
        data, _ = load_csv(single_row_csv)
        assert data.ndim == 2

    def test_dtype_is_float64(self, numeric_csv):
        data, _ = load_csv(numeric_csv)
        assert data.dtype == np.float64

    def test_column_names_is_list(self, numeric_csv):
        _, column_names = load_csv(numeric_csv)
        assert isinstance(column_names, list)
        assert all(isinstance(name, str) for name in column_names)

    def test_column_count_matches_header(self, numeric_csv):
        data, column_names = load_csv(numeric_csv)
        assert len(column_names) == data.shape[1]

    def test_no_header_returns_none(self, no_header_csv):
        _, column_names = load_csv(no_header_csv, skip_header=False)
        assert column_names is None

    def test_missing_values_become_nan(self, missing_csv):
        data, _ = load_csv(missing_csv)
        assert np.isnan(data).any()

    def test_custom_fill_value_replaces_missing(self, missing_csv):
        fill_value = -999.0
        data, _ = load_csv(missing_csv, fill_value=fill_value)
        assert (data == fill_value).any()
        assert not np.isnan(data).any()

    def test_invalid_file_path_type_raises_type_error(self):
        with pytest.raises(TypeError):
            load_csv(123)

    def test_nonexistent_file_raises_error(self):
        with pytest.raises(Exception):
            load_csv("non_existent_file.csv")

    def test_empty_file_returns_empty_array(self, empty_csv):
        data, _ = load_csv(empty_csv, skip_header=False)
        # genfromtxt on empty file returns array with 0 rows and 0 columns
        assert data.shape == (0, 0) or data.size == 0



# Tests for load_csv_fill_missing()

class TestLoadCsvFillMissing:
    def test_no_nans_in_result(self, missing_csv):
        data, _ = load_csv_fill_missing(missing_csv)
        assert not np.isnan(data).any()

    def test_custom_fill_value_used(self, missing_csv):
        fill_value = -1.0
        data, _ = load_csv_fill_missing(missing_csv, fill_value=fill_value)
        assert (data == fill_value).any()


# Tests for load_csv_skip_missing_rows()

class TestLoadCsvSkipMissingRows:
    def test_no_nans_in_result(self, missing_csv):
        data, _ = load_csv_skip_missing_rows(missing_csv)
        assert not np.isnan(data).any()

    def test_all_missing_raises_value_error(self, all_missing_csv):
        with pytest.raises(ValueError):
            load_csv_skip_missing_rows(all_missing_csv)


# Tests for load_csv_chunked()

class TestLoadCsvChunked:
    def test_total_rows_across_chunks_matches_file(self, large_csv):
        """Check that the sum of chunk rows equals total rows."""
        data_full, _ = load_csv(large_csv)
        total_rows_in_chunks = sum(
            chunk.shape[0]
            for chunk in load_csv_chunked(large_csv, chunk_size=50)
        )
        assert total_rows_in_chunks == data_full.shape[0]

    def test_chunked_with_missing_values(self, missing_csv):
        """Test chunked reading of a file containing missing values."""
        # Load full file for reference
        data_full, _ = load_csv(missing_csv, fill_value=np.nan)
        # Collect all chunks
        chunks = list(load_csv_chunked(missing_csv, chunk_size=2, fill_value=np.nan))
        # Reconstruct
        reconstructed = np.vstack(chunks)
        # Compare shapes and values (NaNs will compare as False, so use allclose with equal_nan=True)
        assert reconstructed.shape == data_full.shape
        assert np.allclose(reconstructed, data_full, equal_nan=True)

    def test_chunked_tab_delimiter(self, tab_csv):
        """Test chunked reading with tab delimiter."""
        data_full, _ = load_csv(tab_csv, delimiter="\t")
        chunks = list(load_csv_chunked(tab_csv, chunk_size=2, delimiter="\t"))
        reconstructed = np.vstack(chunks)
        assert np.allclose(reconstructed, data_full, equal_nan=True)

    def test_chunked_empty_file(self, empty_csv):
        """Chunked reading of an empty file should yield nothing."""
        chunks = list(load_csv_chunked(empty_csv, chunk_size=10, skip_header=False))
        assert len(chunks) == 0