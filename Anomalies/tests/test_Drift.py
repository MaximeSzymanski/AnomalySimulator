import numpy as np
import pytest
from Anomalies.classes.Drift import Drift

def test_positive_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 2)
    expected = np.array([2, 4, 5, 5, 5])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_negative_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, -1)
    expected = np.array([0, 0, 0, 0, 0])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_drift_with_start_and_end_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 2, 1, 4)
    expected = np.array([1, 4, 5, 5, 5])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_zero_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 0)
    assert np.array_equal(result, data), f"Expected {data}, but got {result}"

def test_negative_start_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError, match="Start index must be greater than or equal to 0"):
        drift.apply_anomaly(data, 0, 5, 2, -1, 4)

def test_start_index_greater_than_data_length():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError, match="Start index must be less than the length of the data"):
        drift.apply_anomaly(data, 0, 5, 2, 6, 4)

def test_start_index_greater_than_end_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError, match="Start index must be less than end index"):
        drift.apply_anomaly(data, 0, 5, 2, 3, 2)