import numpy as np
import pytest
from Anomalies.classes.Drift import Drift

def test_drift_anomaly_positive_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 2)
    assert np.array_equal(result, np.array([2, 4, 5, 5, 5]))

def test_drift_anomaly_negative_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, -1)
    assert np.array_equal(result, np.array([0, 0, 0, 0, 0]))

def test_drift_anomaly_with_start_and_end_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 2, 1, 4)
    assert np.array_equal(result, np.array([1, 4, 5, 5, 5]))

def test_drift_anomaly_with_zero_drift():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    result = drift.apply_anomaly(data, 0, 5, 0)
    assert np.array_equal(result, data)

def test_drift_anomaly_with_negative_start_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drift.apply_anomaly(data, 0, 5, 2, -1, 4)

def test_drift_anomaly_with_start_index_greater_than_data_length():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drift.apply_anomaly(data, 0, 5, 2, 6, 4)

def test_drift_anomaly_with_start_index_greater_than_end_index():
    drift = Drift()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drift.apply_anomaly(data, 0, 5, 2, 3, 2)