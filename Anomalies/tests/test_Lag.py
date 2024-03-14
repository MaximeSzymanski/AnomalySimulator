import numpy as np
import pytest
from Anomalies.classes.Lag import Lag

def test_lag_anomaly_positive_lag():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    result = lag.apply_anomaly(data, 2, 5, 0)
    assert np.array_equal(result, np.array([0, 0, 1, 2, 3]))

def test_lag_anomaly_negative_lag():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    result = lag.apply_anomaly(data, -2, 5, 0)
    assert np.array_equal(result, np.array([3, 4, 5, 5, 5]))

def test_lag_anomaly_with_start_and_end_index():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    result = lag.apply_anomaly(data, 2, 5, 0, 1, 4)
    assert np.array_equal(result, np.array([1, 0, 0, 2, 5]))

def test_lag_anomaly_with_zero_lag():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        lag.apply_anomaly(data, 0, 5, 0)

def test_lag_anomaly_with_lag_greater_than_data_length():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        lag.apply_anomaly(data, 6, 5, 0)

def test_lag_anomaly_with_negative_start_index():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        lag.apply_anomaly(data, 2, 5, 0, -1, 4)

def test_lag_anomaly_with_start_index_greater_than_data_length():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        lag.apply_anomaly(data, 2, 5, 0, 6, 4)

def test_lag_anomaly_with_start_index_greater_than_end_index():
    lag = Lag()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        lag.apply_anomaly(data, 2, 5, 0, 3, 2)