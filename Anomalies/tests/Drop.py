import numpy as np
import pytest
from Anomalies.classes.Drop import Drop

def test_drop_anomaly_within_bounds():
    drop = Drop()
    data = np.array([1, 2, 3, 4, 5])
    result = drop.apply_anomaly(data, 2)
    assert np.array_equal(result, np.array([2, 2, 2, 2, 2]))

def test_drop_anomaly_with_start_and_end_index():
    drop = Drop()
    data = np.array([1, 2, 3, 4, 5])
    result = drop.apply_anomaly(data, 2, 1, 4)
    assert np.array_equal(result, np.array([1, 2, 2, 2, 5]))

def test_drop_anomaly_with_negative_start_index():
    drop = Drop()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drop.apply_anomaly(data, 2, -1, 4)

def test_drop_anomaly_with_start_index_greater_than_data_length():
    drop = Drop()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drop.apply_anomaly(data, 2, 6, 4)

def test_drop_anomaly_with_start_index_greater_than_end_index():
    drop = Drop()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        drop.apply_anomaly(data, 2, 3, 2)