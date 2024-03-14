import numpy as np
import pytest
from Anomalies.classes.Saturation import Saturation

def test_saturation_anomaly_within_bounds():
    saturation = Saturation()
    data = np.array([1, 2, 3, 4, 5])
    result = saturation.apply_anomaly(data, 5)
    assert np.array_equal(result, np.array([5, 5, 5, 5, 5]))

def test_saturation_anomaly_with_start_and_end_index():
    saturation = Saturation()
    data = np.array([1, 2, 3, 4, 5])
    result = saturation.apply_anomaly(data, 5, 1, 4)
    assert np.array_equal(result, np.array([1, 5, 5, 5, 5]))

def test_saturation_anomaly_with_negative_start_index():
    saturation = Saturation()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        saturation.apply_anomaly(data, 5, -1, 4)

def test_saturation_anomaly_with_start_index_greater_than_data_length():
    saturation = Saturation()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        saturation.apply_anomaly(data, 5, 6, 4)

def test_saturation_anomaly_with_start_index_greater_than_end_index():
    saturation = Saturation()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        saturation.apply_anomaly(data, 5, 3, 2)