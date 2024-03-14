import numpy as np
import pytest
from Anomalies.classes.Spike import Spike

def test_spike_anomaly_within_bounds():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    result = spike.apply_anomaly(data, 2, 5, 0)
    assert np.array_equal(result, np.array([3, 4, 5, 5, 5]))

def test_spike_anomaly_exceeding_upper_bound():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    result = spike.apply_anomaly(data, 5, 5, 0)
    assert np.array_equal(result, np.array([5, 5, 5, 5, 5]))

def test_spike_anomaly_exceeding_lower_bound():
    spike = Spike()
    data = np.array([-1, -2, -3, -4, -5])
    result = spike.apply_anomaly(data, -5, 5, -5)
    assert np.array_equal(result, np.array([-5, -5, -5, -5, -5]))

def test_spike_anomaly_with_start_and_end_index():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    result = spike.apply_anomaly(data, 2, 5, 0, 1, 4)
    assert np.array_equal(result, np.array([1, 4, 5, 5, 5]))

def test_spike_anomaly_with_negative_start_index():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        spike.apply_anomaly(data, 2, 5, 0, -1, 4)

def test_spike_anomaly_with_start_index_greater_than_data_length():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        spike.apply_anomaly(data, 2, 5, 0, 6, 4)

def test_spike_anomaly_with_start_index_greater_than_end_index():
    spike = Spike()
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        spike.apply_anomaly(data, 2, 5, 0, 3, 2)