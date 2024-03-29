from Anomalies.classes.Anomaly import Anomaly
import numpy as np


class Drift(Anomaly):
    def __init__(self):
        super().__init__("Drift")

    def apply_anomaly(self, data: np.ndarray, lower_bound_sensor: float, upper_bound_sensor: float, drift_coeff: float,
                      start_index: int or None = None, end_index: int or None = None) -> np.ndarray:

        """
        Apply the drift anomaly to the data. If the data is above the upper bound, it will be set to the upper bound.
        If the data is below the lower bound, it will be set to the lower bound.
        If the drift coefficient is positive, the data will be increased. If the drift coefficient is negative, the data will be decreased.

        :param data: The data to apply the anomaly to
        :param lower_bound_sensor: The lower bound of the sensor to apply the anomaly to
        :param upper_bound_sensor: The upper bound of the sensor to apply the anomaly to
        :param drift_coeff: The coefficient to apply to the data
        :param start_index: The start index of the anomaly
        :param end_index: The end index of the anomaly
        :return The data with the anomaly applied
        """
        if start_index is not None:
            assert start_index >= 0, "Start index must be greater than or equal to 0"
            assert start_index < len(data), "Start index must be less than the length of the data"
        if end_index is not None:
            assert end_index >= 0, "End index must be greater than or equal to 0"
            assert end_index < len(data), "End index must be less than the length of the data"
        if start_index is not None and end_index is not None:
            assert start_index < end_index, "Start index must be less than end index"

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(data)

        # Apply the drift anomaly to the data
        for i in range(start_index, end_index):
            data[i] += drift_coeff * i*1e-1

            # Ensure the data doesn't go beyond sensor bounds
            data[i] = max(lower_bound_sensor, min(upper_bound_sensor, data[i]))

        return data
