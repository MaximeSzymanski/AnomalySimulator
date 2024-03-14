from Anomalies.classes.Anomaly import Anomaly
import numpy as np
class Lag(Anomaly):
    def __init__(self):
        super().__init__("Lag")

    def apply_anomaly(self, data: np.ndarray, lag_coeff: int, upper_bound: float, lower_bound: float, start_index: int or None = None, end_index: int or None = None) -> np.ndarray:
        """
        Apply the lag anomaly to the data
        :param data: The data to apply the anomaly to
        :param lag_coeff:  The lag to apply to the data (positive or negative)
        :param upper_bound: The upper bound of the sensor to apply the anomaly to
        :param lower_bound: The lower bound of the sensor to apply the anomaly to
        :param start_index: The start index of the anomaly
        :param end_index: The end index of the anomaly
        :return: The data with the anomaly applied
        """

        assert lag_coeff != 0, "Lag must be non-zero"
        assert abs(lag_coeff) < len(data), "Lag must be less than the length of the data"

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

        lag_coeff = int(round(lag_coeff))  # Convert lag_coeff to the nearest integer

        # if lag is > 0, shift the data to the right (add lag_coeff to the index) and pad the left with lower_bound

        if lag_coeff > 0:
            data[start_index:end_index] = np.concatenate([np.full(lag_coeff, lower_bound), data[start_index:end_index - lag_coeff]])
        # if lag is < 0, shift the data to the left (subtract lag_coeff from the index) and pad the right with upper_bound
        elif lag_coeff < 0:
            data[start_index:end_index] = np.concatenate([data[start_index - lag_coeff:end_index], np.full(-lag_coeff, upper_bound)])


        return data
