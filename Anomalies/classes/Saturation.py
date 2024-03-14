import numpy as np

from Anomalies.classes.Anomaly import Anomaly


class Saturation(Anomaly):
    """
    Class that represents the drop anomaly
    """

    def __init__(self):
        super().__init__("Saturation")

    def apply_anomaly(self, data: np.ndarray, upper_bound_sensor: float, start_index: int or None = None,
                      end_index: int or None = None) -> np.ndarray:

        """
        Apply the drop anomaly to the data
        :param data: The data to apply the anomaly to
        :param upper_bound_sensor: The upper bound of the sensor to apply the anomaly to
        :param start_index: The start index of the anomaly
        :param end_index: The end index of the anomaly
        :return: The data with the anomaly applied
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

        data[start_index:end_index] = upper_bound_sensor

        return data