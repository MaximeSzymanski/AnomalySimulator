from abc import ABC, abstractmethod
from typing import List
import numpy as np
class Anomaly(ABC):
    """
    Abstract class that represents an anomaly
    """
    name : str


    @abstractmethod
    def __init__(self, name: str):
        """
        Constructor for the Anomaly class
        :param name:
        """
        self.name = name

