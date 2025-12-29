from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd


class IStorage(ABC):
    """Interface for data storage."""

    @abstractmethod
    def save_csv(self, filename: str, header: List[str], data: List[List[Any]]) -> None:
        """Save data to CSV."""
        pass


class PandasStorage(IStorage):
    """Implementation of simple pandas storage."""

    def save_csv(self, filename: str, header: List[str], data: List[List[Any]]) -> None:
        df = pd.DataFrame(data, columns=header)
        df.to_csv(filename, index=False)
