from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from PyQt6.QtCore import QMutex, QMutexLocker


class IDataBuffer(ABC):
    """Interface for thread-safe data buffer."""

    @abstractmethod
    def push(self, timestamp: float, values: dict) -> None:
        pass

    @abstractmethod
    def pop_all(self) -> List[Tuple[float, dict]]:
        """Extracts all data from buffer and clears it."""
        pass

    @abstractmethod
    def get_all_data(self) -> List[Tuple[float, dict]]:
        """Returns a copy of all data without clearing."""
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ThreadSafeBuffer(IDataBuffer):
    """
    Implementation of IDataBuffer using QMutex for thread safety.
    """
    def __init__(self):
        self.buffer: List[Tuple[float, Dict[str, Any]]] = []
        self.mutex = QMutex()

    def push(self, timestamp: float, values: dict) -> None:
        with QMutexLocker(self.mutex):
            self.buffer.append((timestamp, values))

    def pop_all(self) -> List[Tuple[float, dict]]:
        with QMutexLocker(self.mutex):
            data = list(self.buffer)
            self.buffer.clear()
            return data

    def get_all_data(self) -> List[Tuple[float, dict]]:
        with QMutexLocker(self.mutex):
            return list(self.buffer)

    def clear(self) -> None:
        with QMutexLocker(self.mutex):
            self.buffer.clear()
