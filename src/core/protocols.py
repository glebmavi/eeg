from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import threading
import time
import numpy as np

# Third-party imports
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    from pyfirmata2 import Arduino
except ImportError:
    Arduino = None

from src.models.config import ProtocolConfig


class IProtocol(ABC):
    """Abstract interface for Arduino communication protocols."""

    def __init__(self):
        self.config: Optional[ProtocolConfig] = None

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""

    @abstractmethod
    def read_raw(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        """Read a single frame/packet/line of raw data."""

    @abstractmethod
    def configure(self, config: ProtocolConfig, channels: List[str]) -> None:
        """Apply/Update configuration."""

    @staticmethod
    def list_available_ports() -> List[str]:
        """Static method to list available COM ports."""
        if serial:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        return []


class PySerialProtocol(IProtocol):
    """Implementation of protocol using pySerial."""

    def __init__(self):
        super().__init__()
        self.serial_conn: Optional[serial.Serial] = None
        self.config: Optional[ProtocolConfig] = None

    def configure(self, config: ProtocolConfig, channels: List[str] = None) -> None:
        self.config = config

    def connect(self) -> None:
        if not serial:
            raise ImportError("pySerial is not installed.")
        if not self.config:
            raise ValueError("Configuration not set. Call configure().")
        try:
            self.serial_conn = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=1.0
            )
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.config.port}: {e}")

    def disconnect(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.serial_conn = None

    def is_connected(self) -> bool:
        return self.serial_conn is not None and self.serial_conn.is_open

    def read_raw(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        if timeout_ms is not None and self.serial_conn:
            self.serial_conn.timeout = timeout_ms / 1000.0

        if self.is_connected():
            try:
                line = self.serial_conn.readline()
                return line if line else None
            except serial.SerialException:
                self.disconnect()
                return None
        return None


class FirmataProtocol(IProtocol):
    """Firmata implementation using pyFirmata2."""

    def __init__(self):
        super().__init__()
        self.board: Optional[Arduino] = None
        self.config: Optional[ProtocolConfig] = None
        self.channels: List[str] = []
        self.pins: Dict[str, object] = {}
        self.latest_values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def configure(self, config: ProtocolConfig, channels: List[str]) -> None:
        self.config = config
        self.channels = channels

    def connect(self) -> None:
        if not Arduino:
            raise ImportError("pyfirmata2 is not installed.")
        if not self.config:
            raise ValueError("Configuration not set.")

        try:
            self.board = Arduino(self.config.port)
            sampling_ms = 19
            if self.config.extra:
                sampling_ms = int(self.config.extra.get("sampling_ms", sampling_ms))

            self.board.samplingOn(sampling_ms)

            for pin_name in self.channels:
                # Expecting 'A0', 'A1' etc.
                idx = int(pin_name[1:])
                pin = self.board.analog[idx]

                def make_cb(name):
                    def _cb(value):
                        with self._lock:
                            self.latest_values[name] = float(value) if value is not None else 0.0
                    return _cb

                pin.register_callback(make_cb(pin_name))
                pin.enable_reporting()
                self.pins[pin_name] = pin
                self.latest_values[pin_name] = 0.0
                time.sleep(0.02)

            time.sleep(0.05)

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Firmata {self.config.port}: {e}")

    def disconnect(self) -> None:
        try:
            if self.board:
                try:
                    self.board.samplingOff()
                except Exception:
                    pass
                try:
                    self.board.exit()
                except Exception:
                    pass
        finally:
            self.board = None
            self.pins.clear()
            with self._lock:
                self.latest_values.clear()

    def is_connected(self) -> bool:
        return self.board is not None

    def read_raw(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        if not self.is_connected():
            return None

        with self._lock:
            values = []
            for pin in self.channels:
                v = self.latest_values.get(pin, 0.0)
                adc = int(max(0.0, min(1.0, v)) * 1023)
                values.append(str(adc))

        raw_line = (",".join(values) + "\r\n").encode("utf-8")
        time.sleep(0.001)
        return raw_line


class SimulatorProtocol(IProtocol):
    """Simulator protocol for testing."""

    def __init__(self):
        super().__init__()
        self.config = None
        self.channels = []
        self.start_time = 0
        self.freqs = {}

    def configure(self, config: ProtocolConfig, channels: List[str]) -> None:
        self.config = config
        self.channels = channels
        base_freqs = [1, 5, 8, 12]
        for i, pin_name in enumerate(self.channels):
            self.freqs[pin_name] = base_freqs[i % len(base_freqs)]

    def connect(self) -> None:
        self.start_time = time.time()

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return self.start_time > 0

    def read_raw(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        t = time.time() - self.start_time
        values = []

        for pin_name in self.channels:
            freq = self.freqs[pin_name]
            # Signal 0-1023
            signal = 512 + 511 * np.sin(2 * np.pi * freq * t)
            noise = np.random.normal(0, 20)
            val = np.clip(signal + noise, 0, 1023)
            values.append(f"{val:.0f}")

        time.sleep(0.002)
        raw_line = (",".join(values) + "\r\n").encode('utf-8')
        return raw_line


class IProtocolFactory(ABC):
    @abstractmethod
    def create(self, config: ProtocolConfig) -> IProtocol:
        pass


class ProtocolFactory(IProtocolFactory):
    def create(self, config: ProtocolConfig) -> IProtocol:
        if config.protocol == 'pyserial':
            return PySerialProtocol()
        elif config.protocol == 'firmata':
            return FirmataProtocol()
        elif config.protocol == 'simulator':
            return SimulatorProtocol()
        else:
            raise ValueError(f"Unknown protocol: {config.protocol}")
