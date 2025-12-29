from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ProtocolType(Enum):
    PYSERIAL = 'pyserial'
    FIRMATA = 'firmata'
    SIMULATOR = 'simulator'


class BaudRate(Enum):
    B_9600 = 9600
    B_19200 = 19200
    B_38400 = 38400
    B_57600 = 57600
    B_115200 = 115200


class BrainWave(Enum):
    ALPHA = "Alpha (8-13 Hz)"
    BETA = "Beta (13-30 Hz)"
    GAMMA = "Gamma (30-100 Hz)"
    THETA = "Theta (4-8 Hz)"
    DELTA = "Delta (0.5-4 Hz)"


@dataclass
class ProtocolConfig:
    protocol: str  # 'pyserial' | 'firmata' | 'simulator'
    port: str
    baudrate: int = 115200
    extra: Optional[Dict[str, Any]] = None


@dataclass
class SignalChannelConfig:
    name: str            # 'ch_0', 'ch_1', etc.
    arduino_pin: str     # 'A0', 'A1' ...
    enabled: bool = True


@dataclass
class AcquisitionConfig:
    refresh_ms: int = 50
    plot_window_s: float = 10.0
    channels: List[SignalChannelConfig] = field(default_factory=list)
    protocol: ProtocolConfig = None
