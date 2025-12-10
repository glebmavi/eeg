from dataclasses import dataclass

@dataclass(frozen=True)
class FrequencyBand:
    """Represents a frequency band for EEG analysis."""
    name: str
    low: float
    high: float
    color: str

class RhythmBands:
    """Standard EEG rhythm bands."""
    DELTA = FrequencyBand('delta', 0.5, 4.0, 'c')
    THETA = FrequencyBand('theta', 4.0, 8.0, 'm')
    ALPHA = FrequencyBand('alpha', 8.0, 13.0, 'r')
    BETA = FrequencyBand('beta', 13.0, 30.0, 'g')
    GAMMA = FrequencyBand('gamma', 30.0, 100.0, 'y')

    @classmethod
    def all_bands(cls):
        return [cls.DELTA, cls.THETA, cls.ALPHA, cls.BETA, cls.GAMMA]

    @classmethod
    def get_band(cls, name: str) -> FrequencyBand | None:
        name_lower = name.lower()
        for band in cls.all_bands():
            if band.name == name_lower:
                return band
        return None

@dataclass
class FilterState:
    """Mutable state for signal processing filters."""
    notch: bool = False
    detrend: bool = False
    l_freq: float = 1.0
    h_freq: float = 40.0

@dataclass
class AnalysisState:
    """Mutable state for interactive analysis toggles."""
    delta: bool = False
    theta: bool = False
    alpha: bool = False
    beta: bool = False
    gamma: bool = False
    peaks: bool = False