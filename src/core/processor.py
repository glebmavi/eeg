import numpy as np
import scipy.signal
import mne

class SignalProcessor:
    """
    Handles signal processing tasks: filtering, detrending, and peak detection.
    """

    @staticmethod
    def apply_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, method: str = 'iir') -> mne.io.BaseRaw:
        """Apply bandpass filter relative to fs."""
        inst = raw.copy()
        inst.filter(l_freq, h_freq, method=method)
        return inst

    @staticmethod
    def apply_notch(raw: mne.io.BaseRaw, freqs: np.ndarray | list) -> mne.io.BaseRaw:
        """Apply notch filter to remove specific frequencies."""
        inst = raw.copy()
        inst.notch_filter(freqs=freqs)
        return inst

    @staticmethod
    def detrend_signal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Remove linear trend from the signal channel-wise."""
        inst = raw.copy()
        inst.apply_function(scipy.signal.detrend, channel_wise=True)
        return inst

    @staticmethod
    def detect_peaks(data: np.ndarray, height: float = None, distance: int = None) -> tuple:
        """Detect peaks in a 1D signal array."""
        if data is None or len(data) == 0:
            return np.array([]), {}
            
        # Avoid processing flat signals
        if np.all(data == data[0]):
             return np.array([]), {}
             
        return scipy.signal.find_peaks(data, height=height, distance=distance)
