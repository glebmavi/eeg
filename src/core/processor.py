import numpy as np
import scipy.signal
import mne

class SignalProcessor:
    """
    Handles signal processing tasks: filtering, detrending, and feature extraction.
    """

    @staticmethod
    def apply_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, method: str = 'iir') -> mne.io.BaseRaw:
        """
        Apply a bandpass filter to the data.
        
        Args:
            raw (mne.io.BaseRaw): The EEG data.
            l_freq (float): Low cut-off frequency.
            h_freq (float): High cut-off frequency.
            method (str): Method to use ('iir' or 'fir').
            
        Returns:
            mne.io.BaseRaw: Filtered data (in-place modification copy returned for chaining).
        """
        # We work on a copy to avoid modifying the original data in place unintentionally if not desired
        inst = raw.copy()
        inst.filter(l_freq, h_freq, method=method)
        return inst

    @staticmethod
    def apply_notch(raw: mne.io.BaseRaw, freqs: np.ndarray | list) -> mne.io.BaseRaw:
        """
        Apply a notch filter to remove specific frequencies (e.g., power line noise).
        
        Args:
            raw (mne.io.BaseRaw): The EEG data.
            freqs (list or np.ndarray): Frequencies to notch filter.
        
        Returns:
            mne.io.BaseRaw: Filtered data.
        """
        inst = raw.copy()
        inst.notch_filter(freqs=freqs)
        return inst

    @staticmethod
    def detrend_signal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Remove linear trend from the signal.
        Note: MNE's filter often handles detrending, but we can do it explicitly or on specific epochs.
        Here we apply it to the continuous raw data channel-wise using scipy.signal.detrend via apply_function.
        """
        inst = raw.copy()
        inst.apply_function(scipy.signal.detrend, channel_wise=True)
        return inst

    @staticmethod
    def detect_peaks(data: np.ndarray, height: float = None, distance: int = None) -> tuple:
        """
        Detect peaks in a 1D signal array (e.g., one channel).
        """
        if data is None or len(data) == 0:
            return np.array([]), {}
            
        # Safety check for flat signal
        if np.all(data == data[0]):
             return np.array([]), {}
             
        peaks, properties = scipy.signal.find_peaks(data, height=height, distance=distance)
        return peaks, properties
