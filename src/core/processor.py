import numpy as np
import scipy.signal
import mne
from src.models.types import RhythmBands


class SignalProcessor:
    """Signal processing operations: filtering, ICA, wavelets, and feature extraction."""

    @staticmethod
    def apply_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, method: str = 'iir') -> mne.io.BaseRaw:
        """Apply bandpass filter to signal."""
        inst = raw.copy()
        inst.filter(l_freq, h_freq, method=method, verbose=False)
        return inst

    @staticmethod
    def apply_notch(raw: mne.io.BaseRaw, freqs: np.ndarray | list) -> mne.io.BaseRaw:
        """Remove specific frequency components (50Hz line noise)."""
        inst = raw.copy()
        inst.notch_filter(freqs=freqs, verbose=False)
        return inst

    @staticmethod
    def detrend_signal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Remove linear trend from signal."""
        inst = raw.copy()
        inst.apply_function(scipy.signal.detrend, channel_wise=True, verbose=False)
        return inst

    @staticmethod
    def apply_ica(raw: mne.io.BaseRaw, n_components: int = 15, random_state=97) -> mne.io.BaseRaw:
        """Apply ICA to remove artifacts (e.g., eye blinks, muscle noise).
        
        Uses FastICA with automatic exclusion of the first component, which often
        captures prominent artifacts. In production, component selection should be
        based on correlation with EOG/ECG channels.
        TODO: rewrite this apply ica, to allow user to mark some channels as non eeg (to get artifacts), then use them for ica. This requires changing the loader for ALL data formats, to create a dictionary of channels, which are non eeg.
        """
        inst = raw.copy()
        ica_fit_raw = inst.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

        # Determine components (min of n_channels or n_components)
        n_ch = len(inst.ch_names)
        n_comp = min(n_ch, n_components)
        if n_comp < 2:
            return inst  # Cannot do ICA with < 2 channels/components efficiently

        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=random_state, method='fastica')
        ica.fit(ica_fit_raw, verbose=False)

        # Heuristic: Exclude the first component (often blinks/cardiac if prominent)
        # In a real scenario, we would correlate with EOG/ECG channels.
        ica.exclude = [0]

        return ica.apply(inst, verbose=False)

    @staticmethod
    def extract_band_powers(data: np.ndarray, sfreq: float) -> dict:
        """
        Calculates relative power in standard EEG bands (Feature Extraction).
        Returns a dictionary of {band_name: relative_power}.
        """
        freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=min(len(data), int(sfreq * 2)))
        bands = RhythmBands.all_bands()

        total_power = np.sum(psd)
        if total_power == 0:
            return {k: 0.0 for k in bands}

        features = {}
        for band in bands:
            label = f"{band.name.capitalize()} ({band.low}-{band.high}Hz)"
            idx = np.logical_and(freqs >= band.low, freqs <= band.high)
            band_power = np.sum(psd[idx])
            features[label] = band_power / total_power

        return features

    @staticmethod
    def detect_peaks(data: np.ndarray, height: float = None, distance: int = None) -> tuple:
        """Detect peaks in 1D signal using scipy.signal.find_peaks."""
        if data is None or len(data) == 0 or np.all(data == data[0]):
            return np.array([]), {}
        return scipy.signal.find_peaks(data, height=height, distance=distance)
