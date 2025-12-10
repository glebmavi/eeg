import numpy as np
import scipy.signal
import mne

class SignalProcessor:
    """
    Handles signal processing tasks: filtering, ICA, Wavelets, and Feature Extraction.
    """

    @staticmethod
    def apply_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, method: str = 'iir') -> mne.io.BaseRaw:
        """Apply bandpass filter relative to fs."""
        inst = raw.copy()
        inst.filter(l_freq, h_freq, method=method, verbose=False)
        return inst

    @staticmethod
    def apply_notch(raw: mne.io.BaseRaw, freqs: np.ndarray | list) -> mne.io.BaseRaw:
        """Apply notch filter to remove specific frequencies."""
        inst = raw.copy()
        inst.notch_filter(freqs=freqs, verbose=False)
        return inst

    @staticmethod
    def detrend_signal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Remove linear trend from the signal channel-wise."""
        inst = raw.copy()
        inst.apply_function(scipy.signal.detrend, channel_wise=True, verbose=False)
        return inst

    @staticmethod
    def apply_ica(raw: mne.io.BaseRaw, n_components: int = 15, random_state=97) -> mne.io.BaseRaw:
        """
        Applies Independent Component Analysis (ICA) to remove artifacts.
        Method: Fits FastICA and automatically excludes the first component (often eye blinks).
        """
        inst = raw.copy()
        # ICA requires filtering for best fit, typically 1Hz highpass
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
        # Welch's Periodogram
        freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=min(len(data), int(sfreq * 2)))

        # Integrate power in bands
        bands = { # TODO: Use power bands from models.types.py
            'Delta (0.5-4Hz)': (0.5, 4),
            'Theta (4-8Hz)': (4, 8),
            'Alpha (8-13Hz)': (8, 13),
            'Beta (13-30Hz)': (13, 30),
            'Gamma (30-100Hz)': (30, 100)
        }

        total_power = np.sum(psd)
        if total_power == 0:
            return {k: 0.0 for k in bands}

        features = {}
        for name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(psd[idx])
            features[name] = band_power / total_power

        return features

    @staticmethod
    def detect_peaks(data: np.ndarray, height: float = None, distance: int = None) -> tuple:
        """Detect peaks in a 1D signal array."""
        if data is None or len(data) == 0:
            return np.array([]), {}
        if np.all(data == data[0]):
            return np.array([]), {}
        return scipy.signal.find_peaks(data, height=height, distance=distance)
