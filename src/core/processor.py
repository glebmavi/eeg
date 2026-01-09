import numpy as np
import scipy.signal
import scipy.integrate
import mne
from src.models.types import RhythmBands


class SignalProcessor:
    """Signal processing operations: filtering, ICA, wavelets, and feature extraction."""

    @staticmethod
    def apply_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, method: str = 'iir',
                     normalize_output: bool = False) -> mne.io.BaseRaw:
        """
        Apply bandpass filter to signal with Nyquist safety checks.

        Args:
            raw: Input MNE Raw object
            l_freq: Low cutoff frequency
            h_freq: High cutoff frequency
            method: Filtering method ('iir' or 'fir')
            normalize_output: If True, applies MinMax normalization to [-1, 1] after filtering.
        """
        inst = raw.copy()
        sfreq = inst.info['sfreq']
        nyquist = sfreq / 2.0

        # Nyquist Check
        if h_freq >= nyquist:
            print(f"Warning: High frequency {h_freq}Hz exceeds/equals Nyquist ({nyquist}Hz). Clamping.")
            h_freq = nyquist - 0.5  # Clamp slightly below Nyquist

        if h_freq <= l_freq:
            print(f"Warning: Adjusted High freq ({h_freq}) <= Low freq ({l_freq}). Skipping filter.")
            return inst

        inst.filter(l_freq, h_freq, method=method, verbose=False)

        if normalize_output:
            return SignalProcessor.normalize_signal(inst, method='minmax')

        return inst

    @staticmethod
    def apply_notch(raw: mne.io.BaseRaw, freqs: np.ndarray | list) -> mne.io.BaseRaw:
        """Remove specific frequency components (50Hz line noise)."""
        inst = raw.copy()
        sfreq = inst.info['sfreq']
        nyquist = sfreq / 2.0

        valid_freqs = [f for f in freqs if f < nyquist]
        if not valid_freqs:
            print(f"Warning: All notch frequencies {freqs} >= Nyquist ({nyquist}). Skipping notch.")
            return inst

        inst.notch_filter(freqs=valid_freqs, verbose=False)
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

        Uses FastICA. If EOG or ECG channels are defined, it attempts to find and
        exclude components correlated with them. Otherwise, it excludes the first
        component by default (often blinks).
        """
        inst = raw.copy()
        ica_fit_raw = inst.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

        # Determine components based on valid data channels (excluding bads/non-data)
        picks = mne.pick_types(inst.info, meg=True, eeg=True, eog=False, ecg=False,
                               stim=False, exclude='bads')
        n_comp = min(len(picks), n_components)

        if n_comp < 2:
            return inst  # Cannot do ICA with < 2 channels/components efficiently

        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=random_state, method='fastica')
        ica.fit(ica_fit_raw, verbose=False)

        # Artifact Detection
        exclude_inds = []

        # 1. EOG (Eye Blinks)
        if 'eog' in inst.get_channel_types():
            eog_inds, _ = ica.find_bads_eog(inst, verbose=False)
            if eog_inds:
                exclude_inds.extend(eog_inds)

        # 2. ECG (Heartbeat)
        if 'ecg' in inst.get_channel_types():
            ecg_inds, _ = ica.find_bads_ecg(inst, verbose=False)
            if ecg_inds:
                exclude_inds.extend(ecg_inds)

        # 3. Fallback: Exclude first component if nothing else found
        if not exclude_inds:
            exclude_inds = [0]

        ica.exclude = list(set(exclude_inds))  # Deduplicate

        return ica.apply(inst, verbose=False)

    @staticmethod
    def extract_band_powers(data: np.ndarray, sfreq: float) -> dict:
        """
        Calculates relative and absolute power in standard EEG bands (Feature Extraction).
        Returns a dictionary of {band_name: {'relative': float, 'absolute': float}}.
        """
        # Improved Resolution: Use sfreq * 4 to match visual PSD plot and minimize leakage
        nperseg = min(len(data), int(sfreq * 4))
        freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=nperseg)
        bands = RhythmBands.all_bands()

        # Integrate total power using Simpson's rule for accuracy
        freq_res = freqs[1] - freqs[0]
        total_power = scipy.integrate.simpson(psd, dx=freq_res)

        if total_power == 0:
            return {f"{band.name.capitalize()} ({band.low}-{band.high}Hz)": {'relative': 0.0, 'absolute': 0.0} for band
                    in bands}

        features = {}
        for band in bands:
            label = f"{band.name.capitalize()} ({band.low}-{band.high}Hz)"

            # Find indices within the band
            idx = np.logical_and(freqs >= band.low, freqs <= band.high)

            # If no frequencies fall in this band (unlikely with sufficient nperseg), 0 power
            if not np.any(idx):
                features[label] = {'relative': 0.0, 'absolute': 0.0}
                continue

            # Integrate power in this band
            band_power = scipy.integrate.simpson(psd[idx], dx=freq_res)

            features[label] = {
                'relative': band_power / total_power,
                'absolute': band_power
            }

        return features

    @staticmethod
    def detect_peaks(data: np.ndarray, height: float = None, prominence: float = None, distance: int = None) -> tuple:
        """Detect peaks in 1D signal using scipy.signal.find_peaks."""
        if data is None or len(data) == 0 or np.all(data == data[0]):
            return np.array([]), {}
        return scipy.signal.find_peaks(data, height=height, prominence=prominence, distance=distance)

    @staticmethod
    def normalize_signal(raw: mne.io.BaseRaw, method: str = 'zscore') -> mne.io.BaseRaw:
        """
        Normalize signal data.
        Supported methods:
        - 'zscore': Subtract mean and divide by standard deviation (standardization).
        - 'minmax': Scale to range [-1, 1].
        """
        inst = raw.copy()
        data = inst.get_data()

        if method == 'zscore':
            # Calculate mean and std for each channel
            means = np.mean(data, axis=1, keepdims=True)
            stds = np.std(data, axis=1, keepdims=True)

            # Avoid division by zero for flat channels
            stds[stds == 0] = 1.0

            # Apply Normalization
            normalized_data = (data - means) / stds

            # Create new RawArray with normalized data (preserving info)
            inst = mne.io.RawArray(normalized_data, inst.info, verbose=False)

        elif method == 'minmax':
            mins = np.min(data, axis=1, keepdims=True)
            maxs = np.max(data, axis=1, keepdims=True)

            ranges = maxs - mins
            ranges[ranges == 0] = 1.0

            # Apply MinMax Scaling to [-1, 1]
            normalized_data = 2 * (data - mins) / ranges - 1

            inst = mne.io.RawArray(normalized_data, inst.info, verbose=False)

        return inst
