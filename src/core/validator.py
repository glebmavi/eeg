import time
import psutil
import numpy as np
import scipy.signal
import mne

class Validator:
    """
    Module for validation and benchmarking of signal processing algorithms.
    """

    @staticmethod
    def compare_filters(raw_data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> dict:
        """
        Compare MNE's filter with a manual SciPy implementation (Butterworth).
        
        Args:
            raw_data (np.ndarray): 1D array of raw signal data (one channel).
            sfreq (float): Sampling frequency.
            l_freq (float): Low cut-off frequency.
            h_freq (float): High cut-off frequency.
            
        Returns:
            dict: Contains 'scipy_filtered', 'mne_filtered', 'residuals', 'mse'.
        """
        # 1. Manual SciPy Filter (Butterworth Bandpass)
        b, a = scipy.signal.butter(4, [l_freq, h_freq], btype='bandpass', fs=sfreq)
        scipy_filtered = scipy.signal.lfilter(b, a, raw_data)
        
        # 2. MNE Filter
        # MNE usually operates on (n_channels, n_times), so reshape if needed
        mne_data = raw_data.reshape(1, -1)
        # method='iir' uses Butterworth by default for IIR in MNE (approx equivalent to above if configured right)
        # Note: MNE default is FIR, so we explicitly ask for IIR to compare apples to apples if possible,
        # OR we compare standard MNE FIR vs Manual SciPy IIR to show differences.
        # The requirement says: "Compare result with filter mne.filter.filter_data (ground truth)"
        
        mne_filtered_full = mne.filter.filter_data(mne_data, sfreq, l_freq, h_freq, method='iir', verbose=False)
        mne_filtered = mne_filtered_full[0]

        # 3. Residuals
        residuals = scipy_filtered - mne_filtered
        mse = np.mean(residuals ** 2)
        
        return {
            "scipy_filtered": scipy_filtered,
            "mne_filtered": mne_filtered,
            "residuals": residuals,
            "mse": mse
        }

    @staticmethod
    def measure_performance(func, *args, **kwargs) -> dict:
        """
        Measure execution time and RAM usage of a function.
        
        Args:
            func (callable): Function to benchmark.
            *args, **kwargs: Arguments for the function.
            
        Returns:
            dict: 'execution_time_ms', 'memory_used_mb', 'result'
        """
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        mem_after = process.memory_info().rss / 1024 / 1024 # MB
        
        exec_time_ms = (end_time - start_time) * 1000
        mem_diff = mem_after - mem_before
        
        return {
            "execution_time_ms": exec_time_ms,
            "memory_used_mb": mem_diff,
            "result": result
        }
