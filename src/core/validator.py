import time
import psutil
import numpy as np
import scipy.signal
import mne


class Validator:
    """Validation and benchmarking utilities for signal processing algorithms."""

    @staticmethod
    def compare_filters(raw_data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> dict:
        """Compare MNE filter (ground truth) with SciPy Butterworth implementation.
        
        Args:
            raw_data: 1D signal array
            sfreq: Sampling frequency in Hz
            l_freq: Low cutoff frequency
            h_freq: High cutoff frequency
            
        Returns:
            Dictionary with filtered signals, residuals, and MSE
        """
        b, a = scipy.signal.butter(4, [l_freq, h_freq], btype='bandpass', fs=sfreq)
        scipy_filtered = scipy.signal.lfilter(b, a, raw_data)
        
        mne_data = raw_data.reshape(1, -1)
        mne_filtered_full = mne.filter.filter_data(mne_data, sfreq, l_freq, h_freq, method='iir', verbose=False)
        mne_filtered = mne_filtered_full[0]

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
        """Measure execution time and memory usage of a function.
        
        Returns:
            Dictionary with execution_time_ms, memory_used_mb, and function result
        """
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / 1024 / 1024
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        mem_after_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "execution_time_ms": (end - start) * 1000,
            "memory_used_mb": mem_after_mb - mem_before_mb,
            "result": result
        }
