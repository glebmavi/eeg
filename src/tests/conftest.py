import pytest
from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import mne
import tempfile
import os
from pathlib import Path
from src.core.data_manager import DataManager

@pytest.fixture(scope="session")
def qapp():
    """
    Fixture to ensure a QApplication exists for the entire test session.
    Required for any PyQt widgets.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app

@pytest.fixture
def clean_data_manager():
    """
    Ensures DataManager singleton is clean before and after each test.
    """
    dm = DataManager()
    dm.signals = {}
    dm.listeners = []
    yield dm
    dm.signals = {}
    dm.listeners = []

@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Creates a temporary CSV file with configurable EEG-like data.
    
    Returns a callable that creates CSV files with specified parameters.
    """
    def _create_csv(n_channels=3, n_samples=100, include_time=True, 
                    include_noise=False, channel_prefix="Ch"):
        file_path = tmp_path / "test_data.csv"
        
        # Generate time column
        time = np.linspace(0, n_samples / 250.0, n_samples)
        
        # Generate channel data
        data = {}
        if include_time:
            data['Time'] = time
        
        for i in range(n_channels):
            if include_noise:
                # Signal with noise
                signal = np.sin(2 * np.pi * 10 * time) + np.random.normal(0, 0.1, n_samples)
            else:
                # Clean sine wave
                signal = np.sin(2 * np.pi * 10 * time) * 100  # Scale to uV range
            data[f'{channel_prefix}{i+1}'] = signal
        
        # Write to CSV
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    return _create_csv

@pytest.fixture
def sample_raw_data():
    """
    Provides MNE Raw objects with known signals for testing.
    
    Returns a callable that creates Raw objects with specified parameters.
    """
    def _create_raw(n_channels=1, n_samples=1000, sfreq=250.0, 
                    signal_type='sine', freq=10.0, amplitude=1.0):
        """
        Create a sample MNE Raw object.
        
        Args:
            n_channels: Number of channels
            n_samples: Number of samples
            sfreq: Sampling frequency
            signal_type: 'sine', 'noise', 'peaks', 'flat', 'composite'
            freq: Frequency for sine wave (Hz)
            amplitude: Signal amplitude
        """
        t = np.arange(n_samples) / sfreq
        data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            if signal_type == 'sine':
                data[ch] = amplitude * np.sin(2 * np.pi * freq * t)
            elif signal_type == 'noise':
                data[ch] = amplitude * np.random.normal(0, 1, n_samples)
            elif signal_type == 'peaks':
                # Signal with distinct peaks
                data[ch] = np.zeros(n_samples)
                peak_indices = np.linspace(100, n_samples-100, 5, dtype=int)
                for idx in peak_indices:
                    data[ch, idx] = amplitude * 5.0
            elif signal_type == 'flat':
                data[ch] = np.zeros(n_samples)
            elif signal_type == 'composite':
                # Multiple frequencies
                data[ch] = (amplitude * np.sin(2 * np.pi * 10 * t) + 
                           amplitude * 0.5 * np.sin(2 * np.pi * 50 * t))
        
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        return raw
    
    return _create_raw

@pytest.fixture
def sample_signals():
    """
    Provides various pre-defined signal arrays for testing signal processing.
    """
    sfreq = 250.0
    duration = 2.0  # seconds
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    
    return {
        'sine_10hz': np.sin(2 * np.pi * 10 * t),
        'sine_50hz': np.sin(2 * np.pi * 50 * t),
        'composite': np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t),
        'noise': np.random.normal(0, 1, n_samples),
        'flat': np.zeros(n_samples),
        'peaks': np.concatenate([np.zeros(100), [10.0], np.zeros(100), [8.0], np.zeros(n_samples-302)]),
        'linear_trend': np.linspace(0, 10, n_samples) + np.sin(2 * np.pi * 10 * t),
        't': t,
        'sfreq': sfreq
    }

@pytest.fixture
def temp_edf_file(tmp_path, sample_raw_data):
    """
    Creates a temporary EDF file for testing.
    """
    def _create_edf(n_channels=2, n_samples=500):
        file_path = tmp_path / "test_data.edf"
        raw = sample_raw_data(n_channels=n_channels, n_samples=n_samples, signal_type='sine')
        mne.export.export_raw(str(file_path), raw, fmt='edf', overwrite=True, verbose=False)
        return str(file_path)
    
    return _create_edf