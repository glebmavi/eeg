import pytest
import numpy as np
import mne
import os
from src.core.loader import DataLoader
from src.core.data_manager import DataManager
from src.core.processor import SignalProcessor

# --- Data Manager Tests ---
def test_data_manager_singleton():
    dm1 = DataManager()
    dm2 = DataManager()
    assert dm1 is dm2
    # Cleanup
    dm1.signals = {}

def test_add_remove_signal():
    dm = DataManager()
    info = mne.create_info(['Ch1'], 100, ['eeg'])
    data = np.zeros((1, 100))
    raw = mne.io.RawArray(data, info)
    
    name = dm.add_signal("test_sig", raw)
    assert name == "test_sig"
    assert "test_sig" in dm.get_signal_names()
    
    dm.remove_signal("test_sig")
    assert "test_sig" not in dm.get_signal_names()

# --- Signal Processor Tests ---
def test_filter_application():
    # Create a signal: 10Hz sine wave + 50Hz noise
    sfreq = 200
    t = np.linspace(0, 1, sfreq)
    # Signal: 10Hz
    sig = np.sin(2 * np.pi * 10 * t)
    # Noise: 50Hz
    noise = np.sin(2 * np.pi * 50 * t)
    data = (sig + noise).reshape(1, -1)
    
    info = mne.create_info(['Ch1'], sfreq, ['eeg'])
    raw = mne.io.RawArray(data, info)
    
    # Filter out 50Hz (Low pass 30Hz)
    filtered = SignalProcessor.apply_filter(raw, l_freq=1, h_freq=30)
    res_data = filtered.get_data()[0]
    
    # Calculate power at 50Hz (should be low)
    # Simple check: amplitude of signal should be close to 1 (the 10Hz part) 
    # and 50Hz part gone.
    assert np.std(res_data) < np.std(data[0]) # Energy should decrease
    assert np.allclose(res_data, sig, atol=0.2) # Should resemble original sine

def test_peak_detection():
    data = np.zeros(100)
    data[50] = 10 # Explicit peak
    
    peaks, _ = SignalProcessor.detect_peaks(data, height=5)
    assert len(peaks) == 1
    assert peaks[0] == 50

# --- Loader Tests ---
def test_csv_loader_heuristics(tmp_path):
    # Create dummy CSV
    d = tmp_path / "test.csv"
    csv_content = "Time,FP1,FP2\n0.0,10,12\n0.1,11,13\n"
    d.write_text(csv_content)
    
    raw = DataLoader.load_data(str(d), sfreq=10.0, time_col='Time', unit_scale=1.0)
    
    assert 'FP1' in raw.ch_names
    assert raw.info['sfreq'] == 10.0
    assert raw.get_data().shape == (2, 2)