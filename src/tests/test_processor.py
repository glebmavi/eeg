import pytest
import numpy as np
import mne
from src.core.processor import SignalProcessor


class TestSignalFiltering:
    """Tests for signal filtering operations."""
    
    def test_bandpass_filter_basic(self, sample_raw_data):
        """Test basic bandpass filtering."""
        raw = sample_raw_data(n_channels=1, signal_type='composite')
        filtered = SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=30.0)
        
        assert filtered is not None
        assert filtered.get_data().shape == raw.get_data().shape
        # Filtered should be different from original
        assert not np.allclose(filtered.get_data(), raw.get_data())
    
    def test_filter_removes_noise(self, sample_signals):
        """Test that low-pass filter removes high-frequency noise."""
        # Create signal: 10Hz clean + 50Hz noise
        sfreq = sample_signals['sfreq']
        signal = sample_signals['sine_10hz'] + sample_signals['sine_50hz']
        
        info = mne.create_info(['Ch1'], sfreq, ['eeg'])
        raw = mne.io.RawArray(signal.reshape(1, -1), info, verbose=False)
        
        # Apply low-pass filter at 30Hz (should remove 50Hz)
        filtered = SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=30.0)
        result = filtered.get_data()[0]
        
        # Result should be closer to 10Hz sine than original composite
        assert np.std(result) < np.std(signal)
    
    def test_filter_preserves_shape(self, sample_raw_data):
        """Test that filtering preserves data shape."""
        raw = sample_raw_data(n_channels=5, n_samples=1000)
        filtered = SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=40.0)
        
        assert filtered.get_data().shape == raw.get_data().shape
    
    def test_highpass_filter(self, sample_raw_data):
        """Test high-pass filter removes DC offset and low frequencies."""
        raw = sample_raw_data(n_channels=1, signal_type='sine', freq=5.0)
        
        # Add DC offset
        data = raw.get_data()
        data += 10.0
        info = mne.create_info(['Ch1'], raw.info['sfreq'], ['eeg'])
        raw_with_dc = mne.io.RawArray(data, info, verbose=False)
        
        # High-pass filter should remove DC
        filtered = SignalProcessor.apply_filter(raw_with_dc, l_freq=1.0, h_freq=None)
        
        # Mean should be close to 0 after high-pass
        assert abs(np.mean(filtered.get_data())) < abs(np.mean(raw_with_dc.get_data()))
    
    def test_filter_with_different_methods(self, sample_raw_data):
        """Test filtering with different methods (iir vs fir)."""
        raw = sample_raw_data(n_channels=1, signal_type='sine')
        
        filtered_iir = SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=40.0, method='iir')
        filtered_fir = SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=40.0, method='fir')
        
        # Both should produce results
        assert filtered_iir is not None
        assert filtered_fir is not None


class TestNotchFilter:
    """Tests for notch filter (line noise removal)."""
    
    def test_notch_filter_basic(self, sample_raw_data):
        """Test basic notch filter application."""
        raw = sample_raw_data(n_channels=1, signal_type='composite')
        filtered = SignalProcessor.apply_notch(raw, freqs=[50.0])
        
        assert filtered is not None
        assert filtered.get_data().shape == raw.get_data().shape
    
    def test_notch_removes_50hz(self, sample_signals):
        """Test that notch filter removes 50Hz component."""
        sfreq = sample_signals['sfreq']
        # Clean 10Hz + strong 50Hz noise
        signal = sample_signals['sine_10hz'] + 2.0 * sample_signals['sine_50hz']
        
        info = mne.create_info(['Ch1'], sfreq, ['eeg'])
        raw = mne.io.RawArray(signal.reshape(1, -1), info, verbose=False)
        
        # Apply notch filter at 50Hz
        filtered = SignalProcessor.apply_notch(raw, freqs=[50.0])
        
        # Power at 50Hz should be reduced
        # We can verify by checking that variance decreased
        assert np.var(filtered.get_data()) < np.var(raw.get_data())
    
    def test_notch_multiple_frequencies(self, sample_raw_data):
        """Test notch filter with multiple frequencies."""
        raw = sample_raw_data(n_channels=1, signal_type='composite')
        filtered = SignalProcessor.apply_notch(raw, freqs=[50.0, 60.0])
        
        assert filtered is not None


class TestDetrending:
    """Tests for signal detrending."""
    
    def test_detrend_removes_linear_trend(self, sample_signals):
        """Test that detrending removes linear trend."""
        sfreq = sample_signals['sfreq']
        signal_with_trend = sample_signals['linear_trend']
        
        info = mne.create_info(['Ch1'], sfreq, ['eeg'])
        raw = mne.io.RawArray(signal_with_trend.reshape(1, -1), info, verbose=False)
        
        # Apply detrending
        detrended = SignalProcessor.detrend_signal(raw)
        
        # Mean should decrease after removing trend
        assert abs(np.mean(detrended.get_data())) < abs(np.mean(raw.get_data()))
    
    def test_detrend_preserves_oscillations(self, sample_raw_data):
        """Test that detrending preserves oscillatory components."""
        raw = sample_raw_data(n_channels=1, signal_type='sine')
        detrended = SignalProcessor.detrend_signal(raw)
        
        # Shape should be preserved
        assert detrended.get_data().shape == raw.get_data().shape
        
        # Oscillations should still exist (std > 0)
        assert np.std(detrended.get_data()) > 0


class TestICA:
    """Tests for Independent Component Analysis."""
    
    def test_ica_basic_application(self, sample_raw_data):
        """Test basic ICA application."""
        # Need at least 2 channels for ICA
        raw = sample_raw_data(n_channels=5, n_samples=1000, signal_type='noise')
        cleaned = SignalProcessor.apply_ica(raw, n_components=3)
        
        assert cleaned is not None
        assert cleaned.get_data().shape == raw.get_data().shape
    
    def test_ica_insufficient_channels(self, sample_raw_data):
        """Test ICA with insufficient channels (should return original)."""
        raw = sample_raw_data(n_channels=1, signal_type='noise')
        result = SignalProcessor.apply_ica(raw, n_components=5)
        
        # Should return original since ICA needs >= 2 components
        assert np.allclose(result.get_data(), raw.get_data())
    
    def test_ica_with_different_components(self, sample_raw_data):
        """Test ICA with different number of components."""
        raw = sample_raw_data(n_channels=10, n_samples=2000, signal_type='noise')
        
        cleaned_3 = SignalProcessor.apply_ica(raw, n_components=3)
        cleaned_5 = SignalProcessor.apply_ica(raw, n_components=5)
        
        assert cleaned_3 is not None
        assert cleaned_5 is not None
    
    def test_ica_removes_first_component(self, sample_raw_data):
        """Test that ICA excludes the first component."""
        raw = sample_raw_data(n_channels=5, n_samples=1000, signal_type='composite')
        cleaned = SignalProcessor.apply_ica(raw, n_components=4)
        
        # Data should be modified (first component removed)
        assert not np.allclose(cleaned.get_data(), raw.get_data())


class TestBandPowerExtraction:
    """Tests for EEG band power extraction."""
    
    def test_band_powers_basic(self, sample_signals):
        """Test basic band power extraction."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        powers = SignalProcessor.extract_band_powers(data, sfreq)
        
        # Should return dict with all bands
        assert 'Alpha (8.0-13.0Hz)' in powers
        assert 'Beta (13.0-30.0Hz)' in powers
        assert 'Gamma (30.0-100.0Hz)' in powers
        assert 'Theta (4.0-8.0Hz)' in powers
        assert 'Delta (0.5-4.0Hz)' in powers
    
    def test_band_powers_sum_to_one(self, sample_signals):
        """Test that relative band powers sum to approximately 1."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        powers = SignalProcessor.extract_band_powers(data, sfreq)
        total = sum(powers.values())
        
        assert np.isclose(total, 1.0, atol=0.01)
    
    def test_band_powers_alpha_dominant(self, sample_signals):
        """Test that 10Hz signal has dominant alpha power."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        powers = SignalProcessor.extract_band_powers(data, sfreq)
        
        # Alpha band should have the most power for 10Hz signal
        alpha_power = powers['Alpha (8.0-13.0Hz)']
        assert alpha_power == max(powers.values())
    
    def test_band_powers_flat_signal(self, sample_signals):
        """Test band power extraction on flat signal."""
        data = sample_signals['flat']
        sfreq = sample_signals['sfreq']
        
        powers = SignalProcessor.extract_band_powers(data, sfreq)
        
        # All powers should be 0 for flat signal
        assert all(p == 0.0 for p in powers.values())
    
    def test_band_powers_noise_signal(self, sample_signals):
        """Test band power extraction on noise signal."""
        data = sample_signals['noise']
        sfreq = sample_signals['sfreq']
        
        powers = SignalProcessor.extract_band_powers(data, sfreq)
        
        # Powers should be distributed across bands for noise
        assert all(p >= 0 for p in powers.values())


class TestPeakDetection:
    """Tests for peak detection in signals."""
    
    def test_peak_detection_basic(self, sample_signals):
        """Test basic peak detection."""
        data = sample_signals['peaks']
        
        peaks, properties = SignalProcessor.detect_peaks(data, height=5.0)
        
        assert len(peaks) > 0
        assert len(peaks) <= 2  # Should detect the two major peaks
    
    def test_peak_detection_with_height_threshold(self):
        """Test peak detection with height threshold."""
        data = np.array([0, 1, 0, 5, 0, 3, 0, 8, 0])
        
        # Only peaks >= 5
        peaks, _ = SignalProcessor.detect_peaks(data, height=5.0)
        
        assert len(peaks) == 2  # Indices 3 and 7
        assert 3 in peaks
        assert 7 in peaks
    
    def test_peak_detection_with_distance(self):
        """Test peak detection with minimum distance constraint."""
        data = np.array([0, 5, 0, 5, 0, 5, 0])
        
        # Require peaks to be at least 3 samples apart
        peaks, _ = SignalProcessor.detect_peaks(data, height=4.0, distance=3)
        
        # Should detect fewer peaks due to distance constraint
        assert len(peaks) <= 3
    
    def test_peak_detection_no_peaks(self, sample_signals):
        """Test peak detection on signal with no peaks."""
        data = sample_signals['flat']
        
        peaks, _ = SignalProcessor.detect_peaks(data)
        
        assert len(peaks) == 0
    
    def test_peak_detection_sine_wave(self, sample_signals):
        """Test peak detection on sine wave."""
        data = sample_signals['sine_10hz']
        
        peaks, _ = SignalProcessor.detect_peaks(data)
        
        # Should detect peaks at the maxima of sine wave
        assert len(peaks) > 0
    
    def test_peak_detection_empty_array(self):
        """Test peak detection on empty array."""
        data = np.array([])
        
        peaks, _ = SignalProcessor.detect_peaks(data)
        
        assert len(peaks) == 0
    
    def test_peak_detection_single_value(self):
        """Test peak detection on single-value array."""
        data = np.array([5.0])
        
        peaks, _ = SignalProcessor.detect_peaks(data)
        
        # Single value cannot be a peak
        assert len(peaks) == 0
    
    def test_peak_detection_constant_signal(self):
        """Test peak detection on constant signal."""
        data = np.ones(100)
        
        peaks, _ = SignalProcessor.detect_peaks(data)
        
        # Constant signal has no peaks
        assert len(peaks) == 0
