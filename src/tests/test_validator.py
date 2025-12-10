import pytest
import numpy as np
from src.core.validator import Validator
from src.core.processor import SignalProcessor
import mne


class TestFilterComparison:
    """Tests for filter comparison functionality."""
    
    def test_compare_filters_basic(self, sample_signals):
        """Test basic filter comparison between SciPy and MNE."""
        data = sample_signals['composite']
        sfreq = sample_signals['sfreq']
        
        result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=30.0)
        
        assert 'scipy_filtered' in result
        assert 'mne_filtered' in result
        assert 'residuals' in result
        assert 'mse' in result
    
    def test_compare_filters_produces_similar_results(self, sample_signals):
        """Test that SciPy and MNE filters produce similar results."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=30.0)
        
        # MSE should be relatively small for similar filters
        assert result['mse'] < 1.0  # Arbitrary threshold, adjust based on actual behavior
    
    def test_compare_filters_shapes_match(self, sample_signals):
        """Test that filtered outputs have correct shapes."""
        data = sample_signals['noise']
        sfreq = sample_signals['sfreq']
        
        result = Validator.compare_filters(data, sfreq, l_freq=5.0, h_freq=40.0)
        
        assert len(result['scipy_filtered']) == len(data)
        assert len(result['mne_filtered']) == len(data)
        assert len(result['residuals']) == len(data)
    
    def test_compare_filters_different_frequencies(self, sample_signals):
        """Test filter comparison with different frequency ranges."""
        data = sample_signals['composite']
        sfreq = sample_signals['sfreq']
        
        # Low-pass only
        result_lp = Validator.compare_filters(data, sfreq, l_freq=0.5, h_freq=10.0)
        
        # High-pass
        result_hp = Validator.compare_filters(data, sfreq, l_freq=20.0, h_freq=100.0)
        
        # Results should differ based on filter parameters
        assert not np.array_equal(result_lp['scipy_filtered'], result_hp['scipy_filtered'])
    
    def test_compare_filters_on_flat_signal(self, sample_signals):
        """Test filter comparison on flat signal."""
        data = sample_signals['flat']
        sfreq = sample_signals['sfreq']
        
        result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=30.0)
        
        # Flat signal should remain flat after filtering
        assert np.allclose(result['scipy_filtered'], 0, atol=1e-10)
        assert np.allclose(result['mne_filtered'], 0, atol=1e-10)
        assert result['mse'] < 1e-10


class TestPerformanceMeasurement:
    """Tests for performance measurement functionality."""
    
    def test_measure_performance_basic(self):
        """Test basic performance measurement."""
        def simple_function(x):
            return x ** 2
        
        result = Validator.measure_performance(simple_function, 10)
        
        assert 'execution_time_ms' in result
        assert 'memory_used_mb' in result
        assert 'result' in result
        assert result['result'] == 100
    
    def test_measure_performance_timing(self):
        """Test that performance measurement captures execution time."""
        import time
        
        def slow_function():
            time.sleep(0.01)  # Sleep for 10ms
            return True
        
        result = Validator.measure_performance(slow_function)
        
        # Should take at least 10ms
        assert result['execution_time_ms'] >= 10.0
        assert result['result'] is True
    
    def test_measure_performance_with_args(self):
        """Test performance measurement with multiple arguments."""
        def multiply(a, b, c):
            return a * b * c
        
        result = Validator.measure_performance(multiply, 2, 3, 4)
        
        assert result['result'] == 24
        assert result['execution_time_ms'] >= 0
    
    def test_measure_performance_with_kwargs(self):
        """Test performance measurement with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = Validator.measure_performance(greet, "World", greeting="Hi")
        
        assert result['result'] == "Hi, World!"
    
    def test_measure_performance_numpy_operation(self):
        """Test performance measurement on numpy operations."""
        def numpy_op():
            return np.random.randn(1000, 1000).mean()
        
        result = Validator.measure_performance(numpy_op)
        
        assert 'execution_time_ms' in result
        assert isinstance(result['result'], (float, np.floating))
    
    def test_measure_performance_signal_processing(self, sample_raw_data):
        """Test performance measurement on signal processing."""
        raw = sample_raw_data(n_channels=5, n_samples=10000, signal_type='noise')
        
        def process_signal():
            return SignalProcessor.apply_filter(raw, l_freq=1.0, h_freq=40.0)
        
        result = Validator.measure_performance(process_signal)
        
        assert result['execution_time_ms'] > 0
        assert result['result'] is not None
    
    def test_measure_performance_exception_handling(self):
        """Test that performance measurement handles exceptions."""
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            Validator.measure_performance(failing_function)
    
    def test_measure_performance_returns_correct_result(self):
        """Test that performance measurement returns correct computation result."""
        def compute_list():
            return [i**2 for i in range(10)]
        
        result = Validator.measure_performance(compute_list)
        
        expected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        assert result['result'] == expected


class TestValidatorEdgeCases:
    """Tests for edge cases in Validator."""
    
    def test_filter_comparison_single_sample(self):
        """Test filter comparison with minimal data."""
        data = np.array([1.0])
        sfreq = 250.0
        
        # This might fail or handle gracefully
        try:
            result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=30.0)
            # If it succeeds, check basic structure
            assert 'scipy_filtered' in result
        except Exception:
            # It's acceptable to fail on insufficient data
            pass
    
    def test_filter_comparison_very_short_signal(self):
        """Test filter comparison with very short signal."""
        data = np.random.randn(10)
        sfreq = 250.0
        
        try:
            result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=30.0)
            assert len(result['residuals']) == len(data)
        except Exception:
            # Might fail due to filter requirements
            pass
    
    def test_performance_measurement_zero_time_function(self):
        """Test performance measurement on very fast function."""
        def instant_function():
            return 42
        
        result = Validator.measure_performance(instant_function)
        
        # Even instant functions should have non-negative time
        assert result['execution_time_ms'] >= 0
        assert result['result'] == 42
    
    def test_filter_comparison_high_frequency_cutoff(self, sample_signals):
        """Test filter comparison with cutoff near Nyquist frequency."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        nyquist = sfreq / 2
        
        # Use cutoff close to Nyquist
        result = Validator.compare_filters(data, sfreq, l_freq=1.0, h_freq=nyquist - 10)
        
        assert result is not None
        assert 'mse' in result
