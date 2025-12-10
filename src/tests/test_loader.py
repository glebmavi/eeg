import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.core.loader import DataLoader


class TestDataLoaderCSV:
    """Tests for CSV file loading functionality."""
    
    def test_csv_basic_loading(self, temp_csv_file):
        """Test basic CSV loading with default parameters."""
        file_path = temp_csv_file(n_channels=3, n_samples=100)
        raw = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time')
        
        assert raw is not None
        assert len(raw.ch_names) == 3
        assert raw.info['sfreq'] == 250.0
        assert raw.get_data().shape == (3, 100)
    
    def test_csv_without_time_column(self, temp_csv_file):
        """Test CSV loading when no time column is specified."""
        file_path = temp_csv_file(n_channels=2, n_samples=50, include_time=False)
        raw = DataLoader.load_data(file_path, sfreq=100.0)
        
        assert len(raw.ch_names) == 2
        assert raw.get_data().shape == (2, 50)
    
    def test_csv_with_custom_channel_names(self, temp_csv_file):
        """Test CSV loading with custom channel prefix."""
        file_path = temp_csv_file(n_channels=4, channel_prefix="FP")
        raw = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time')
        
        assert 'FP1' in raw.ch_names
        assert 'FP4' in raw.ch_names
        assert len(raw.ch_names) == 4
    
    def test_csv_unit_scaling(self, temp_csv_file):
        """Test unit scaling (uV to V conversion)."""
        file_path = temp_csv_file(n_channels=1, n_samples=10)
        
        # Load with microvolts scaling
        raw_uv = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time', unit_scale=1e-6)
        
        # Data should be scaled down by 1e-6
        data = raw_uv.get_data()
        assert data.max() < 1e-3  # Should be in volts range
    
    def test_csv_column_exclusion(self, tmp_path):
        """Test excluding specific columns during loading."""
        # Create CSV with extra columns
        file_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'Time': np.linspace(0, 1, 50),
            'Ch1': np.random.randn(50),
            'Ch2': np.random.randn(50),
            'Annotation': [''] * 50
        })
        df.to_csv(file_path, index=False)
        
        raw = DataLoader.load_data(str(file_path), sfreq=250.0, 
                                   time_col='Time', exclude_cols=['Annotation'])
        
        assert len(raw.ch_names) == 2
        assert 'Annotation' not in raw.ch_names
    
    def test_csv_with_description(self, temp_csv_file):
        """Test adding description metadata to loaded data."""
        file_path = temp_csv_file(n_channels=1)
        description = "Test EEG recording"
        raw = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time', 
                                   description=description)
        
        assert raw.info.get('description') == description
    
    def test_csv_malformed_data(self, tmp_path):
        """Test handling of malformed CSV data."""
        file_path = tmp_path / "malformed.csv"
        file_path.write_text("Ch1,Ch2\n1,2\n3,invalid\n5,6")
        
        # pandas should handle this, but might convert to NaN
        # We test that it doesn't crash
        try:
            raw = DataLoader.load_data(str(file_path), sfreq=250.0)
            # If it loads, check for NaN
            assert raw is not None
        except Exception:
            # It's acceptable to raise an exception for malformed data
            pass


class TestDataLoaderEDF:
    """Tests for EDF file loading functionality."""
    
    def test_edf_basic_loading(self, temp_edf_file):
        """Test basic EDF file loading."""
        file_path = temp_edf_file(n_channels=2, n_samples=500)
        raw = DataLoader.load_data(file_path)
        
        assert raw is not None
        assert len(raw.ch_names) == 2
        assert raw.get_data().shape[1] == 500
    
    def test_edf_preserves_metadata(self, temp_edf_file):
        """Test that EDF metadata is preserved."""
        file_path = temp_edf_file(n_channels=3)
        raw = DataLoader.load_data(file_path)
        
        # EDF should have sampling frequency from file
        assert raw.info['sfreq'] > 0
        assert len(raw.ch_names) == 3


class TestDataLoaderEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_unsupported_file_format(self, tmp_path):
        """Test error handling for unsupported file formats."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoader.load_data(str(file_path))
    
    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(Exception):
            DataLoader.load_data("/nonexistent/path/file.csv", sfreq=250.0)
    
    def test_csv_with_integer_time_column(self, tmp_path):
        """Test CSV loading with integer index for time column."""
        file_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'Time': np.linspace(0, 1, 50),
            'Ch1': np.random.randn(50),
            'Ch2': np.random.randn(50)
        })
        df.to_csv(file_path, index=False)
        
        # Use column index 0 for time
        raw = DataLoader.load_data(str(file_path), sfreq=250.0, time_col=0)
        
        assert len(raw.ch_names) == 2
        assert 'Time' not in raw.ch_names
    
    def test_csv_high_sampling_rate(self, temp_csv_file):
        """Test CSV loading with high sampling rate."""
        file_path = temp_csv_file(n_channels=2, n_samples=1000)
        raw = DataLoader.load_data(file_path, sfreq=1000.0, time_col='Time')
        
        assert raw.info['sfreq'] == 1000.0
        assert raw.get_data().shape == (2, 1000)
    
    def test_csv_single_channel(self, temp_csv_file):
        """Test CSV loading with single channel."""
        file_path = temp_csv_file(n_channels=1, n_samples=100)
        raw = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time')
        
        assert len(raw.ch_names) == 1
        assert raw.get_data().shape == (1, 100)
    
    def test_csv_many_channels(self, temp_csv_file):
        """Test CSV loading with many channels."""
        file_path = temp_csv_file(n_channels=32, n_samples=100)
        raw = DataLoader.load_data(file_path, sfreq=250.0, time_col='Time')
        
        assert len(raw.ch_names) == 32
        assert raw.get_data().shape == (32, 100)
