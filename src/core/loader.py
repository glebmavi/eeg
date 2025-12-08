import pandas as pd
import mne
import numpy as np

class DataLoader:
    """
    Handles loading of EEG data from various formats (.csv, .edf, .set).
    Returns a unified MNE Raw object.
    """

    @staticmethod
    def load_data(file_path: str, sfreq: float = 250.0, time_col: str | int = None) -> mne.io.BaseRaw:
        """
        Load data from a file and return an MNE Raw object.
        
        Args:
            file_path (str): Path to the file.
            sfreq (float): Sampling frequency for CSV files (default: 250.0).
            time_col (str | int, optional): Name or index of the time column in CSV.
            
        Returns:
            mne.io.BaseRaw: The loaded raw EEG data.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        if file_path.lower().endswith('.csv'):
            return DataLoader._load_csv(file_path, sfreq, time_col)
        elif file_path.lower().endswith('.edf'):
            return DataLoader._load_edf(file_path)
        elif file_path.lower().endswith('.set'):
            return DataLoader._load_set(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def _load_csv(file_path: str, sfreq: float, time_col: str | int = None) -> mne.io.RawArray:
        """Loads data from a CSV file."""
        df = pd.read_csv(file_path)
        
        # Handle time column if specified
        if time_col is not None:
            if isinstance(time_col, int):
                # If index, get column name
                if 0 <= time_col < len(df.columns):
                    col_name = df.columns[time_col]
                    # We might store it or use it, but for MNE data we usually exclude it
                    df = df.drop(columns=[col_name])
                else:
                    raise ValueError(f"Time column index {time_col} needs to check validity.")
            elif isinstance(time_col, str):
                if time_col in df.columns:
                    df = df.drop(columns=[time_col])
                else:
                    raise ValueError(f"Time column '{time_col}' not found in CSV.")
            
        data = df.values.T  # MNE expects (n_channels, n_times)
        ch_names = list(df.columns)
        ch_types = ['eeg'] * len(ch_names)
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        return raw

    @staticmethod
    def _load_edf(file_path: str) -> mne.io.BaseRaw:
        """Loads data from an EDF file."""
        return mne.io.read_raw_edf(file_path, preload=True)

    @staticmethod
    def _load_set(file_path: str) -> mne.io.BaseRaw:
        """Loads data from an EEGLab .set file."""
        return mne.io.read_raw_eeglab(file_path, preload=True)
