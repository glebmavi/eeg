import pandas as pd
import mne
import numpy as np

class DataLoader:
    """
    Handles loading of EEG data from various formats (.csv, .edf, .set).
    Returns a unified MNE Raw object.
    """

    @staticmethod
    def load_data(file_path: str, sfreq: float = 250.0, time_col: str | int = None, 
                  unit_scale: float = 1.0, exclude_cols: list = None, description: str = None) -> mne.io.BaseRaw:
        """
        Load data from a file and return an MNE Raw object.
        ...
        description (str): Description to store in info (e.g. data unit type).
        """
        if file_path.lower().endswith('.csv'):
            return DataLoader._load_csv(file_path, sfreq, time_col, unit_scale, exclude_cols, description)
        elif file_path.lower().endswith('.edf'):
            return DataLoader._load_edf(file_path)
        elif file_path.lower().endswith('.set'):
            return DataLoader._load_set(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def _load_csv(file_path: str, sfreq: float, time_col: str | int = None, 
                  unit_scale: float = 1.0, exclude_cols: list = None, description: str = None) -> mne.io.RawArray:
        """Loads data from a CSV file."""
        df = pd.read_csv(file_path)
        
        # Handle time column
        if time_col is not None:
             if isinstance(time_col, str) and time_col in df.columns:
                 df = df.drop(columns=[time_col])
        
        # Handle exclusions
        if exclude_cols:
            for col in exclude_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
        data = df.values.T
        
        # Apply scaling to convert to Volts for MNE storage
        data = data * unit_scale
        
        ch_names = list(df.columns)
        ch_types = ['eeg'] * len(ch_names)
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        if description:
            info['description'] = description
            
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
