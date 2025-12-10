import pandas as pd
import mne
from typing import Union


class DataLoader:
    """Handles loading of EEG data from CSV, EDF, and SET formats."""

    @staticmethod
    def load_data(file_path: str, sfreq: float = 250.0, time_col: Union[str, int, None] = None, 
                  unit_scale: float = 1.0, exclude_cols: list = None, description: str = None) -> mne.io.BaseRaw:
        """Load EEG data and return an MNE Raw object.
        
        Args:
            file_path: Path to the data file
            sfreq: Sampling frequency for CSV files
            time_col: Column name or index to exclude as time column
            unit_scale: Scaling factor (e.g., 1e-6 for uV to V conversion)
            exclude_cols: Columns to exclude from loading
            description: Optional metadata description
            
        Returns:
            Loaded MNE Raw object
        """
        file_lower = file_path.lower()
        
        if file_lower.endswith('.csv'):
            return DataLoader._load_csv(file_path, sfreq, time_col, unit_scale, exclude_cols, description)
        elif file_lower.endswith('.edf'):
            return mne.io.read_raw_edf(file_path, preload=True)
        elif file_lower.endswith('.set'):
            return mne.io.read_raw_eeglab(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def _load_csv(file_path: str, sfreq: float, time_col: Union[str, int, None] = None, 
                  unit_scale: float = 1.0, exclude_cols: list = None, description: str = None) -> mne.io.RawArray:
        """Load CSV file and convert to MNE RawArray."""
        df = pd.read_csv(file_path)
        
        # Remove time column if specified
        if isinstance(time_col, str) and time_col in df.columns:
            df = df.drop(columns=[time_col])
        elif isinstance(time_col, int) and time_col < len(df.columns):
            df = df.drop(df.columns[time_col], axis=1)

        # Remove excluded columns
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
        
        # Apply scaling
        data = df.values.T * unit_scale
        
        info = mne.create_info(ch_names=list(df.columns), sfreq=sfreq, ch_types='eeg')
        if description:
            info['description'] = description
            
        return mne.io.RawArray(data, info)
