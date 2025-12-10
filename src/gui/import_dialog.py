from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLabel, QComboBox, QDialogButtonBox, QHBoxLayout, QMessageBox,
                             QTabWidget, QWidget, QHeaderView)
from PyQt6.QtCore import Qt
import pandas as pd
import numpy as np
import mne

class ImportDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Data Configuration")
        self.resize(800, 600)
        self.file_path = file_path
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # --- Tab 1: General Settings (Preview & Units) ---
        self.tab_general = QWidget()
        gen_layout = QVBoxLayout(self.tab_general)
        
        self.preview_table = QTableWidget()
        gen_layout.addWidget(QLabel("Data Preview (First 5 rows):"))
        gen_layout.addWidget(self.preview_table)
        
        # Heuristics & Config
        config_layout = QHBoxLayout()
        
        # Time Column
        self.time_col_combo = QComboBox()
        self.time_col_combo.addItem("Index (Auto-generated)")
        config_layout.addWidget(QLabel("Time Column:"))
        config_layout.addWidget(self.time_col_combo)
        
        # Units
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Microvolts (uV)", "Volts", "Raw/ADC"])
        config_layout.addWidget(QLabel("Input Units:"))
        config_layout.addWidget(self.unit_combo)
        
        gen_layout.addLayout(config_layout)
        self.tabs.addTab(self.tab_general, "General Settings")
        
        # --- Tab 2: Channel Configuration ---
        self.tab_channels = QWidget()
        ch_layout = QVBoxLayout(self.tab_channels)
        
        ch_layout.addWidget(QLabel("Configure Channel Types (for Artifact Removal):"))
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(2)
        self.channel_table.setHorizontalHeaderLabels(["Channel Name", "Type"])
        self.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.channel_table.verticalHeader().setDefaultSectionSize(35)
        ch_layout.addWidget(self.channel_table)
        
        self.tabs.addTab(self.tab_channels, "Channel Configuration")
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.df = None
        self.channel_names = []
        self._load_preview()

    def _load_preview(self):
        """Load and display data preview with heuristics."""
        try:
            if self.file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(self.file_path, nrows=5)
                self.channel_names = list(self.df.columns)
            elif self.file_path.lower().endswith(('.edf', '.set')):
                # Read header only
                if self.file_path.lower().endswith('.edf'):
                    raw_preview = mne.io.read_raw_edf(self.file_path, preload=False, verbose=False)
                else:
                    raw_preview = mne.io.read_raw_eeglab(self.file_path, preload=False, verbose=False)
                self.channel_names = raw_preview.ch_names
                self.df = pd.DataFrame(columns=self.channel_names) # Empty DF just for column names compatibility logic if needed
                
                # Disable CSV specific controls
                self.preview_table.setDisabled(True)
                self.time_col_combo.setDisabled(True)
                self.unit_combo.setDisabled(True)
                self.tab_general.setDisabled(False) # Keep enabled but empty/disabled inside
                
                # Show info label
                self.preview_table.setRowCount(1)
                self.preview_table.setColumnCount(1)
                self.preview_table.setItem(0,0, QTableWidgetItem("Binary file (EDF/SET). Preview not available in table."))
                self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

            # Populate Settings
            if self.file_path.lower().endswith('.csv') and self.df is not None:
                self._populate_table()
                self._populate_time_combo()
                self._apply_heuristics()
            
            # Populate Channels Tab
            self._populate_channel_config()
                
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _populate_table(self):
        """Populate preview table with dataframe contents."""
        self.preview_table.setEnabled(True)
        self.preview_table.setColumnCount(len(self.df.columns))
        self.preview_table.setRowCount(len(self.df))
        self.preview_table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iloc[i, j]))
                self.preview_table.setItem(i, j, item)

    def _populate_time_combo(self):
        self.time_col_combo.clear()
        self.time_col_combo.addItem("Index (Auto-generated)")
        self.time_col_combo.addItems(list(self.df.columns))

    def _populate_channel_config(self):
        self.channel_table.setRowCount(len(self.channel_names))
        
        common_types = ["eeg", "eog", "ecg", "emg", "stim", "misc"]
        
        for i, name in enumerate(self.channel_names):
            # Name
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            self.channel_table.setItem(i, 0, item_name)
            
            # Type Combo
            combo = QComboBox()
            combo.addItems(common_types)
            
            # Heuristic for type
            lower_name = name.lower()
            if 'eog' in lower_name or 'eye' in lower_name:
                combo.setCurrentText('eog')
            elif 'ecg' in lower_name or 'ekg' in lower_name:
                combo.setCurrentText('ecg')
            elif 'emg' in lower_name:
                combo.setCurrentText('emg')
            else:
                combo.setCurrentText('eeg')
                
            self.channel_table.setCellWidget(i, 1, combo)

    def _apply_heuristics(self):
        """Auto-detect time column and data units based on content."""
        time_candidates = ['time', 'sec', 't', 'timestamp']
        for col in self.df.columns:
            if any(cand in col.lower() for cand in time_candidates):
                self.time_col_combo.setCurrentText(col)
                break
                
        try:
            sample_data = self.df.select_dtypes(include=[np.number]).values.flatten()
            if len(sample_data) == 0: return
            
            max_val = np.max(np.abs(sample_data))
            
            if max_val < 0.01:
                self.unit_combo.setCurrentText("Volts")
            elif max_val > 500:
                self.unit_combo.setCurrentText("Raw/ADC")
            else:
                self.unit_combo.setCurrentText("Microvolts (uV)")
        except:
            pass
            
    def get_settings(self):
        time_col = self.time_col_combo.currentText()
        if time_col == "Index (Auto-generated)":
            time_col = None
            
        unit = self.unit_combo.currentText()
        
        # Collect Channel Types
        channel_types = {}
        for i in range(self.channel_table.rowCount()):
            name_item = self.channel_table.item(i, 0)
            if not name_item: continue
            name = name_item.text()
            
            combo = self.channel_table.cellWidget(i, 1)
            ctype = combo.currentText()
            channel_types[name] = ctype
            
        return time_col, unit, channel_types
