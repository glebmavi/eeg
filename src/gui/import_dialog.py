from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLabel, QComboBox, QDialogButtonBox, QHBoxLayout, QMessageBox)
import pandas as pd
import numpy as np

class ImportDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Data Configuration")
        self.resize(600, 400)
        self.file_path = file_path
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Preview
        self.preview_table = QTableWidget()
        layout.addWidget(QLabel("Data Preview (First 5 rows):"))
        layout.addWidget(self.preview_table)
        
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
        
        layout.addLayout(config_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.df = None
        self._load_preview()

    def _load_preview(self):
        try:
            if self.file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(self.file_path, nrows=5)
            elif self.file_path.lower().endswith('.edf'):
                # For EDF we can't really preview rows easily without MNE logic, 
                # but EDF usually has units defined. We assume this dialog is mainly for CSV or ambiguous sources.
                # If EDF, we might skip this or just show minimal info.
                return 
                
            if self.df is not None:
                self._populate_table()
                self._populate_time_combo()
                self._apply_heuristics()
                
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _populate_table(self):
        self.preview_table.setColumnCount(len(self.df.columns))
        self.preview_table.setRowCount(len(self.df))
        self.preview_table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iloc[i, j]))
                self.preview_table.setItem(i, j, item)

    def _populate_time_combo(self):
        self.time_col_combo.addItems(list(self.df.columns))

    def _apply_heuristics(self):
        # 1. Time Column Detection
        time_candidates = ['time', 'sec', 't', 'timestamp']
        for col in self.df.columns:
            if any(cand in col.lower() for cand in time_candidates):
                self.time_col_combo.setCurrentText(col)
                break
                
        # 2. Unit Detection
        # Check numerical columns (exclude time if possible)
        # Flatten sample data
        try:
            sample_data = self.df.select_dtypes(include=[np.number]).values.flatten()
            if len(sample_data) == 0: return
            
            avg_val = np.mean(np.abs(sample_data))
            max_val = np.max(np.abs(sample_data))
            
            # Heuristics from plan:
            # < 0.01 -> Volts
            # > 500 -> Raw/ADC
            # Else -> Microvolts
            
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
        scaling = 1.0
        if unit == "Volts":
            scaling = 1e6
        elif unit == "Raw/ADC":
            scaling = 1.0 # No scaling or special handling?? 
            # Actually user said: "The axis values... seem to be too high... because removing 1e6..."
            # So if Raw, we probably just want to display as is. 
            # But MNE stores as Volts. If we input 1024 as volts, it's 1024V. 
            # If we want 1024 to be 1024 uV, we treat as uV. 
            # If we want 1024 raw units, we treat as generic. MNE has 'misc' type, or we just trust the value.
            # Currently system assumes Volts and multiplies by 1e6 for plot.
            # If unit is Microvolts, we load as 1e-6 Volts into MNE.
            
            # Let's clarify:
            # Loader currently: data = df.values.T -> RawArray(data). MNE treats data as VOLTS by default.
            # PlotWidget currently: data * 1e6.
            
            # Case 1: Input is Volts (e.g. 0.0001). MNE gets 0.0001. Plot gets 100. Correct.
            # Case 2: Input is uV (e.g. 100). MNE gets 100. Plot gets 100,000,000. WRONG.
            # Fix: If uV, we must convert to Volts for MNE (divide by 1e6).
            
            # Case 3: Input is Raw (e.g. 512). MNE gets 512. Plot gets 512,000,000. WRONG.
            # Fix: If Raw, we probably want to visualize 512. So we need MNE to store 512e-6 (if we treat as uV) OR 512 (if we treat as V but Plotting doesn't scale).
            # BUT PlotWidget is hardcoded to * 1e6.
            # SO: We need to remove hardcoded scaling in PlotWidget AND return a 'plot_scaling' factor from here or store in data.
            pass
            
        return time_col, unit
