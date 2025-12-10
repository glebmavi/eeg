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
        """Load and display data preview with heuristics."""
        try:
            if self.file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(self.file_path, nrows=5)
            elif self.file_path.lower().endswith('.edf'):
                # TODO: For EDF we need to implement a column preview, or just a channel preview to allow user to select/mark channels.
                return 
                
            if self.df is not None:
                self._populate_table()
                self._populate_time_combo()
                self._apply_heuristics()
                
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _populate_table(self):
        """Populate preview table with dataframe contents."""
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
            
        return time_col, unit
