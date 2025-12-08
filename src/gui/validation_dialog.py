from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMessageBox
import pyqtgraph as pg
import numpy as np
from src.core.validator import Validator
import mne

class ValidationDialog(QDialog):
    def __init__(self, raw_data: np.ndarray = None, sfreq: float = 250.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Validation & Benchmark")
        self.resize(800, 600)
        
        self.raw_data = raw_data
        self.sfreq = sfreq
        if self.raw_data is None:
             self.sfreq = 250.0 # Default if generating
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Info
        layout.addWidget(QLabel("Validates MNE Filter (IIR) against Manual SciPy Implementation (Butterworth)."))
        if self.raw_data is None:
            layout.addWidget(QLabel("NOTE: No signal loaded. Using generated synthetic noise."))
        
        # Controls
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Validation Benchmark")
        self.btn_run.clicked.connect(self.run_validation)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)
        
        # Metrics
        self.lbl_mse = QLabel("MSE (Accuracy): N/A")
        self.lbl_time = QLabel("Execution Time (MNE Filter): N/A")
        self.lbl_ram = QLabel("Memory Usage Change: N/A")
        
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.lbl_mse)
        metrics_layout.addWidget(self.lbl_time)
        metrics_layout.addWidget(self.lbl_ram)
        layout.addLayout(metrics_layout)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Filter Residuals (SciPy vs MNE)")
        self.plot_widget.setLabel('left', "Difference (uV)")
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)

    def run_validation(self):
        try:
            data_to_use = self.raw_data
            if data_to_use is None:
                # Generate synthetic data: 10 seconds of noise + sine
                t = np.linspace(0, 10, int(10 * self.sfreq))
                data_to_use = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, len(t))
            
            # Ensure 1D
            if data_to_use.ndim > 1:
                data_to_use = data_to_use[0].flatten()
                
            # 1. Compare Filters (Accuracy)
            # Filter 1-40Hz
            valid_res = Validator.compare_filters(data_to_use, self.sfreq, 1.0, 40.0)
            
            # 2. Performance Benchmark (MNE only)
            def run_mne():
                # Re-run MNE filter to measure exact cost
                # Reshape for MNE
                mne_in = data_to_use.reshape(1, -1)
                return mne.filter.filter_data(mne_in, self.sfreq, 1.0, 40.0, method='iir', verbose=False)
                
            perf = Validator.measure_performance(run_mne)
            
            # Update UI
            mse = valid_res['mse']
            self.lbl_mse.setText(f"MSE (Accuracy): {mse:.5e} (Goal: Close to 0)")
            self.lbl_time.setText(f"Execution Time (MNE Filter): {perf['execution_time_ms']:.2f} ms")
            self.lbl_ram.setText(f"Memory Usage Change: {perf['memory_used_mb']:.4f} MB")
            
            # Plot Residuals
            self.plot_widget.clear()
            self.plot_widget.plot(valid_res['residuals'], pen='r')
            
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
