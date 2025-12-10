import time
import psutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QCheckBox,
                             QLabel, QDoubleSpinBox, QPushButton, QFormLayout, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal, QTimer
from src.models.types import FilterState, AnalysisState
from src.core.validator import Validator


class ControlPanel(QWidget):
    # Filter & View Signals
    filter_applied = pyqtSignal(dict)
    load_clicked = pyqtSignal()
    theme_toggled = pyqtSignal(bool)

    # Analysis Toggles
    alpha_toggled = pyqtSignal(bool)
    beta_toggled = pyqtSignal(bool)
    gamma_toggled = pyqtSignal(bool)
    theta_toggled = pyqtSignal(bool)
    delta_toggled = pyqtSignal(bool)
    peaks_toggled = pyqtSignal(bool)

    # Advanced Processing Signals
    ica_requested = pyqtSignal()
    features_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- System Monitor ---
        self._init_system_monitor()

        # --- Filter Group ---
        self._init_filter_controls()

        # --- Analysis Group ---
        self._init_analysis_controls()

        # --- Advanced Processing (Artifacts & Features) ---
        self._init_advanced_controls()

        # --- Data Controls ---
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Data File...")
        self.load_btn.clicked.connect(self.load_clicked.emit)
        data_layout.addWidget(self.load_btn)
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

        self.layout.addStretch()

        # Timer for real-time stats
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_stats)
        self.timer.start(1000)  # Update every second

    def _init_system_monitor(self):
        monitor_group = QGroupBox("System Status")
        monitor_layout = QFormLayout()

        self.lbl_memory = QLabel("Memory: Calculating...")
        self.lbl_uptime = QLabel("Uptime: 0s")
        self.lbl_mse = QLabel("Filter Accuracy (MSE): N/A")

        self.btn_benchmark = QPushButton("Run Filter Benchmark")
        self.btn_benchmark.clicked.connect(self.run_benchmark)

        monitor_layout.addRow(self.lbl_memory)
        monitor_layout.addRow(self.lbl_uptime)
        monitor_layout.addRow(self.lbl_mse)
        monitor_layout.addRow(self.btn_benchmark)

        monitor_group.setLayout(monitor_layout)
        self.layout.addWidget(monitor_group)

    def _init_filter_controls(self):
        filter_group = QGroupBox("Signal Pre-processing")
        filter_layout = QFormLayout()

        self.notch_cb = QCheckBox("Notch (50Hz - Line Noise)")
        self.notch_cb.toggled.connect(self.emit_filter_settings)

        self.detrend_cb = QCheckBox("Detrend (Remove Drift)")
        self.detrend_cb.toggled.connect(self.emit_filter_settings)

        self.l_freq_spin = QDoubleSpinBox()
        self.l_freq_spin.setRange(0.1, 100.0)
        self.l_freq_spin.setValue(1.0)
        self.l_freq_spin.valueChanged.connect(self.emit_filter_settings)

        self.h_freq_spin = QDoubleSpinBox()
        self.h_freq_spin.setRange(0.5, 200.0)
        self.h_freq_spin.setValue(40.0)
        self.h_freq_spin.valueChanged.connect(self.emit_filter_settings)

        filter_layout.addRow(self.notch_cb)
        filter_layout.addRow(self.detrend_cb)
        filter_layout.addRow("Low Cut (Hz):", self.l_freq_spin)
        filter_layout.addRow("High Cut (Hz):", self.h_freq_spin)

        filter_group.setLayout(filter_layout)
        self.layout.addWidget(filter_group)

    def _init_analysis_controls(self):
        analysis_group = QGroupBox("Rhythm Visualization")
        analysis_layout = QVBoxLayout()

        # TODO: use models.types.py for power bands
        self.delta_cb = QCheckBox("Delta (0.5-4 Hz)")
        self.delta_cb.toggled.connect(self.delta_toggled.emit)
        self.theta_cb = QCheckBox("Theta (4-8 Hz)")
        self.theta_cb.toggled.connect(self.theta_toggled.emit)
        self.alpha_cb = QCheckBox("Alpha (8-13 Hz)")
        self.alpha_cb.toggled.connect(self.alpha_toggled.emit)
        self.beta_cb = QCheckBox("Beta (13-30 Hz)")
        self.beta_cb.toggled.connect(self.beta_toggled.emit)
        self.gamma_cb = QCheckBox("Gamma (30-100 Hz)")
        self.gamma_cb.toggled.connect(self.gamma_toggled.emit)

        self.peaks_cb = QCheckBox("Show Detected Peaks")
        self.peaks_cb.toggled.connect(self.peaks_toggled.emit)

        analysis_layout.addWidget(self.delta_cb)
        analysis_layout.addWidget(self.theta_cb)
        analysis_layout.addWidget(self.alpha_cb)
        analysis_layout.addWidget(self.beta_cb)
        analysis_layout.addWidget(self.gamma_cb)
        analysis_layout.addWidget(self.peaks_cb)

        analysis_group.setLayout(analysis_layout)
        self.layout.addWidget(analysis_group)

    def _init_advanced_controls(self):
        adv_group = QGroupBox("Advanced Analysis & Artifacts")
        adv_layout = QVBoxLayout()

        # System Theme
        self.theme_cb = QCheckBox("Dark Mode")
        self.theme_cb.setChecked(True)
        self.theme_cb.toggled.connect(self.theme_toggled.emit)
        adv_layout.addWidget(self.theme_cb)

        # Artifact Removal
        self.btn_ica = QPushButton("Auto-Remove Artifacts (ICA)")
        self.btn_ica.setToolTip("Uses Independent Component Analysis to remove the first component (often blinks).")
        self.btn_ica.clicked.connect(self.ica_requested.emit)
        adv_layout.addWidget(self.btn_ica)

        # Features
        self.btn_features = QPushButton("Extract Signal Features")
        self.btn_features.setToolTip("Calculate Band Powers and Signal Complexity.")
        self.btn_features.clicked.connect(self.features_requested.emit)
        adv_layout.addWidget(self.btn_features)

        adv_group.setLayout(adv_layout)
        self.layout.addWidget(adv_group)

    def update_system_stats(self):
        # Memory
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        self.lbl_memory.setText(f"Memory: {mem_mb:.1f} MB")

        # Uptime
        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.lbl_uptime.setText(f"Uptime: {hours:02}:{minutes:02}:{seconds:02}")

    def run_benchmark(self):
        """Runs the validator logic in place."""
        self.btn_benchmark.setText("Running...")
        self.btn_benchmark.setEnabled(False)
        self.repaint()  # Force update

        try:
            # Run validation on synthetic data
            # 10 seconds of 250Hz noise + sine
            import numpy as np
            t = np.linspace(0, 10, 2500)
            data = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, len(t))

            # Compare Filters
            res = Validator.compare_filters(data, 250.0, 1.0, 40.0)
            mse = res['mse']

            self.lbl_mse.setText(f"Filter Accuracy (MSE): {mse:.2e}")
        except Exception as e:
            self.lbl_mse.setText("Error")
            print(e)
        finally:
            self.btn_benchmark.setText("Run Filter Benchmark")
            self.btn_benchmark.setEnabled(True)

    def update_ui_state(self, filter_state: FilterState, analysis_state: AnalysisState):
        """
        Updates the UI elements to reflect the state of the active view.
        Blocks signals to prevent triggering processing loops.
        """
        self.blockSignals(True)
        try:
            # Update Filters
            self.notch_cb.setChecked(filter_state.notch)
            self.detrend_cb.setChecked(filter_state.detrend)
            self.l_freq_spin.setValue(filter_state.l_freq)
            self.h_freq_spin.setValue(filter_state.h_freq)

            # Update Analysis
            self.delta_cb.setChecked(analysis_state.delta)
            self.theta_cb.setChecked(analysis_state.theta)
            self.alpha_cb.setChecked(analysis_state.alpha)
            self.beta_cb.setChecked(analysis_state.beta)
            self.gamma_cb.setChecked(analysis_state.gamma)
            self.peaks_cb.setChecked(analysis_state.peaks)
        finally:
            self.blockSignals(False)

    def emit_filter_settings(self):
        params = {
            "notch": self.notch_cb.isChecked(),
            "detrend": self.detrend_cb.isChecked(),
            "l_freq": self.l_freq_spin.value(),
            "h_freq": self.h_freq_spin.value()
        }
        self.filter_applied.emit(params)