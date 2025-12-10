import time
import psutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QCheckBox,
                             QLabel, QDoubleSpinBox, QPushButton, QFormLayout, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal, QTimer
from src.models.types import FilterState, AnalysisState, RhythmBands


class ControlPanel(QWidget):
    # Filter & View Signals
    filter_applied = pyqtSignal(dict)
    
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

        self._init_system_monitor()
        self._init_filter_controls()
        self._init_analysis_controls()
        self._init_advanced_controls()

        self.layout.addStretch()

        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_stats)
        self.timer.start(1000)

    def _init_system_monitor(self):
        """Initialize real-time system resource monitoring panel."""
        monitor_group = QGroupBox("System Status")
        monitor_layout = QFormLayout()

        self.lbl_memory = QLabel("Memory: Calculating...")
        self.lbl_uptime = QLabel("Uptime: 0s")

        monitor_layout.addRow(self.lbl_memory)
        monitor_layout.addRow(self.lbl_uptime)

        monitor_group.setLayout(monitor_layout)
        self.layout.addWidget(monitor_group)

    def _init_filter_controls(self):
        """Initialize signal preprocessing controls (notch, detrend, bandpass)."""
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
        """Initialize rhythm band visualization toggles (delta, theta, alpha, beta, gamma)."""
        analysis_group = QGroupBox("Rhythm Visualization")
        analysis_layout = QVBoxLayout()

        b = RhythmBands

        self.delta_cb = QCheckBox(f"Delta ({b.DELTA.low}-{b.DELTA.high} Hz)")
        self.delta_cb.toggled.connect(self.delta_toggled.emit)
        
        self.theta_cb = QCheckBox(f"Theta ({b.THETA.low}-{b.THETA.high} Hz)")
        self.theta_cb.toggled.connect(self.theta_toggled.emit)
        
        self.alpha_cb = QCheckBox(f"Alpha ({b.ALPHA.low}-{b.ALPHA.high} Hz)")
        self.alpha_cb.toggled.connect(self.alpha_toggled.emit)
        
        self.beta_cb = QCheckBox(f"Beta ({b.BETA.low}-{b.BETA.high} Hz)")
        self.beta_cb.toggled.connect(self.beta_toggled.emit)
        
        self.gamma_cb = QCheckBox(f"Gamma ({b.GAMMA.low}-{b.GAMMA.high} Hz)")
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
        """Initialize advanced processing controls (ICA, feature extraction)."""
        adv_group = QGroupBox("Advanced Analysis & Artifacts")
        adv_layout = QVBoxLayout()

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
        """Update memory usage and uptime displays."""
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        self.lbl_memory.setText(f"Memory: {mem_mb:.1f} MB")

        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.lbl_uptime.setText(f"Uptime: {hours:02}:{minutes:02}:{seconds:02}")


    def update_ui_state(self, filter_state: FilterState, analysis_state: AnalysisState):
        """Synchronize UI controls with active view state, blocking signals to prevent loops."""
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
