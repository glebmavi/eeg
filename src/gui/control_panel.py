from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QCheckBox, 
                               QLabel, QDoubleSpinBox, QPushButton, QFormLayout)
from PyQt6.QtCore import pyqtSignal

class ControlPanel(QWidget):
    # Signal emitted when "Apply" is clicked
    filter_applied = pyqtSignal(dict)
    # Signal emitted when "Load Data" is clicked
    load_clicked = pyqtSignal()
    # Signal emitted when Theme is toggled (True=Dark, False=Light)
    theme_toggled = pyqtSignal(bool)
    
    # Analysis Signals
    alpha_toggled = pyqtSignal(bool)
    beta_toggled = pyqtSignal(bool)
    gamma_toggled = pyqtSignal(bool)
    theta_toggled = pyqtSignal(bool)
    delta_toggled = pyqtSignal(bool)
    
    peaks_toggled = pyqtSignal(bool)
    
    # Validation Signal
    validation_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # --- System Controls ---
        sys_group = QGroupBox("System")
        sys_layout = QFormLayout()
        
        self.theme_cb = QCheckBox("Dark Theme")
        self.theme_cb.setChecked(True) # Default to dark
        self.theme_cb.toggled.connect(self.theme_toggled.emit)
        
        sys_layout.addRow(self.theme_cb)
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)

        # --- Filter Group ---
        filter_group = QGroupBox("Signal Processing")
        filter_layout = QFormLayout()
        
        self.notch_cb = QCheckBox("Notch Filter (50 Hz)")
        self.notch_cb.toggled.connect(self.emit_filter_settings)
        
        self.detrend_cb = QCheckBox("Detrend (Linear)")
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
        
        # apply_btn removed for reactive UI
        # self.apply_btn = QPushButton("Apply Filters to Active View")
        # self.apply_btn.clicked.connect(self.on_apply)
        # filter_layout.addRow(self.apply_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # --- Analysis Group ---
        analysis_group = QGroupBox("Interactive Analysis")
        analysis_layout = QVBoxLayout()
        
        self.delta_cb = QCheckBox("Show Delta (0.5-4 Hz)")
        self.delta_cb.toggled.connect(self.delta_toggled.emit)
        
        self.theta_cb = QCheckBox("Show Theta (4-8 Hz)")
        self.theta_cb.toggled.connect(self.theta_toggled.emit)
        
        self.alpha_cb = QCheckBox("Show Alpha (8-13 Hz)")
        self.alpha_cb.toggled.connect(self.alpha_toggled.emit)
        
        self.beta_cb = QCheckBox("Show Beta (13-30 Hz)")
        self.beta_cb.toggled.connect(self.beta_toggled.emit)
        
        self.gamma_cb = QCheckBox("Show Gamma (30-100 Hz)")
        self.gamma_cb.toggled.connect(self.gamma_toggled.emit)
        
        self.peaks_cb = QCheckBox("Show Peaks")
        self.peaks_cb.toggled.connect(self.peaks_toggled.emit)
        
        analysis_layout.addWidget(self.delta_cb)
        analysis_layout.addWidget(self.theta_cb)
        analysis_layout.addWidget(self.alpha_cb)
        analysis_layout.addWidget(self.beta_cb)
        analysis_layout.addWidget(self.gamma_cb)
        analysis_layout.addWidget(self.peaks_cb)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # --- Data Controls ---
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Data File...")
        self.load_btn.clicked.connect(self.load_clicked.emit)
        data_layout.addWidget(self.load_btn)
        data_group.setLayout(data_layout)
        
        layout.addWidget(data_group)
        
        # --- Validation ---
        self.val_btn = QPushButton("System Status & Validation")
        self.val_btn.clicked.connect(self.validation_clicked.emit)
        layout.addWidget(self.val_btn)
        
        layout.addStretch()

    def emit_filter_settings(self):
        params = {
            "notch": self.notch_cb.isChecked(),
            "detrend": self.detrend_cb.isChecked(),
            "l_freq": self.l_freq_spin.value(),
            "h_freq": self.h_freq_spin.value()
        }
        self.filter_applied.emit(params)
    
    # Alias for legacy or just to keep logic
    on_apply = emit_filter_settings
