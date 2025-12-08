from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QMdiArea, QFileDialog, QInputDialog, QMessageBox, QApplication)
from PyQt6.QtCore import Qt
from src.gui.control_panel import ControlPanel
from src.gui.plot_widget import PlotWidget
from src.gui.theme_manager import ThemeManager
from src.core.loader import DataLoader
from src.core.data_manager import DataManager

from src.gui.validation_dialog import ValidationDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVisor - EEG Analysis Environment")
        self.resize(1200, 800)
        
        # Central Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Side: Visualization Area (MDI or Grid)
        self.view_container = QWidget()
        self.grid_layout = QGridLayout(self.view_container)
        main_layout.addWidget(self.view_container, stretch=3)
        
        self.views = []
        self._init_views()

        # Right Side: Control Global Panel
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)
        
        # Connect signals
        self.control_panel.filter_applied.connect(self.apply_filter_to_active_view)
        self.control_panel.load_clicked.connect(self.load_data_file)
        self.control_panel.theme_toggled.connect(self.toggle_theme)
        self.control_panel.validation_clicked.connect(self.open_validation_dialog)
        
        # Analysis signals
        self.control_panel.delta_toggled.connect(lambda c: self.toggle_analysis('delta', c))
        self.control_panel.theta_toggled.connect(lambda c: self.toggle_analysis('theta', c))
        self.control_panel.alpha_toggled.connect(lambda c: self.toggle_analysis('alpha', c))
        self.control_panel.beta_toggled.connect(lambda c: self.toggle_analysis('beta', c))
        self.control_panel.gamma_toggled.connect(lambda c: self.toggle_analysis('gamma', c))
        self.control_panel.peaks_toggled.connect(lambda c: self.toggle_analysis('peaks', c))

    def toggle_analysis(self, feature, enabled):
        if self.active_view_index is not None:
            view = self.views[self.active_view_index]
            if feature == 'peaks':
                view.toggle_peaks(enabled)
            else:
                view.toggle_rhythm(feature, enabled)

    def open_validation_dialog(self):
        data = None
        sfreq = 250.0
        
        # Try to use active view data
        if self.active_view_index is not None:
             view = self.views[self.active_view_index]
             if view.processed_data is not None:
                 data = view.processed_data.get_data()
                 sfreq = view.processed_data.info['sfreq']
                 
        dialog = ValidationDialog(data, sfreq, self)
        dialog.exec()

    def _init_views(self):
        """Initialize 4 plot widgets in a 2x2 grid."""
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, pos in enumerate(positions):
            plot = PlotWidget(index=i)
            self.grid_layout.addWidget(plot, *pos)
            self.views.append(plot)
            
            # Click handling to set active view
            plot.clicked.connect(self.set_active_view)

        # Set first one active by default
        if self.views:
            self.set_active_view(0)

    def set_active_view(self, index):
        for i, view in enumerate(self.views):
            view.set_active(i == index)
        self.active_view_index = index

    def apply_filter_to_active_view(self, filter_params):
        """
        Slot to handle filter application.
        """
        if self.active_view_index is not None:
             view = self.views[self.active_view_index]
             view.apply_processing(filter_params)

    def load_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG Data", "", "EEG Files (*.edf *.set *.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            sfreq = 250.0
            # If CSV, ask for sampling frequency
            if file_path.lower().endswith('.csv'):
                val, ok = QInputDialog.getDouble(self, "Sampling Frequency", 
                                               "Enter sampling frequency (Hz):", 250.0, 0.1, 10000.0)
                if ok:
                    sfreq = val
                else:
                    return # Cancelled
            
            raw = DataLoader.load_data(file_path, sfreq=sfreq)
            
            # Add to DataManager
            name = file_path.split("/")[-1]
            manager = DataManager()
            final_name = manager.add_signal(name, raw)
            
            # Temporary: Load into active view automatically for immediate feedback
            # In Phase 2, this will be handled by the specialized ComboBox in PlotWidget
            if self.active_view_index is not None:
                self.views[self.active_view_index].load_data(raw)
                
            QMessageBox.information(self, "Success", f"Loaded {final_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def toggle_theme(self, is_dark):
        app = QApplication.instance()
        theme = "dark" if is_dark else "light"
        ThemeManager.apply_theme(app, theme)
