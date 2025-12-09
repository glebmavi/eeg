from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QMdiArea, QFileDialog, QInputDialog, QMessageBox, QApplication, QSplitter, QDialog)
from PyQt6.QtCore import Qt
from src.gui.control_panel import ControlPanel
from src.gui.plot_widget import PlotWidget
from src.gui.theme_manager import ThemeManager
from src.core.loader import DataLoader
from src.core.data_manager import DataManager
from src.gui.import_dialog import ImportDialog
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
        
        # Left Side: Visualization Area
        self.root_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.root_splitter, stretch=3)

        # Right Side: Control Global Panel
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)
        
        # Signals
        self.control_panel.filter_applied.connect(self.apply_filter_to_active_view)
        self.control_panel.load_clicked.connect(self.load_data_file)
        self.control_panel.theme_toggled.connect(self.toggle_theme)
        self.control_panel.validation_clicked.connect(self.open_validation_dialog)
        
        self.control_panel.delta_toggled.connect(lambda c: self.toggle_analysis('delta', c))
        self.control_panel.theta_toggled.connect(lambda c: self.toggle_analysis('theta', c))
        self.control_panel.alpha_toggled.connect(lambda c: self.toggle_analysis('alpha', c))
        self.control_panel.beta_toggled.connect(lambda c: self.toggle_analysis('beta', c))
        self.control_panel.gamma_toggled.connect(lambda c: self.toggle_analysis('gamma', c))
        self.control_panel.peaks_toggled.connect(lambda c: self.toggle_analysis('peaks', c))

        self.views = []
        self._init_views()
        self.active_view_index = None

    def toggle_analysis(self, feature, enabled):
        if self.active_view_index is not None and self.active_view_index < len(self.views):
            view = self.views[self.active_view_index]
            if feature == 'peaks':
                view.toggle_peaks(enabled)
            else:
                view.toggle_rhythm(feature, enabled)

    def open_validation_dialog(self):
        data = None
        sfreq = 250.0
        
        if self.active_view_index is not None and self.active_view_index < len(self.views):
             view = self.views[self.active_view_index]
             if view.processed_data is not None:
                 data = view.processed_data.get_data()
                 sfreq = view.processed_data.info['sfreq']
                 
        dialog = ValidationDialog(data, sfreq, self)
        dialog.exec()

    def _init_views(self):
        """Initialize with a single view."""
        self.create_view(self.root_splitter)
        if self.views:
            self.set_active_view(0)

    def create_view(self, parent_splitter):
        index = len(self.views)
        plot = PlotWidget(index=index)
        parent_splitter.addWidget(plot)
        self.views.append(plot)
        
        plot.clicked.connect(self.set_active_view_by_widget)
        plot.split_requested.connect(self.split_view)
        plot.close_requested.connect(self.close_view)
        
        is_dark = self.control_panel.theme_cb.isChecked()
        plot.apply_theme(is_dark)
        
        return plot

    def set_active_view(self, index):
        if 0 <= index < len(self.views):
            self.set_active_view_by_widget(self.views[index])

    def set_active_view_by_widget(self, widget=None):
        if widget is None and self.sender():
            widget = self.sender()
            
        for i, view in enumerate(self.views):
            is_active = (view == widget)
            view.set_active(is_active)
            if is_active:
                self.active_view_index = i

    def split_view(self, view_widget, orientation):
        """Splits the view_widget, enforcing max 4 views."""
        if len(self.views) >= 4:
            return

        parent_splitter = view_widget.parent()
        if not isinstance(parent_splitter, QSplitter):
            return

        index = parent_splitter.indexOf(view_widget)
        current_size = parent_splitter.sizes()[index]
        
        new_splitter = QSplitter(orientation)
        new_splitter.setHandleWidth(4)
        new_splitter.setStyleSheet("QSplitter::handle { background-color: #555; }")
        
        parent_splitter.insertWidget(index, new_splitter)
        new_splitter.addWidget(view_widget)
        
        self.create_view(new_splitter)
        new_splitter.setSizes([current_size // 2, current_size // 2])
        
    def close_view(self, view_widget):
        if len(self.views) <= 1:
            return 
            
        parent = view_widget.parent()
        view_widget.setParent(None)
        view_widget.deleteLater()
        if view_widget in self.views:
            self.views.remove(view_widget)
            
        if isinstance(parent, QSplitter):
            self.cleanup_splitter(parent)
            
        if self.views:
            self.set_active_view(len(self.views) - 1)

    def cleanup_splitter(self, splitter):
        if splitter == self.root_splitter:
            return
            
        count = splitter.count()
        if count == 0:
            parent = splitter.parent()
            splitter.setParent(None)
            splitter.deleteLater()
            if isinstance(parent, QSplitter):
                self.cleanup_splitter(parent)
        elif count == 1:
            # Move single child up to parent, replacing this splitter
            child = splitter.widget(0)
            parent = splitter.parent()
            
            if isinstance(parent, QSplitter):
                index = parent.indexOf(splitter)
                parent.insertWidget(index, child)
                splitter.setParent(None)
                splitter.deleteLater()

    def apply_filter_to_active_view(self, filter_params):
        if self.active_view_index is not None and self.active_view_index < len(self.views):
             view = self.views[self.active_view_index]
             view.apply_processing(filter_params)

    def load_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG Data", "", "EEG Files (*.edf *.set *.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            sfreq, time_col, unit_scale, description = self._get_import_config(file_path)
            if sfreq is None: # Cancelled
                return
            
            raw = DataLoader.load_data(file_path, sfreq=sfreq, time_col=time_col, 
                                       unit_scale=unit_scale, description=description)
            
            name = file_path.split("/")[-1]
            manager = DataManager()
            final_name = manager.add_signal(name, raw)
            
            self._select_loaded_signal(name)
                
            QMessageBox.information(self, "Success", f"Loaded {final_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def _get_import_config(self, file_path):
        sfreq = 250.0
        time_col = None
        unit_scale = 1.0
        description = None

        if file_path.lower().endswith('.csv'):
            dlg = ImportDialog(file_path, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None, None, None, None
                
            time_col, unit_str = dlg.get_settings()
            
            if unit_str == "Microvolts (uV)":
                unit_scale = 1e-6
            elif unit_str == "Raw/ADC":
                unit_scale = 1.0
                description = "Raw/ADC"

            val, ok = QInputDialog.getDouble(self, "Sampling Frequency", 
                                            "Enter sampling frequency (Hz):", 250.0, 0.1, 10000.0)
            if not ok:
                 return None, None, None, None
            sfreq = val

        return sfreq, time_col, unit_scale, description

    def _select_loaded_signal(self, name_prefix):
        """Helper to auto-select signal in active view."""
        if self.active_view_index is not None and self.active_view_index < len(self.views):
            view = self.views[self.active_view_index]
            if view.raw_data is None:
                combo = view.signal_combo
                for i in range(combo.count()):
                    data = combo.itemData(i)
                    if data and data[0] == name_prefix:
                        combo.setCurrentIndex(i)
                        break

    def toggle_theme(self, is_dark):
        app = QApplication.instance()
        theme = "dark" if is_dark else "light"
        ThemeManager.apply_theme(app, theme)
        
        for view in self.views:
            view.apply_theme(is_dark)
