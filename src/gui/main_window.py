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
        
        # Left Side: Visualization Area (Dynamic Splitter)
        self.root_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.root_splitter, stretch=3)

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

        self.views = []
        self._init_views()

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
        """Initialize with a single view."""
        self.create_view(self.root_splitter)
        
        # Set first one active
        if self.views:
            self.set_active_view(0)

    def create_view(self, parent_splitter):
        index = len(self.views)
        plot = PlotWidget(index=index)
        parent_splitter.addWidget(plot)
        self.views.append(plot)
        
        # Connect signals
        plot.clicked.connect(self.set_active_view_by_widget)
        plot.split_requested.connect(self.split_view)
        plot.close_requested.connect(self.close_view)
        
        # Apply current theme
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
        """
        Splits the given view_widget in the specified orientation.
        orientation: Qt.Orientation.Horizontal or Qt.Orientation.Vertical
        """
        # Find parent splitter
        parent = view_widget.parent()
        if not isinstance(parent, QSplitter):
            return

        parent_splitter = parent
        index = parent_splitter.indexOf(view_widget)
        sizes = parent_splitter.sizes()
        current_size = sizes[index]
        new_splitter = QSplitter(orientation)
        
        # Add new splitter to parent
        parent_splitter.insertWidget(index, new_splitter)
        
        # Move current view to new splitter
        new_splitter.addWidget(view_widget)
        
        # Create new view
        self.create_view(new_splitter)
        
        # Restore sizes roughly (distribute space)
        new_splitter.setSizes([current_size // 2, current_size // 2])
        
    def close_view(self, view_widget):
        if len(self.views) <= 1:
            return # Don't close the last one
            
        view_widget.setParent(None)
        view_widget.deleteLater()
        if view_widget in self.views:
            self.views.remove(view_widget)
            
        # If we closed the active view, select another
        if self.views:
            self.set_active_view(len(self.views) - 1)

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
            time_col = None
            unit_scale = 1.0
            
            # If CSV, use Import Dialog
            if file_path.lower().endswith('.csv'):
                # First ask for Import Config
                dlg = ImportDialog(file_path, self)
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    time_col, unit_str = dlg.get_settings()
                    
                    description = None
                    
                    # Logic for unit scale:
                    # Input "Volts" -> MNE wants Volts -> scale = 1.0
                    # Input "uV" -> MNE wants Volts -> scale = 1e-6
                    # Input "Raw" -> MNE wants Volts -> scale = 1.0 (interpret as Volts for storage simplicity)
                    
                    if unit_str == "Microvolts (uV)":
                        unit_scale = 1e-6
                    elif unit_str == "Raw/ADC":
                        unit_scale = 1.0
                        description = "Raw/ADC" # Flag for PlotWidget to not scale
                    else:
                        unit_scale = 1.0
                        
                    # Also ask for Frequency
                    val, ok = QInputDialog.getDouble(self, "Sampling Frequency", 
                                                    "Enter sampling frequency (Hz):", 250.0, 0.1, 10000.0)
                    if ok:
                        sfreq = val
                    else:
                        return
                        
                else:
                    return # Cancelled
            
            raw = DataLoader.load_data(file_path, sfreq=sfreq, time_col=time_col, 
                                       unit_scale=unit_scale, description=description)
            
            # Add to DataManager
            name = file_path.split("/")[-1]
            manager = DataManager()
            final_name = manager.add_signal(name, raw)
            
            if self.active_view_index is not None and len(self.views) > self.active_view_index:
                view = self.views[self.active_view_index]
                # If view is empty (raw_data is None), auto-load
                if view.raw_data is None:
                    # We need to trigger the combo box update in the view since DataManager updated
                    # But the view is a listener, so it might have already updated its list? 
                    # Yes, DataManager notifies listeners. 
                    # We Just need to select it.
                    
                    # Find the index of the new item in the combo
                    # The combo format is "filename - channel"
                    # We'll just select the first channel of the new file
                    # Or simpler: just select index 1 (the first real item) if we want any data
                    
                    # Search for items starting with our file name
                    combo = view.signal_combo
                    for i in range(combo.count()):
                        data = combo.itemData(i)
                        if data and data[0] == name: # data is (fname, ch_idx)
                            combo.setCurrentIndex(i)
                            break 
                
            QMessageBox.information(self, "Success", f"Loaded {final_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def toggle_theme(self, is_dark):
        app = QApplication.instance()
        theme = "dark" if is_dark else "light"
        ThemeManager.apply_theme(app, theme)
        
        for view in self.views:
            view.apply_theme(is_dark)
