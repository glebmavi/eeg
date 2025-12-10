from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QMessageBox,
                             QSplitter, QFileDialog, QInputDialog, QDialog, QApplication, QToolBar)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from src.gui.control_panel import ControlPanel
from src.gui.plot_widget import PlotWidget
from src.gui.theme_manager import ThemeManager
from src.core.loader import DataLoader
from src.core.data_manager import DataManager
from src.gui.import_dialog import ImportDialog
from src.core.processor import SignalProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVisor - EEG Analysis Environment")
        self.resize(1600, 900)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # Dark Mode Action
        self.action_theme = QAction("Dark Mode", self)
        self.action_theme.setCheckable(True)
        self.action_theme.setChecked(True)
        self.action_theme.toggled.connect(self.toggle_theme)
        self.toolbar.addAction(self.action_theme)

        # Visualization (Left)
        self.root_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.root_splitter, stretch=3)

        # Control Panel (Right)
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)

        # Connections
        self._connect_signals()

        self.views = []
        self._init_views()
        self.active_view_index = None

    def _connect_signals(self):
        cp = self.control_panel
        cp.filter_applied.connect(self.apply_filter_to_active_view)
        cp.load_clicked.connect(self.load_data_file)

        # Analysis
        cp.delta_toggled.connect(lambda c: self.toggle_analysis('delta', c))
        cp.theta_toggled.connect(lambda c: self.toggle_analysis('theta', c))
        cp.alpha_toggled.connect(lambda c: self.toggle_analysis('alpha', c))
        cp.beta_toggled.connect(lambda c: self.toggle_analysis('beta', c))
        cp.gamma_toggled.connect(lambda c: self.toggle_analysis('gamma', c))
        cp.peaks_toggled.connect(lambda c: self.toggle_analysis('peaks', c))

        # Advanced Processing
        cp.ica_requested.connect(self.apply_ica_to_view)
        cp.features_requested.connect(self.show_features)

    def _init_views(self):
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

        is_dark = self.action_theme.isChecked()
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
                f_state, a_state = view.get_state()
                self.control_panel.update_ui_state(f_state, a_state)

    # --- View Management (Split/Close) ---
    def split_view(self, view_widget, orientation):
        if len(self.views) >= 4: return
        parent_splitter = view_widget.parent()
        if not isinstance(parent_splitter, QSplitter): return

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
        if len(self.views) <= 1: return
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
        if splitter == self.root_splitter: return
        if splitter.count() == 0:
            parent = splitter.parent()
            splitter.setParent(None)
            splitter.deleteLater()
            if isinstance(parent, QSplitter):
                self.cleanup_splitter(parent)
        elif splitter.count() == 1:
            child = splitter.widget(0)
            parent = splitter.parent()
            if isinstance(parent, QSplitter):
                index = parent.indexOf(splitter)
                parent.insertWidget(index, child)
                splitter.setParent(None)
                splitter.deleteLater()

    # --- Data & Processing ---
    def load_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG Data", "", "EEG Files (*.edf *.set *.csv);;All Files (*)"
        )
        if not file_path: return

        try:
            sfreq, time_col, unit_scale, description = self._get_import_config(file_path)
            if sfreq is None: return

            raw = DataLoader.load_data(file_path, sfreq=sfreq, time_col=time_col,
                                       unit_scale=unit_scale, description=description)

            name = file_path.split("/")[-1]
            manager = DataManager()
            final_name = manager.add_signal(name, raw)

            self._select_loaded_signal(name)
            if self.active_view_index is not None:
                view = self.views[self.active_view_index]
                self.control_panel.update_ui_state(*view.get_state())

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

            val, ok = QInputDialog.getDouble(self, "Sampling Frequency", "Hz:", 250.0, 0.1, 10000.0)
            if not ok: return None, None, None, None
            sfreq = val

        return sfreq, time_col, unit_scale, description

    def _select_loaded_signal(self, name_prefix):
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

    def toggle_analysis(self, feature, enabled):
        if self.active_view_index is not None and self.active_view_index < len(self.views):
            view = self.views[self.active_view_index]
            if feature == 'peaks':
                view.toggle_peaks(enabled)
            else:
                view.toggle_rhythm(feature, enabled)

    def apply_filter_to_active_view(self, filter_params):
        if self.active_view_index is not None and self.active_view_index < len(self.views):
            self.views[self.active_view_index].apply_processing(filter_params)

    def apply_ica_to_view(self):
        """Applies ICA to the active view to remove artifacts."""
        if self.active_view_index is None: return
        view = self.views[self.active_view_index]
        if view.raw_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded in active view.")
            return

        try:
            # Show waiting cursor
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtGui import QCursor
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

            # Apply ICA
            cleaned = SignalProcessor.apply_ica(view.raw_data)

            # Update View Data (Treat as new raw data essentially, or just update processed)
            # Updating processed means filters re-apply on top of Cleaned.
            # Ideally: Cleaned -> Filters -> Display.
            # We will replace raw_data with cleaned data for this session?
            # Usually ICA is a preprocessing step. Let's update raw_data so filters apply to clean data.
            view.raw_data = cleaned
            view.apply_processing(view.get_state()[0].__dict__)  # Re-apply current filters

            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Success", "ICA Artifact Removal Applied.\n(Removed 1st component)")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "ICA Error", f"Failed to apply ICA:\n{str(e)}")

    def show_features(self):
        """Calculates and displays features for the current signal."""
        if self.active_view_index is None: return
        view = self.views[self.active_view_index]
        if view.processed_data is None: return

        try:
            data = view.processed_data.get_data()[view.current_ch_index]
            sfreq = view.processed_data.info['sfreq']

            features = SignalProcessor.extract_band_powers(data, sfreq)

            # Format Text
            txt = "<b>Relative Band Powers (PSD):</b><br><br>"
            for band, val in features.items():
                txt += f"<b>{band}:</b> {val:.2%}<br>"

            # Add basic stats
            txt += "<br><b>Signal Statistics:</b><br>"
            txt += f"Mean: {data.mean():.2e}<br>"
            txt += f"Std Dev: {data.std():.2e}<br>"

            QMessageBox.information(self, "Signal Features", txt)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Feature extraction failed:\n{e}")