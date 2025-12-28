import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.gui.plot_widget import PlotWidget
from src.gui.control_panel import ControlPanel
from src.gui.import_dialog import ImportDialog
from src.gui.spectrum_window import SpectrumWindow
from src.gui.theme_manager import ThemeManager
from src.gui.custom_axis import PaddedBottomAxis
from src.core.data_manager import DataManager
from src.models.types import FilterState, AnalysisState
import numpy as np
import mne
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pyqtgraph as pg


# ===== MainWindow Tests =====

@pytest.fixture
def main_window(qapp, qtbot, clean_data_manager):
    """Fixture that initializes the MainWindow and DataManager."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


class TestMainWindowInitialization:
    """Tests for MainWindow initialization."""
    
    def test_window_creation(self, main_window):
        """Test that MainWindow is created successfully."""
        assert main_window is not None
        assert main_window.windowTitle() == "NeuroVisor - EEG Analysis Environment"
    
    def test_initial_view_created(self, main_window):
        """Test that at least one view is created initially."""
        assert len(main_window.views) >= 1
    
    def test_control_panel_exists(self, main_window):
        """Test that control panel is created."""
        assert main_window.control_panel is not None
    
    def test_toolbar_exists(self, main_window):
        """Test that toolbar exists."""
        assert main_window.toolbar is not None


class TestMainWindowViewManagement:
    """Tests for view splitting and closing."""
    
    def test_split_view_horizontal(self, main_window, qtbot):
        """Test splitting view horizontally."""
        initial_count = len(main_window.views)
        view = main_window.views[0]
        
        main_window.split_view(view, Qt.Orientation.Horizontal)
        qtbot.wait(50)
        
        assert len(main_window.views) == initial_count + 1
    
    def test_split_view_vertical(self, main_window, qtbot):
        """Test splitting view vertically."""
        initial_count = len(main_window.views)
        view = main_window.views[0]
        
        main_window.split_view(view, Qt.Orientation.Vertical)
        qtbot.wait(50)
        
        assert len(main_window.views) == initial_count + 1
    
    def test_close_view(self, main_window, qtbot):
        """Test closing a view."""
        # First split to have 2 views
        view1 = main_window.views[0]
        main_window.split_view(view1, Qt.Orientation.Horizontal)
        qtbot.wait(50)
        
        initial_count = len(main_window.views)
        view_to_close = main_window.views[1]
        
        main_window.close_view(view_to_close)
        qtbot.wait(50)
        
        assert len(main_window.views) == initial_count - 1
    
    def test_cannot_close_last_view(self, main_window, qtbot):
        """Test that last view cannot be closed."""
        # Ensure only one view
        while len(main_window.views) > 1:
            main_window.close_view(main_window.views[-1])
        
        initial_count = len(main_window.views)
        main_window.close_view(main_window.views[0])
        
        # Should still have one view
        assert len(main_window.views) == initial_count
    
    def test_set_active_view(self, main_window, qtbot):
        """Test setting active view."""
        # Split to have multiple views
        main_window.split_view(main_window.views[0], Qt.Orientation.Horizontal)
        qtbot.wait(50)
        
        main_window.set_active_view(1)
        
        assert main_window.active_view_index == 1


class TestMainWindowDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_data_updates_views(self, main_window, qtbot, clean_data_manager, sample_raw_data):
        """Test that loading data updates all views."""
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=2, signal_type='sine')
        
        dm.add_signal("test_signal.edf", raw)
        qtbot.wait(100)
        
        # Views should have the signal in their combo boxes
        view = main_window.views[0]
        assert view.signal_combo.count() > 1  # More than just "Select Signal..."
    
    def test_select_loaded_signal(self, main_window, qtbot, clean_data_manager, sample_raw_data):
        """Test selecting a loaded signal in a view."""
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=1, signal_type='sine')
        dm.add_signal("test.edf", raw)
        qtbot.wait(50)
        
        view = main_window.views[0]
        view.signal_combo.setCurrentIndex(1)  # Select first signal
        qtbot.wait(50)
        
        assert view.raw_data is not None


class TestMainWindowTheme:
    """Tests for theme management."""
    
    def test_toggle_theme_to_light(self, main_window, qapp, qtbot):
        """Test toggling to light theme."""
        main_window.toggle_theme(False)  # False = light theme
        qtbot.wait(50)
        
        # Should apply theme without crashing
        assert True  # If we get here, theme was applied
    
    def test_toggle_theme_to_dark(self, main_window, qapp, qtbot):
        """Test toggling to dark theme."""
        main_window.toggle_theme(True)  # True = dark theme
        qtbot.wait(50)
        
        assert True  # If we get here, theme was applied


class TestMainWindowProcessing:
    """Tests for signal processing operations."""
    
    def test_toggle_analysis_alpha(self, main_window, qtbot, clean_data_manager, sample_raw_data):
        """Test toggling alpha rhythm analysis."""
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=1, n_samples=2000, signal_type='sine')
        dm.add_signal("test.edf", raw)
        qtbot.wait(50)
        
        view = main_window.views[0]
        view.signal_combo.setCurrentIndex(1)
        qtbot.wait(50)
        
        main_window.set_active_view(0)
        main_window.toggle_analysis('alpha', True)
        qtbot.wait(100)
        
        # Should not crash
        assert True
    
    def test_apply_filter(self, main_window, qtbot, clean_data_manager, sample_raw_data):
        """Test applying filter to active view."""
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=1, signal_type='composite')
        dm.add_signal("test.edf", raw)
        qtbot.wait(50)
        
        view = main_window.views[0]
        view.signal_combo.setCurrentIndex(1)
        main_window.set_active_view(0)
        qtbot.wait(50)
        
        filter_params = {'notch': False, 'detrend': False, 'l_freq': 1.0, 'h_freq': 30.0}
        main_window.apply_filter_to_active_view(filter_params)
        qtbot.wait(100)
        
        # Processed data should exist
        assert view.processed_data is not None


# ===== PlotWidget Tests =====

@pytest.fixture
def plot_widget(qapp, qtbot, clean_data_manager):
    """Fixture for PlotWidget."""
    widget = PlotWidget(index=0)
    qtbot.addWidget(widget)
    return widget


class TestPlotWidgetInitialization:
    """Tests for PlotWidget initialization."""
    
    def test_plot_widget_creation(self, plot_widget):
        """Test PlotWidget is created successfully."""
        assert plot_widget is not None
    
    def test_plot_has_pyqtgraph_widget(self, plot_widget):
        """Test that plot has a pyqtgraph widget."""
        assert plot_widget.plot_item is not None
    
    def test_signal_combo_exists(self, plot_widget):
        """Test that signal combo box exists."""
        assert plot_widget.signal_combo is not None
    
    def test_initial_state(self, plot_widget):
        """Test initial state of PlotWidget."""
        state = plot_widget.get_state()
        assert state is not None


class TestPlotWidgetDataLoading:
    """Tests for PlotWidget data loading."""
    
    def test_load_data(self, plot_widget, sample_raw_data, qtbot):
        """Test loading data into PlotWidget."""
        raw = sample_raw_data(n_channels=1, signal_type='sine')
        
        plot_widget.load_data(raw, "test.edf", ch_index=0)
        qtbot.wait(50)
        
        assert plot_widget.raw_data is not None
        assert plot_widget.label.text() == f"View 1: test.edf - Ch1"
    
    def test_load_multichannel_data(self, plot_widget, sample_raw_data, qtbot):
        """Test loading multichannel data."""
        raw = sample_raw_data(n_channels=5, signal_type='noise')
        
        plot_widget.load_data(raw, "multi.edf", ch_index=2)
        qtbot.wait(50)
        
        assert plot_widget.raw_data is not None
        assert plot_widget.current_ch_index == 2
    
    def test_signal_selection_updates(self, plot_widget, clean_data_manager, sample_raw_data, qtbot):
        """Test that signal selection updates the plot."""
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=1, signal_type='sine')
        dm.add_signal("test.edf", raw)
        qtbot.wait(50)
        
        plot_widget.update_signal_list()
        assert plot_widget.signal_combo.count() > 1


class TestPlotWidgetRhythmVisualization:
    """Tests for rhythm overlay visualization."""
    
    def test_toggle_alpha_rhythm(self, plot_widget, sample_raw_data, qtbot):
        """Test toggling alpha rhythm overlay."""
        raw = sample_raw_data(n_channels=1, n_samples=2000, signal_type='sine')
        plot_widget.load_data(raw, "test.edf")
        qtbot.wait(50)
        
        plot_widget.toggle_rhythm('alpha', True)
        qtbot.wait(100)
        
        assert 'alpha' in plot_widget.rhythm_curves
    
    def test_toggle_beta_rhythm(self, plot_widget, sample_raw_data, qtbot):
        """Test toggling beta rhythm overlay."""
        raw = sample_raw_data(n_channels=1, n_samples=2000, signal_type='sine')
        plot_widget.load_data(raw, "test.edf")
        qtbot.wait(50)
        
        plot_widget.toggle_rhythm('beta', True)
        qtbot.wait(100)
        
        assert 'beta' in plot_widget.rhythm_curves
    
    def test_disable_rhythm(self, plot_widget, sample_raw_data, qtbot):
        """Test disabling rhythm overlay."""
        raw = sample_raw_data(n_channels=1, n_samples=2000, signal_type='sine')
        plot_widget.load_data(raw, "test.edf")
        qtbot.wait(50)
        
        plot_widget.toggle_rhythm('alpha', True)
        qtbot.wait(50)
        plot_widget.toggle_rhythm('alpha', False)
        qtbot.wait(50)
        
        assert 'alpha' not in plot_widget.rhythm_curves


class TestPlotWidgetPeakDetection:
    """Tests for peak detection visualization."""
    
    def test_toggle_peaks_on(self, plot_widget, sample_raw_data, qtbot):
        """Test enabling peak detection."""
        raw = sample_raw_data(n_channels=1, n_samples=1000, signal_type='peaks')
        plot_widget.load_data(raw, "test.edf")
        qtbot.wait(50)
        
        plot_widget.toggle_peaks(True)
        qtbot.wait(100)
        
        assert plot_widget.peak_scatter is not None
    
    def test_toggle_peaks_off(self, plot_widget, sample_raw_data, qtbot):
        """Test disabling peak detection."""
        raw = sample_raw_data(n_channels=1, n_samples=1000, signal_type='peaks')
        plot_widget.load_data(raw, "test.edf")
        qtbot.wait(50)
        
        plot_widget.toggle_peaks(True)
        qtbot.wait(50)
        plot_widget.toggle_peaks(False)
        qtbot.wait(50)
        
        # Peak scatter should be removed
        assert plot_widget.peak_scatter is None or not plot_widget.peak_scatter.isVisible()


class TestPlotWidgetDownsampling:
    """Tests for downsampling functionality."""
    
    def test_downsample_large_data(self, plot_widget):
        """Test downsampling for large datasets."""
        # Create large dataset
        times = np.linspace(0, 100, 50000)
        data = np.sin(times)
        
        downsampled_times, downsampled_data = plot_widget._downsample_for_plot(times, data, max_points=10000)
        
        assert len(downsampled_times) <= 10000
        assert len(downsampled_data) <= 10000
    
    def test_downsample_small_data_unchanged(self, plot_widget):
        """Test that small datasets are not downsampled."""
        times = np.linspace(0, 1, 100)
        data = np.sin(times)
        
        downsampled_times, downsampled_data = plot_widget._downsample_for_plot(times, data, max_points=10000)
        
        assert len(downsampled_times) == len(times)
        assert len(downsampled_data) == len(data)


# ===== ControlPanel Tests =====

@pytest.fixture
def control_panel(qapp, qtbot):
    """Fixture for ControlPanel."""
    panel = ControlPanel()
    qtbot.addWidget(panel)
    return panel


class TestControlPanelInitialization:
    """Tests for ControlPanel initialization."""
    
    def test_control_panel_creation(self, control_panel):
        """Test ControlPanel is created successfully."""
        assert control_panel is not None
    
    def test_filter_controls_exist(self, control_panel):
        """Test that filter controls exist."""
        assert control_panel.notch_cb is not None
        assert control_panel.detrend_cb is not None
        assert control_panel.l_freq_spin is not None
        assert control_panel.h_freq_spin is not None
    
    def test_analysis_controls_exist(self, control_panel):
        """Test that analysis controls exist."""
        assert control_panel.alpha_cb is not None
        assert control_panel.beta_cb is not None
        assert control_panel.gamma_cb is not None
        assert control_panel.theta_cb is not None
        assert control_panel.delta_cb is not None
        assert control_panel.peaks_cb is not None
    
    def test_advanced_controls_exist(self, control_panel):
        """Test that advanced controls exist."""
        assert control_panel.btn_ica is not None
        assert control_panel.btn_features is not None


class TestControlPanelSignals:
    """Tests for ControlPanel signal emissions."""
    
    def test_filter_signal_emission(self, control_panel, qtbot):
        """Test that changing filter settings emits signal."""
        with qtbot.waitSignal(control_panel.filter_applied, timeout=1000):
            control_panel.notch_cb.setChecked(True)
    
    def test_alpha_toggle_signal(self, control_panel, qtbot):
        """Test alpha toggle signal."""
        with qtbot.waitSignal(control_panel.alpha_toggled, timeout=1000):
            control_panel.alpha_cb.setChecked(True)
    
    def test_peaks_toggle_signal(self, control_panel, qtbot):
        """Test peaks toggle signal."""
        with qtbot.waitSignal(control_panel.peaks_toggled, timeout=1000):
            control_panel.peaks_cb.setChecked(True)
    
    def test_ica_button_signal(self, control_panel, qtbot):
        """Test ICA button click signal."""
        with qtbot.waitSignal(control_panel.ica_requested, timeout=1000):
            control_panel.btn_ica.click()
    
    def test_features_button_signal(self, control_panel, qtbot):
        """Test features button click signal."""
        with qtbot.waitSignal(control_panel.features_requested, timeout=1000):
            control_panel.btn_features.click()


class TestControlPanelUIState:
    """Tests for UI state synchronization."""
    
    def test_update_ui_state(self, control_panel, qtbot):
        """Test updating UI state from FilterState and AnalysisState."""
        filter_state = FilterState(notch=True, detrend=True, l_freq=2.0, h_freq=50.0)
        analysis_state = AnalysisState(alpha=True, peaks=True)
        
        control_panel.update_ui_state(filter_state, analysis_state)
        qtbot.wait(50)
        
        assert control_panel.notch_cb.isChecked() is True
        assert control_panel.detrend_cb.isChecked() is True
        assert control_panel.l_freq_spin.value() == 2.0
        assert control_panel.h_freq_spin.value() == 50.0
        assert control_panel.alpha_cb.isChecked() is True
        assert control_panel.peaks_cb.isChecked() is True
    
    def test_emit_filter_settings(self, control_panel, qtbot):
        """Test manual emission of filter settings."""
        control_panel.notch_cb.setChecked(True)
        control_panel.l_freq_spin.setValue(5.0)
        
        with qtbot.waitSignal(control_panel.filter_applied, timeout=1000) as blocker:
            control_panel.emit_filter_settings()
        
        params = blocker.args[0]
        assert params['notch'] is True
        assert params['l_freq'] == 5.0


# ===== ImportDialog Tests =====

class TestImportDialog:
    """Tests for ImportDialog."""
    
    def test_import_dialog_creation(self, qapp, temp_csv_file, qtbot):
        """Test creating ImportDialog with CSV file."""
        file_path = temp_csv_file(n_channels=3, n_samples=50)
        dialog = ImportDialog(file_path)
        qtbot.addWidget(dialog)
        
        assert dialog is not None
    
    def test_import_dialog_preview(self, qapp, temp_csv_file, qtbot):
        """Test that dialog shows data preview."""
        file_path = temp_csv_file(n_channels=2, n_samples=10)
        dialog = ImportDialog(file_path)
        qtbot.addWidget(dialog)
        
        # Preview table should be populated
        assert dialog.preview_table.rowCount() > 0
    
    def test_get_settings(self, qapp, temp_csv_file, qtbot):
        """Test retrieving settings from dialog."""
        file_path = temp_csv_file(n_channels=2)
        dialog = ImportDialog(file_path)
        qtbot.addWidget(dialog)
        
        time_col, unit, channel_types = dialog.get_settings()
        
        # Should return some values
        assert unit is not None


# ===== SpectrumWindow Tests =====

class TestSpectrumWindow:
    """Tests for SpectrumWindow."""
    
    def test_spectrum_window_creation(self, qapp, sample_signals, qtbot):
        """Test creating SpectrumWindow."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        window = SpectrumWindow(data, sfreq)
        qtbot.addWidget(window)
        
        assert window is not None
    
    def test_spectrum_psd_display(self, qapp, sample_signals, qtbot):
        """Test PSD display mode."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        window = SpectrumWindow(data, sfreq)
        qtbot.addWidget(window)
        
        # Default should be PSD
        window.combo_method.setCurrentIndex(0)
        qtbot.wait(100)
        
        # Should not crash
        assert True
    
    def test_spectrum_wavelet_display(self, qapp, sample_signals, qtbot):
        """Test wavelet display mode."""
        data = sample_signals['sine_10hz']
        sfreq = sample_signals['sfreq']
        
        window = SpectrumWindow(data, sfreq)
        qtbot.addWidget(window)
        
        # Switch to wavelet
        window.combo_method.setCurrentIndex(1)
        qtbot.wait(200)
        
        # Should not crash
        assert True


# ===== ThemeManager Tests =====

class TestThemeManager:
    """Tests for ThemeManager."""
    
    def test_apply_dark_theme(self, qapp):
        """Test applying dark theme."""
        ThemeManager.apply_theme(qapp, "dark")
        
        # Should apply without crashing
        assert True
    
    def test_apply_light_theme(self, qapp):
        """Test applying light theme."""
        ThemeManager.apply_theme(qapp, "light")
        
        # Should apply without crashing
        assert True
    
    def test_apply_nonexistent_theme(self, qapp):
        """Test applying nonexistent theme."""
        # Should handle gracefully
        ThemeManager.apply_theme(qapp, "nonexistent")
        
        # Should not crash
        assert True


# ===== CustomAxis Tests =====

class TestCustomAxis:
    """Tests for PaddedBottomAxis."""
    
    def test_custom_axis_creation(self):
        """Test creating PaddedBottomAxis."""
        axis = PaddedBottomAxis('bottom')
        
        assert axis is not None
    
    def test_custom_axis_bounding_rect(self):
        """Test that bounding rect is adjusted."""
        axis = PaddedBottomAxis('bottom')
        rect = axis.boundingRect()
        
        # Should have some size
        assert rect is not None
    
    def test_custom_axis_orientation(self):
        """Test axis with different orientations."""
        bottom_axis = PaddedBottomAxis('bottom')
        left_axis = PaddedBottomAxis('left')
        
        assert bottom_axis.orientation == 'bottom'
        assert left_axis.orientation == 'left'


# ===== Integration Tests =====

class TestGUIIntegration:
    """Integration tests for E2E GUI workflows."""
    
    def test_complete_workflow(self, main_window, qtbot, clean_data_manager, sample_raw_data):
        """Test complete workflow: load data -> apply filter -> show peaks."""
        # 1. Load data
        dm = clean_data_manager
        raw = sample_raw_data(n_channels=1, n_samples=2000, signal_type='peaks')
        dm.add_signal("test.edf", raw)
        qtbot.wait(100)
        
        # 2. Select signal
        view = main_window.views[0]
        view.signal_combo.setCurrentIndex(1)
        main_window.set_active_view(0)
        qtbot.wait(100)
        
        # 3. Apply filter
        filter_params = {'notch': False, 'detrend': False, 'l_freq': 1.0, 'h_freq': 40.0}
        main_window.apply_filter_to_active_view(filter_params)
        qtbot.wait(100)
        
        # 4. Show peaks
        main_window.control_panel.peaks_cb.setChecked(True)
        qtbot.wait(200)
        
        # Verify workflow completed
        assert view.processed_data is not None
        assert view.peak_scatter is not None
