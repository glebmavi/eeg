import pytest
from PyQt6.QtCore import Qt
from src.gui.main_window import MainWindow
from src.core.data_manager import DataManager
import numpy as np
import mne

@pytest.fixture
def main_window(qapp, qtbot):
    """Fixture that initializes the MainWindow and DataManager."""
    # Clear DataManager before test
    dm = DataManager()
    dm.signals = {}
    
    window = MainWindow()
    qtbot.addWidget(window)
    return window

def test_gui_load_and_display(main_window, qtbot):
    """E2E Test: Inject data into manager and verify GUI updates."""
    # 1. Manually load data into Manager (bypassing file dialog for test stability)
    info = mne.create_info(['Fp1'], 250, ['eeg'])
    data = np.random.normal(0, 1, (1, 1000))
    raw = mne.io.RawArray(data, info)
    
    dm = DataManager()
    dm.add_signal("mock_eeg.edf", raw)
    
    # 2. Check if View 0 got the signal list update
    view = main_window.views[0]
    qtbot.wait(50) # Allow signal propagation
    
    # 3. Simulate selecting the signal in the combo box
    # Index 0 is "Select Signal...", Index 1 is our file
    assert view.signal_combo.count() > 1
    view.signal_combo.setCurrentIndex(1)
    
    # 4. Verify plot loaded
    assert view.raw_data is not None
    assert view.label.text().__contains__("mock_eeg.edf")

def test_gui_processing_toggle(main_window, qtbot):
    """Test clicking buttons in Control Panel affects the PlotWidget."""
    # Setup Data: Use data with a clear peak for peak detection to succeed.
    info = mne.create_info(['Fp1'], 250, ['eeg'])
    data = np.zeros((1, 1000))
    # Inject a peak: data will have non-zero standard deviation and a clear peak
    data[0, 500] = 5.0
    data[0, 501] = 4.0
    raw = mne.io.RawArray(data, info)
    DataManager().add_signal("flat.edf", raw)
    
    view = main_window.views[0]
    view.signal_combo.setCurrentIndex(1)
    main_window.set_active_view(0)

    # 1. Toggle Peaks
    cb = main_window.control_panel.peaks_cb
    # Use qtbot.mouseClick for reliable interaction and event processing
    qtbot.mouseClick(cb, Qt.MouseButton.LeftButton)

    # Wait for the signal to process the data and update the plot
    qtbot.wait(100)

    # Verify view state: peak_scatter should now be created
    assert view.peak_scatter is not None

    # 2. Toggle Alpha Rhythm
    cb_alpha = main_window.control_panel.alpha_cb
    qtbot.mouseClick(cb_alpha, Qt.MouseButton.LeftButton)

    # Wait for the signal to process
    qtbot.wait(100)

    # Verify rhythm curve exists
    assert 'alpha' in view.rhythm_curves

def test_plot_split_close(main_window, qtbot):
    """Test splitting and closing views."""
    assert len(main_window.views) == 1
    
    # Split Horizontal
    view1 = main_window.views[0]
    main_window.split_view(view1, Qt.Orientation.Horizontal)
    
    assert len(main_window.views) == 2
    
    # Close second view
    view2 = main_window.views[1]
    main_window.close_view(view2)
    
    assert len(main_window.views) == 1