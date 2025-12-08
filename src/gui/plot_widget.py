from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QMenu
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import numpy as np
import mne
import scipy.signal
from src.core.data_manager import DataManager
from src.gui.spectrum_window import SpectrumWindow
from src.core.processor import SignalProcessor

class PlotWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Header / Status / Selection
        self.top_bar = QWidget()
        top_layout = QVBoxLayout(self.top_bar) 
        top_layout.setContentsMargins(0,0,0,0)
        
        self.label = QLabel(f"View {index + 1}: No Data")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.label)
        
        self.signal_combo = QComboBox()
        self.signal_combo.addItem("Select Signal...")
        self.signal_combo.currentIndexChanged.connect(self.on_signal_selected)
        top_layout.addWidget(self.signal_combo)
        
        self.layout.addWidget(self.top_bar)
        
        # PyQtGraph Plot
        self.plot_item = pg.PlotWidget()
        self.plot_item.setBackground('w')
        self.plot_item.showGrid(x=True, y=True)
        self.layout.addWidget(self.plot_item)

        self.raw_data = None
        self.processed_data = None
        self.is_active = False
        
        # Analysis items
        self.main_curve = None
        self.rhythm_curves = {}
        self.peak_scatter = None
        
        # DataManager connection
        self.data_manager = DataManager()
        self.data_manager.add_listener(self.update_signal_list)
        self.update_signal_list(self.data_manager.get_signal_names())

    def update_signal_list(self, names):
        current = self.signal_combo.currentText()
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        self.signal_combo.addItem("Select Signal...")
        self.signal_combo.addItems(names)
        
        index = self.signal_combo.findText(current)
        if index >= 0:
            self.signal_combo.setCurrentIndex(index)
        self.signal_combo.blockSignals(False)

    def on_signal_selected(self, index):
        if index <= 0: return 
        name = self.signal_combo.currentText()
        raw = self.data_manager.get_signal(name)
        if raw:
            self.load_data(raw)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        if self.processed_data is None:
            return
        
        menu = QMenu(self)
        action_spectrum = QAction("Show Frequency Spectrum (FFT)", self)
        action_spectrum.triggered.connect(self.show_spectrum)
        menu.addAction(action_spectrum)
        
        menu.exec(event.globalPos())

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.setStyleSheet("border: 2px solid blue;")
        else:
            self.setStyleSheet("border: 1px solid gray;")

    def load_data(self, raw: mne.io.BaseRaw):
        self.raw_data = raw
        self.processed_data = raw.copy()
        
        # Clear previous analysis
        self.rhythm_curves = {}
        if self.peak_scatter:
            self.plot_item.removeItem(self.peak_scatter)
            self.peak_scatter = None
            
        self.label.setText(f"View {self.index + 1}: Loaded {raw.filenames[0] if raw.filenames else 'Data'}")
        self.update_plot()

    def apply_processing(self, params):
        if self.raw_data is None:
            return
            
        # Start from copy of raw
        temp = self.raw_data.copy()
        
        if params.get('detrend'):
            temp = SignalProcessor.detrend_signal(temp)
            
        if params.get('notch'):
            temp = SignalProcessor.apply_notch(temp, np.array([50.0]))
            
        l_freq = params.get('l_freq')
        h_freq = params.get('h_freq')
        if l_freq and h_freq:
            temp = SignalProcessor.apply_filter(temp, l_freq, h_freq)
            
        self.processed_data = temp
        self.update_plot()

    def show_spectrum(self):
        if self.processed_data is None: 
            return
        data = self.processed_data.get_data()[0]
        sfreq = self.processed_data.info['sfreq']
        
        self.spectrum_win = SpectrumWindow(data, sfreq, self)
        self.spectrum_win.show()

    def toggle_rhythm(self, rhythm_type: str, enabled: bool):
        """
        rhythm_type: 'alpha', 'beta', 'gamma', 'theta', 'delta'
        """
        if self.processed_data is None: return

        if rhythm_type in self.rhythm_curves:
            self.plot_item.removeItem(self.rhythm_curves[rhythm_type])
            del self.rhythm_curves[rhythm_type]
            
        if not enabled:
            return

        # Calculate Rhythm
        # Bands: Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-100)
        # Colors: Delta=Cyan, Theta=Magenta, Alpha=Red, Beta=Green, Gamma=Blue(ish) or Yellow
        bands = {
            'delta': (0.5, 4, 'c'),
            'theta': (4, 8, 'm'),
            'alpha': (8, 13, 'r'), 
            'beta': (13, 30, 'g'),
            'gamma': (30, 100, 'y') 
        }
        if rhythm_type not in bands: return
        
        l_freq, h_freq, color = bands[rhythm_type]
        
        # Filter a copy
        rhythm_raw = self.processed_data.copy()
        try:
             # Just in case low freq is too low for standard filter defaults or high is too high
            rhythm_raw.filter(l_freq, h_freq, verbose=False)
        except Exception as e:
            print(f"Filter error for {rhythm_type}: {e}")
            return

        data = rhythm_raw.get_data()[0] * 1e6
        times = rhythm_raw.times
        
        # Plot on top
        curve = self.plot_item.plot(times, data, pen=pg.mkPen(color, width=2))
        self.rhythm_curves[rhythm_type] = curve

    def toggle_peaks(self, enabled: bool):
        if self.peak_scatter:
            self.plot_item.removeItem(self.peak_scatter)
            self.peak_scatter = None
            
        if not enabled or self.processed_data is None:
            return
            
        data = self.processed_data.get_data()[0] * 1e6
        times = self.processed_data.times
        
        # Heuristic height: > 2 std dev? or just simple
        height = np.std(data) * 2
        peaks, _ = SignalProcessor.detect_peaks(data, height=height, distance=50)
        
        if len(peaks) > 0:
            self.peak_scatter = pg.ScatterPlotItem(x=times[peaks], y=data[peaks], pen='r', brush='r', size=8)
            self.plot_item.addItem(self.peak_scatter)

    def update_plot(self):
        if self.processed_data is None:
            return
        
        self.plot_item.clear()
        self.rhythm_curves = {}
        self.peak_scatter = None
        
        data, times = self.processed_data[:, :]
        valid_data = data[0].flatten() * 1e6 
        
        self.main_curve = self.plot_item.plot(times, valid_data, pen='k')

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.setStyleSheet("border: 2px solid blue;")
        else:
            self.setStyleSheet("border: 1px solid gray;")

    def load_data(self, raw: mne.io.BaseRaw):
        self.raw_data = raw
        self.processed_data = raw.copy()
        self.label.setText(f"View {self.index + 1}: Loaded {raw.filenames[0] if raw.filenames else 'Data'}")
        self.update_plot()

    def apply_processing(self, params):
        if self.raw_data is None:
            return
            
        from src.core.processor import SignalProcessor
        
        # Start from fresh copy of raw to avoid accumulating filters (simplified logic)
        temp = self.raw_data.copy()
        
        if params.get('detrend'):
            temp = SignalProcessor.detrend_signal(temp)
            
        if params.get('notch'):
            # Assuming 50Hz for now
            temp = SignalProcessor.apply_notch(temp, np.array([50.0]))
            
        l_freq = params.get('l_freq')
        h_freq = params.get('h_freq')
        if l_freq and h_freq:
            temp = SignalProcessor.apply_filter(temp, l_freq, h_freq)
            
        self.processed_data = temp
        self.update_plot()

    def update_plot(self):
        if self.processed_data is None:
            return
            
        self.plot_item.clear()
        
        # Plotting the first channel for demonstration
        data, times = self.processed_data[:, :]  # Get all data
        # data is (n_channels, n_times)
        
        # Just plot the first channel for performance/demo
        valid_data = data[0].flatten() * 1e6 # Convert to microvolts usually
        
        self.plot_item.plot(times, valid_data, pen='k')
