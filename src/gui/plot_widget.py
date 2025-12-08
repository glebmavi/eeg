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
    split_requested = pyqtSignal(QWidget, Qt.Orientation)
    close_requested = pyqtSignal(QWidget)

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
        self.current_ch_index = 0
        self.is_active = False
        
        # Analysis items
        self.main_curve = None
        self.rhythm_curves = {}
        self.peak_scatter = None
        
        # DataManager connection
        self.data_manager = DataManager()
        self.data_manager.add_listener(self.update_signal_list)
        # Initial population
        self.update_signal_list()

    def update_signal_list(self):
        current_text = self.signal_combo.currentText()
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        self.signal_combo.addItem("Select Signal...", None)
        
        channels = self.data_manager.get_all_channels()
        # channels is list of (filename, ch_idx, ch_name)
        
        for fname, ch_idx, ch_name in channels:
            display_name = f"{fname} - {ch_name}"
            # Store tuple (fname, ch_idx) as user data
            self.signal_combo.addItem(display_name, (fname, ch_idx))
        
        # Restore selection if possible
        index = self.signal_combo.findText(current_text)
        if index >= 0:
            self.signal_combo.setCurrentIndex(index)
        self.signal_combo.blockSignals(False)

    def on_signal_selected(self, index):
        if index <= 0: return 
        
        data = self.signal_combo.currentData()
        if data:
            fname, ch_idx = data
            raw = self.data_manager.get_signal(fname)
            if raw:
                # Load the raw object but set the specific channel index
                self.load_data(raw, fname, ch_idx)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        # Analysis Group
        if self.processed_data is not None:
             action_spectrum = QAction("Show Frequency Spectrum (FFT)", self)
             action_spectrum.triggered.connect(self.show_spectrum)
             menu.addAction(action_spectrum)
             menu.addSeparator()
        
        # View Management Group
        split_h = QAction("Split Horizontal", self)
        split_h.triggered.connect(lambda: self.split_requested.emit(self, Qt.Orientation.Horizontal))
        
        split_v = QAction("Split Vertical", self)
        split_v.triggered.connect(lambda: self.split_requested.emit(self, Qt.Orientation.Vertical))
        
        close_view = QAction("Close View", self)
        close_view.triggered.connect(lambda: self.close_requested.emit(self))
        
        menu.addAction(split_h)
        menu.addAction(split_v)
        menu.addSeparator()
        menu.addAction(close_view)
        
        menu.exec(event.globalPos())

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            # We used blue to highlight active view
            self.setStyleSheet("border: 2px solid blue;")
        else:
            self.setStyleSheet("border: 1px solid gray;")

    def load_data(self, raw: mne.io.BaseRaw, filename: str, ch_index: int = 0):
        self.raw_data = raw
        self.current_ch_index = ch_index
        # We start with a copy of the raw data. 
        # Note: If raw is very large, this copy might be expensive. 
        # But MNE usually preloads if we asked it to.
        self.processed_data = raw.copy()
        
        # Clear previous analysis
        self.rhythm_curves = {}
        if self.peak_scatter:
            self.plot_item.removeItem(self.peak_scatter)
            self.peak_scatter = None
            
        ch_name = raw.ch_names[ch_index] if raw.ch_names else f"Ch{ch_index}"
        
        fname = filename
            
        self.label.setText(f"View {self.index + 1}: {fname} - {ch_name}")
        self.update_plot()

    def apply_processing(self, params):
        if self.raw_data is None:
            return
            
        # Start from clean copy to ensure non-destructive filtering
        # Reactive filtering needs this base state
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
        
        try:
             # Get specific channel data
            data = self.processed_data.get_data()[self.current_ch_index]
            sfreq = self.processed_data.info['sfreq']
            
            self.spectrum_win = SpectrumWindow(data, sfreq, self)
            self.spectrum_win.show()
        except Exception as e:
            print(f"Error showing spectrum: {e}")

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
            rhythm_raw.filter(l_freq, h_freq, verbose=False)
        except Exception as e:
            print(f"Filter error for {rhythm_type}: {e}")
            return

        data = rhythm_raw.get_data()[self.current_ch_index] * 1e6
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
            
        data = self.processed_data.get_data()[self.current_ch_index] * 1e6
        times = self.processed_data.times
        
        # Heuristic height: > 2 std dev
        height = np.std(data) * 2
        peaks, _ = SignalProcessor.detect_peaks(data, height=height, distance=50)
        
        if len(peaks) > 0:
            self.peak_scatter = pg.ScatterPlotItem(x=times[peaks], y=data[peaks], pen='r', brush='r', size=8)
            self.plot_item.addItem(self.peak_scatter)

    def apply_theme(self, is_dark: bool):
        if is_dark:
            self.plot_item.setBackground('#2b2b2b')
            # self.plot_item.getAxis('bottom').setPen('#ffffff') # Optional polish
        else:
            self.plot_item.setBackground('w')

    def update_plot(self):
        if self.processed_data is None:
            return
            
        self.plot_item.clear()
        self.rhythm_curves = {}
        self.peak_scatter = None
        
        # Get all data (n_channels, n_times)
        data = self.processed_data.get_data()
        times = self.processed_data.times
        
        # Pick just the current channel
        if self.current_ch_index < len(data):
            # Check scaling
            scaling = 1e6 # Default MNE Volts -> uV
            if self.raw_data.info.get('description') == "Raw/ADC":
                scaling = 1.0
            
            valid_data = data[self.current_ch_index] * scaling 
            
            # Choose Pen Color based on theme/default
            # Simple black/white contrast handled by pyqtgraph usually if k is black.
            # If dark theme, 'k' (black) might be invisible on dark bg.
            # We should make pen adaptive or use a color that works on both (like cyan/green) or check theme.
            # For now, default 'd' (default) might be better or adaptive.
            # 'w' for white, 'k' for black.
            # Let's check background.
            bg = self.plot_item.backgroundBrush.color().name() if hasattr(self.plot_item.backgroundBrush, 'color') else 'w'
            pen_color = 'w' if bg == '#2b2b2b' else 'k'
            
            self.main_curve = self.plot_item.plot(times, valid_data, pen=pen_color)
        else:
            self.label.setText("Error: Channel Index Out of Bounds")
