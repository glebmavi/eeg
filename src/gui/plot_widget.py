from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QMenu, QToolTip
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import numpy as np
import mne

from src.core.data_manager import DataManager
from src.gui.spectrum_window import SpectrumWindow
from src.core.processor import SignalProcessor
from src.models.types import RhythmBands, FilterState, AnalysisState
from src.gui.custom_axis import PaddedBottomAxis


class PlotWidget(QWidget):
    """
    Widget for visualizing EEG signals. 
    Handles plotting, interactions (zoom/pan), and visual analysis overlays (peaks, rhythms).
    """
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
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(f"View {index + 1}: No Data")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.label)

        self.signal_combo = QComboBox()
        self.signal_combo.addItem("Select Signal...")
        self.signal_combo.currentIndexChanged.connect(self.on_signal_selected)
        top_layout.addWidget(self.signal_combo)

        self.layout.addWidget(self.top_bar)

        # PyQtGraph Plot
        self.plot_item = pg.PlotWidget(axisItems={'bottom': PaddedBottomAxis(orientation='bottom')})
        self.plot_item.setBackground('w')
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('bottom', "Time (s)")

        self.plot_item.getPlotItem().setContentsMargins(5, 5, 5, 20)
        self.layout.addWidget(self.plot_item)

        # Signal Proxy for Hover
        self.proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_move)

        self.raw_data = None
        self.processed_data = None
        self.current_ch_index = 0
        self.is_active = False

        self.main_curve = None
        self.rhythm_curves = {}
        self.peak_scatter = None

        self.data_manager = DataManager()
        self.data_manager.add_listener(self.update_signal_list)
        self.update_signal_list()

        # State Tracking
        self.filter_state = FilterState()
        self.analysis_state = AnalysisState()

    def get_state(self):
        """Returns the current filter and analysis state objects."""
        return self.filter_state, self.analysis_state

    def _get_scale_and_unit(self):
        """Determine scaling and unit string."""
        if self.raw_data and self.raw_data.info.get('description') == "Raw/ADC":
            return 1.0, "ADC"
        return 1e6, "uV"  # Default MNE Volts -> Microvolts

    def update_signal_list(self):
        current_text = self.signal_combo.currentText()
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        self.signal_combo.addItem("Select Signal...", None)

        for fname, ch_idx, ch_name in self.data_manager.get_all_channels():
            display_name = f"{fname} - {ch_name}"
            self.signal_combo.addItem(display_name, (fname, ch_idx))

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
                self.load_data(raw, fname, ch_idx)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        child = self.childAt(event.pos())
        if child and (child == self.top_bar or self.top_bar.isAncestorOf(child)):
            self.show_window_menu(event.globalPos())

    def show_window_menu(self, pos):
        menu = QMenu(self)
        if self.processed_data is not None:
            action_spectrum = QAction("Show Frequency Spectrum (FFT)", self)
            action_spectrum.triggered.connect(self.show_spectrum)
            menu.addAction(action_spectrum)
            menu.addSeparator()

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
        menu.exec(pos)

    def set_active(self, active: bool):
        self.is_active = active
        self.setStyleSheet("border: 2px solid blue;" if active else "border: 1px solid gray;")

    def load_data(self, raw: mne.io.BaseRaw, filename: str, ch_index: int = 0):
        self.raw_data = raw
        self.current_ch_index = ch_index
        self.processed_data = raw.copy()

        self.rhythm_curves = {}
        if self.peak_scatter:
            self.plot_item.removeItem(self.peak_scatter)
            self.peak_scatter = None

        # Reset state for new data load
        self.filter_state = FilterState()
        self.analysis_state = AnalysisState()

        ch_name = raw.ch_names[ch_index] if raw.ch_names else f"Ch{ch_index}"
        self.label.setText(f"View {self.index + 1}: {filename} - {ch_name}")
        self.update_plot()

        # Ensure signal is visible immediately
        self.plot_item.autoRange()

    def apply_processing(self, params: dict):
        if self.raw_data is None: return

        # Update State
        self.filter_state.notch = params.get('notch', False)
        self.filter_state.detrend = params.get('detrend', False)
        self.filter_state.l_freq = params.get('l_freq', 1.0)
        self.filter_state.h_freq = params.get('h_freq', 40.0)

        temp = self.raw_data.copy()
        if self.filter_state.detrend:
            temp = SignalProcessor.detrend_signal(temp)
        if self.filter_state.notch:
            temp = SignalProcessor.apply_notch(temp, np.array([50.0]))

        l_freq = self.filter_state.l_freq
        h_freq = self.filter_state.h_freq
        if l_freq and h_freq:
            temp = SignalProcessor.apply_filter(temp, l_freq, h_freq)

        self.processed_data = temp
        self.update_plot()
        self.plot_item.autoRange()

    def show_spectrum(self):
        if self.processed_data is None: return
        try:
            data = self.processed_data.get_data()[self.current_ch_index]
            sfreq = self.processed_data.info['sfreq']
            self.spectrum_win = SpectrumWindow(data, sfreq, self)
            self.spectrum_win.show()
        except Exception as e:
            print(f"Error showing spectrum: {e}")

    def toggle_rhythm(self, rhythm_type: str, enabled: bool):
        # Update State
        if hasattr(self.analysis_state, rhythm_type):
            setattr(self.analysis_state, rhythm_type, enabled)

        if self.processed_data is None: return

        if rhythm_type in self.rhythm_curves:
            self.plot_item.removeItem(self.rhythm_curves[rhythm_type])
            del self.rhythm_curves[rhythm_type]

        if not enabled:
            if not self.rhythm_curves:
                _, unit = self._get_scale_and_unit()
                self.plot_item.setLabel('left', f"Amplitude ({unit})")
            return

        band = RhythmBands.get_band(rhythm_type)
        if not band: return

        rhythm_raw = self.processed_data.copy()
        try:
            rhythm_raw.filter(band.low, band.high, verbose=False)
        except Exception as e:
            print(f"Filter error for {rhythm_type}: {e}")
            return

        scale, unit = self._get_scale_and_unit()
        data = rhythm_raw.get_data()[self.current_ch_index] * scale
        times = rhythm_raw.times

        curve = self.plot_item.plot(times, data, pen=pg.mkPen(band.color, width=2))
        self.rhythm_curves[rhythm_type] = curve

        self.plot_item.setLabel('left', f"Amplitude ({unit})")
        self.plot_item.autoRange()

    def toggle_peaks(self, enabled: bool):
        # Update State
        self.analysis_state.peaks = enabled

        if self.peak_scatter:
            self.plot_item.removeItem(self.peak_scatter)
            self.peak_scatter = None

        if not enabled:
            if self.processed_data is not None:
                self.plot_item.autoRange()
            return

        if self.processed_data is None: return

        if self.main_curve is None:
            self.update_plot()

        data = self.processed_data.get_data()[self.current_ch_index]
        times = self.processed_data.times
        scale, unit = self._get_scale_and_unit()

        if unit == "ADC":
            data_scaled = data - np.mean(data)
        else:
            data_scaled = data * scale

        height = np.std(data_scaled) * 2
        peaks, _ = SignalProcessor.detect_peaks(data_scaled, height=height, distance=50)

        if len(peaks) > 0:
            self.peak_scatter = pg.ScatterPlotItem(
                x=times[peaks],
                y=data_scaled[peaks],
                pen='r', brush='r', size=12,
                hoverable=True,
                hoverPen='w', hoverBrush='b'
            )
            self.plot_item.addItem(self.peak_scatter)

        self.plot_item.autoRange()

    def on_mouse_move(self, evt):
        pos = evt[0]
        if self.plot_item.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_item.plotItem.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()

            _, unit = self._get_scale_and_unit()

            is_peak = False
            peak_val = 0.0

            if self.peak_scatter is not None:
                local_pos = self.peak_scatter.mapFromScene(pos)
                points = self.peak_scatter.pointsAt(local_pos)
                if len(points) > 0:
                    is_peak = True
                    peak_val = points[0].pos().y()

            view_pos = self.plot_item.mapFromScene(pos)
            global_pos = self.plot_item.mapToGlobal(view_pos)

            if is_peak:
                QToolTip.showText(global_pos, f"PEAK DETECTED\nTime: {x_val:.3f} s\nValue: {peak_val:.2f} {unit}",
                                  self.plot_item)
            else:
                QToolTip.showText(global_pos, f"Time: {x_val:.3f} s\nAmp: {y_val:.2f} {unit}", self.plot_item)

    def apply_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if is_dark:
            self.plot_item.setBackground('#2b2b2b')
        else:
            self.plot_item.setBackground('w')
        self.update_plot()

    def update_plot(self):
        if self.processed_data is None: return

        self.plot_item.clear()
        self.rhythm_curves = {}
        self.peak_scatter = None

        data = self.processed_data.get_data()
        times = self.processed_data.times

        if self.current_ch_index < len(data):
            valid_data = data[self.current_ch_index]
            scale, unit = self._get_scale_and_unit()

            if unit == "ADC":
                valid_data = valid_data - np.mean(valid_data)
                self.plot_item.setLabel('left', "Amplitude (ADC Centered)")
            else:
                valid_data = valid_data * scale
                self.plot_item.setLabel('left', f"Amplitude ({unit})")

            pen_color = '#dddddd' if hasattr(self, 'is_dark') and self.is_dark else '#050505'
            self.main_curve = self.plot_item.plot(times, valid_data, pen=pen_color)
        else:
            self.label.setText("Error: Channel Index Out of Bounds")