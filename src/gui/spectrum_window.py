from PyQt6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np
import scipy.signal
import mne
from src.core.processor import SignalProcessor
from src.models.types import RhythmBands

class SpectrumWindow(QDialog):
    def __init__(self, data: np.ndarray, sfreq: float, parent=None, initial_method: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Spectral Analysis")
        self.resize(900, 600)

        self.data = data
        self.sfreq = sfreq

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Method Selector
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Power Spectral Density (Welch FFT)", "Time-Frequency (Morlet Wavelet)", "Band Power Histogram"])
        self.combo_method.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(QLabel("Analysis Method:"))
        layout.addWidget(self.combo_method)

        # Plot Area
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        layout.addWidget(self.plot_widget)

        # Image Item for Spectrogram (Hidden initially)
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setVisible(False)

        self.combo_method.setCurrentIndex(initial_method)
        if initial_method == 0:
            self.update_plot(0)

    def update_plot(self, index):
        self.plot_widget.clear()

        if index == 0:
            # --- Welch PSD ---
            self.plot_widget.setLabel('left', "Power Spectral Density (V**2/Hz)")
            self.plot_widget.setLabel('bottom', "Frequency (Hz)")

            # Use consistent nperseg as Processor to match visual
            nperseg = min(len(self.data), int(self.sfreq * 4))
            freqs, psd = scipy.signal.welch(self.data, fs=self.sfreq, nperseg=nperseg)
            self.plot_widget.plot(freqs, psd, pen=pg.mkPen('b', width=2))
            self.plot_widget.autoRange()

        elif index == 1:
            # --- Morlet Wavelet (Time-Frequency) ---
            self.plot_widget.setLabel('left', "Frequency (Hz)")
            self.plot_widget.setLabel('bottom', "Time (s)")

            # Frequencies 1-50Hz
            freqs = np.arange(1, 50, 1)
            n_cycles = freqs / 2.  # Variable cycles

            data_reshaped = self.data[np.newaxis, np.newaxis, :]

            power = mne.time_frequency.tfr_array_morlet(
                data_reshaped, self.sfreq, freqs, n_cycles=n_cycles, output='power'
            )
            spectrogram = power[0, 0, :, :]

            # Log scale
            spectrogram = np.log10(spectrogram + 1e-15)

            # Create ImageItem
            img = pg.ImageItem()
            img.setImage(spectrogram.T)

            # Scale axes
            tr = pg.QtGui.QTransform()
            tr.scale(1.0 / self.sfreq, 1.0)
            tr.translate(0, 1.0)
            img.setTransform(tr)

            # Color Map
            colormap = pg.colormap.get('viridis')
            img.setLookupTable(colormap.getLookupTable())

            self.plot_widget.addItem(img)
            self.plot_widget.autoRange()

        elif index == 2:
            # --- Band Power Histogram ---
            # Correct label: This is now Integral Power (V^2), not Density
            self.plot_widget.setLabel('left', "Absolute Power (V**2)")
            self.plot_widget.setLabel('bottom', "Frequency Bands")

            features = SignalProcessor.extract_band_powers(self.data, self.sfreq)

            # Extract data and colors
            labels = []
            vals = []
            brushes = []

            for k, v in features.items():
                # k is "Delta (0.5-4.0Hz)"
                band_name = k.split(' ')[0]  # "Delta"
                labels.append(band_name)
                vals.append(v['absolute'])

                # Get Color
                band = RhythmBands.get_band(band_name)
                brushes.append(band.color if band else 'b')

            x = np.arange(len(vals))

            # Bar Graph
            bg = pg.BarGraphItem(x=x, height=vals, width=0.6, brushes=brushes)
            self.plot_widget.addItem(bg)

            # Custom Axis Labels
            ax = self.plot_widget.getAxis('bottom')
            ax.setTicks([list(zip(x, labels))])

            # Add Value Labels on top of bars
            for i, val in enumerate(vals):
                text = pg.TextItem(text=f"{val:.2e}", color='k', anchor=(0.5, 1))
                text.setPos(i, val)
                self.plot_widget.addItem(text)

            self.plot_widget.autoRange()