from PyQt6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np
import scipy.signal
import mne

class SpectrumWindow(QDialog):
    def __init__(self, data: np.ndarray, sfreq: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectral Analysis")
        self.resize(900, 600)

        self.data = data
        self.sfreq = sfreq

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Method Selector
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Power Spectral Density (Welch FFT)", "Time-Frequency (Morlet Wavelet)"])
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

        self.update_plot(0)

    def update_plot(self, index):
        self.plot_widget.clear()

        if index == 0:
            # --- Welch PSD ---
            self.plot_widget.setLabel('left', "Power Spectral Density (V**2/Hz)")
            self.plot_widget.setLabel('bottom', "Frequency (Hz)")

            freqs, psd = scipy.signal.welch(self.data, fs=self.sfreq, nperseg=min(len(self.data), int(self.sfreq * 4)))
            self.plot_widget.plot(freqs, psd, pen=pg.mkPen('b', width=2))
            self.plot_widget.autoRange()

        elif index == 1:
            # --- Morlet Wavelet (Time-Frequency) ---
            self.plot_widget.setLabel('left', "Frequency (Hz)")
            self.plot_widget.setLabel('bottom', "Time (s)")

            # Define frequencies of interest (e.g., 1 to 50 Hz)
            freqs = np.arange(1, 50, 1)
            n_cycles = freqs / 2.  # Variable cycles

            # Use MNE's tfr_array_morlet
            # Input needs to be (n_epochs, n_channels, n_times) -> (1, 1, n_times)
            data_reshaped = self.data[np.newaxis, np.newaxis, :]

            power = mne.time_frequency.tfr_array_morlet(
                data_reshaped, self.sfreq, freqs, n_cycles=n_cycles, output='power'
            )
            # Power shape: (1, 1, n_freqs, n_times) -> squeeze to (n_freqs, n_times)
            spectrogram = power[0, 0, :, :]

            # Log scale for better visualization
            spectrogram = np.log10(spectrogram + 1e-15)

            # Create ImageItem
            img = pg.ImageItem()
            img.setImage(spectrogram.T)  # Transpose for pyqtgraph (x=time, y=freq)

            # Scale axes
            # x scale: 1 sample = 1/sfreq seconds
            # y scale: 1 index = 1 Hz (since np.arange(1, 50, 1))
            tr = pg.QtGui.QTransform()
            tr.scale(1.0 / self.sfreq, 1.0)
            tr.translate(0, 1.0)  # Start freq at 1
            img.setTransform(tr)

            # Color Map
            colormap = pg.colormap.get('viridis')
            img.setLookupTable(colormap.getLookupTable())

            self.plot_widget.addItem(img)
            self.plot_widget.autoRange()