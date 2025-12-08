from PyQt6.QtWidgets import QDialog, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import scipy.signal

class SpectrumWindow(QDialog):
    def __init__(self, data: np.ndarray, sfreq: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frequency Spectrum (FFT)")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', "Power Spectral Density (V**2/Hz)")
        self.plot_widget.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)
        
        self.plot_spectrum(data, sfreq)
        
    def plot_spectrum(self, data, sfreq):
        # Use Welch's method for smooth PSD estimation
        # data assumed to be 1D array of the channel
        freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=min(len(data), int(sfreq * 4)))
        
        # Log scale mainly for power? Or just plot linear. Standard is often linear or SemilogY.
        # Let's do linear for now but make pen distinct.
        self.plot_widget.plot(freqs, psd, pen=pg.mkPen('b', width=2))
