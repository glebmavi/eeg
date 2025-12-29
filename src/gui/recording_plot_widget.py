from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from typing import Dict, List


class RecordingPlotWidget(QWidget):
    """
    Lightweight plot widget for real-time recording.
    """
    def __init__(self, title: str = "Signal"):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setLabel('left', "Value", units='ADC')
        
        self.layout.addWidget(self.plot_widget)
        self.curves: Dict[str, pg.PlotDataItem] = {}

    def get_plot_item(self):
        return self.plot_widget.getPlotItem()

    def set_labels(self, left: str, bottom: str):
        self.plot_widget.setLabel('left', left)
        self.plot_widget.setLabel('bottom', bottom)

    def clear(self):
        self.plot_widget.clear()
        self.curves.clear()

    def set_x_range(self, min_x, max_x):
        self.plot_widget.setXRange(min_x, max_x, padding=0)

    def update_curves(self, data: Dict[str, Dict[str, List[float]]], window_s: float, current_time: float):
        # NOT used currently if logic is in RecordTab, but helper method if needed.
        pass
