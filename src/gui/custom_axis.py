# src/gui/custom_axis.py

import pyqtgraph as pg


class PaddedBottomAxis(pg.AxisItem):
    """
    Custom AxisItem that pads the bottom bounding box AND sets a minimum height
    to ensure the axis label is not cut off.
    """

    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)

    def boundingRect(self):
        rect = super().boundingRect()

        if self.orientation == 'bottom':
            rect.adjust(0, 0, 0, 20)

        return rect