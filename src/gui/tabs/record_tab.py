from typing import List, Dict, Optional, Tuple, Any
import time
import pandas as pd
import numpy as np

from PyQt6.QtCore import QThread, pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QFileDialog, QMessageBox,
    QLineEdit
)

from src.core.protocols import IProtocol, ProtocolFactory
from src.core.storage import IStorage
from src.core.buffer import ThreadSafeBuffer
from src.models.config import ProtocolConfig, SignalChannelConfig, ProtocolType, BaudRate
from src.gui.recording_plot_widget import RecordingPlotWidget
from src.core.reader import ReaderWorker


class RecordTabWidget(QWidget):

    def __init__(self, storage: IStorage, protocol_factory: ProtocolFactory, parent: QWidget = None):
        super().__init__(parent)

        self.storage = storage
        self.protocol_factory = protocol_factory
        self.data_buffer = ThreadSafeBuffer()
        self.reader_worker: Optional[ReaderWorker] = None
        self.reader_thread: Optional[QThread] = None
        
        # UI Elements
        self.plot_curves = None
        self.plot_widget = None
        self.save_button = None
        self.stop_button = None
        self.start_button = None
        self.window_spin = None
        self.refresh_spin = None
        self.channels_input = None
        self.baud_combo = None
        self.refresh_ports_btn = None
        self.port_combo = None
        self.protocol_combo = None

        self.full_data: List[Tuple[float, Dict[str, Any]]] = []
        self.plot_data: Dict[str, Dict[str, List[float]]] = {}
        self.start_time = 0

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self._update_plot)

        self.init_ui()
        self.connect_signals()
        self._refresh_ports()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Controls Panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_panel.setFixedWidth(350)

        # Connection Settings
        conn_group = QGroupBox("Configuration")
        conn_layout = QFormLayout()

        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems([p.value for p in ProtocolType])

        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("Refresh")
        port_layout = QHBoxLayout()
        port_layout.addWidget(self.port_combo, stretch=1)
        port_layout.addWidget(self.refresh_ports_btn)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems([str(b.value) for b in BaudRate])
        self.baud_combo.setCurrentText(str(BaudRate.B_115200.value))

        self.channels_input = QLineEdit("A0")
        self.channels_input.setPlaceholderText("e.g. A0,A1")

        conn_layout.addRow("Protocol:", self.protocol_combo)
        conn_layout.addRow("Port:", port_layout)
        conn_layout.addRow("Baudrate:", self.baud_combo)
        conn_layout.addRow("Channels:", self.channels_input)
        conn_group.setLayout(conn_layout)

        # Plot Settings
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout()
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(10, 1000)
        self.refresh_spin.setValue(50)
        self.refresh_spin.setSuffix(" ms")

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 60.0)
        self.window_spin.setValue(10.0)
        self.window_spin.setSuffix(" s")

        plot_layout.addRow("Refresh Rate:", self.refresh_spin)
        plot_layout.addRow("Window Size:", self.window_spin)
        plot_group.setLayout(plot_layout)

        # Control Buttons
        controls_group = QGroupBox("Control")
        controls_vlayout = QVBoxLayout()
        controls_hlayout = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        self.stop_button.setEnabled(False)
        controls_hlayout.addWidget(self.start_button)
        controls_hlayout.addWidget(self.stop_button)

        self.save_button = QPushButton("Save CSV")
        self.save_button.setEnabled(False)

        controls_vlayout.addLayout(controls_hlayout)
        controls_vlayout.addWidget(self.save_button)
        controls_group.setLayout(controls_vlayout)

        controls_layout.addWidget(conn_group)
        controls_layout.addWidget(plot_group)
        controls_layout.addWidget(controls_group)
        controls_layout.addStretch()

        # Plot Widget
        self.plot_widget = RecordingPlotWidget(title="Real-Time EEG")
        self.plot_widget.set_labels(left='Value (ADC)', bottom='Time (s)')
        self.plot_widget.get_plot_item().addLegend()
        self.plot_curves = {}

        main_layout.addWidget(controls_panel)
        main_layout.addWidget(self.plot_widget, stretch=1)

    def connect_signals(self):
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.save_button.clicked.connect(self.save_current_buffer)

    @pyqtSlot()
    def _refresh_ports(self):
        self.port_combo.clear()
        try:
            ports = IProtocol.list_available_ports()
            self.port_combo.addItems(ports)
            if 'simulator' not in ports:
                self.port_combo.addItem('simulator')
        except Exception as e:
            self.port_combo.addItem("Error")
            QMessageBox.critical(self, "Error", f"Failed to scan ports: {e}")

    @pyqtSlot()
    def start_acquisition(self):
        try:
            proto_config = ProtocolConfig(
                protocol=self.protocol_combo.currentText(),
                port=self.port_combo.currentText(),
                baudrate=int(self.baud_combo.currentText())
            )

            channels_str = self.channels_input.text().strip()
            if not channels_str:
                raise ValueError("No channels specified.")

            pin_names = [ch.strip() for ch in channels_str.split(',')]
            channels = [SignalChannelConfig(name=f'{pin}', arduino_pin=pin) for pin in pin_names]
            
            refresh_ms = self.refresh_spin.value()

        except Exception as e:
            QMessageBox.warning(self, "Configuration Error", f"Invalid parameters: {e}")
            return

        self.data_buffer.clear()
        self.full_data = []
        self._reset_plots(pin_names)
        self.start_time = time.time()

        try:
            protocol_impl = self.protocol_factory.create(proto_config)
            protocol_impl.configure(proto_config, pin_names)
        except Exception as e:
            QMessageBox.critical(self, "Protocol Error", f"Failed to create protocol: {e}")
            return

        self.reader_worker = ReaderWorker(protocol_impl, channels, self.data_buffer)
        self.reader_thread = QThread()
        self.reader_worker.moveToThread(self.reader_thread)

        self.reader_worker.error.connect(self._on_reader_error)
        self.reader_thread.started.connect(self.reader_worker.run)
        self.reader_thread.finished.connect(self.reader_thread.quit) # Clean up on finish

        self.reader_thread.start()
        self.plot_timer.start(refresh_ms)

        self._set_ui_recording(True)

    @pyqtSlot()
    def stop_acquisition(self):
        self.plot_timer.stop()

        if self.reader_worker:
            self.reader_worker.stop()
        if self.reader_thread:
            self.reader_thread.quit()
            self.reader_thread.wait(2000)

        self.reader_worker = None
        self.reader_thread = None

        self._set_ui_recording(False)

    def _reset_plots(self, pin_names: List[str]):
        self.plot_widget.clear()
        self.plot_curves = {}
        self.plot_data = {}

        colors = ['g', 'r', 'c', 'm', 'y', 'w'] 

        for i, pin_name in enumerate(pin_names):
            color = colors[i % len(colors)]
            self.plot_curves[pin_name] = self.plot_widget.plot_widget.plot(
                pen=color, name=pin_name
            )
            self.plot_data[pin_name] = {'time': [], 'value': []}

    @pyqtSlot()
    def _update_plot(self):
        new_data = self.data_buffer.pop_all()
        if not new_data:
            return

        self.full_data.extend(new_data)
        window_s = self.window_spin.value()
        current_time = time.time() - self.start_time

        for timestamp, values_dict in new_data:
            rel_time = timestamp - self.start_time
            if rel_time < 0: rel_time = 0
            
            for pin_name, value in values_dict.items():
                if pin_name in self.plot_data:
                    self.plot_data[pin_name]['time'].append(rel_time)
                    self.plot_data[pin_name]['value'].append(value)

        for pin_name, curve in self.plot_curves.items():
            data = self.plot_data[pin_name]
            
            times_np = np.array(data['time'])
            values_np = np.array(data['value'])

            min_time = current_time - window_s
            idx_start = np.searchsorted(times_np, min_time)

            x_data = times_np[idx_start:]
            y_data = values_np[idx_start:]
            
            k = max(1, len(x_data) // 1000) # Simple downsampling
            curve.setData(x=x_data[::k], y=y_data[::k])

        self.plot_widget.set_x_range(max(0, current_time - window_s), current_time + (window_s * 0.05))

    @pyqtSlot()
    def save_current_buffer(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not path:
            return

        try:
            if not self.full_data:
                QMessageBox.warning(self, "No Data", "No data to save.")
                return

            rows = []
            header_set = {'time'}
            for timestamp, values_dict in self.full_data:
                row = {'time': timestamp}
                row.update(values_dict)
                rows.append(row)
                header_set.update(values_dict.keys())

            header = sorted(list(header_set))
            # Convert dicts to lists for DataFrame, or better just use dicts list
            # We used List[List[Any]] in interface, but let's adapt usage or storage
            # My PandasStorage uses DataFrame so list of dicts is also fine if I adjust logic or pass list of lists
            
            # Let's clean up for storage interface which asks for List[List[Any]]
            data_lists = []
            for row_dict in rows:
                data_lists.append([row_dict.get(h, None) for h in header])

            self.storage.save_csv(path, header, data_lists)
            QMessageBox.information(self, "Success", f"Saved to {path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")

    @pyqtSlot(str)
    def _on_reader_error(self, error_msg):
        QMessageBox.critical(self, "Reader Error", error_msg)
        self.stop_acquisition()

    def _set_ui_recording(self, recording: bool):
        self.start_button.setEnabled(not recording)
        self.stop_button.setEnabled(recording)
        self.save_button.setEnabled(not recording and bool(self.full_data))
        self.protocol_combo.setEnabled(not recording)
        self.port_combo.setEnabled(not recording)
        self.refresh_ports_btn.setEnabled(not recording)
        self.baud_combo.setEnabled(not recording)
        self.channels_input.setEnabled(not recording)

    def shutdown(self):
        self.stop_acquisition()
