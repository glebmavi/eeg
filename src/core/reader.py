import time
from typing import List
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.core.protocols import IProtocol
from src.core.buffer import IDataBuffer
from src.models.config import SignalChannelConfig


class ReaderWorker(QObject):
    """
    Worker that runs in a separate QThread.
    Reads data from IProtocol and pushes it to IDataBuffer.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, protocol: IProtocol, channels: List[SignalChannelConfig], buffer: IDataBuffer):
        super().__init__()
        self.protocol = protocol
        self.channels = channels
        self.buffer = buffer
        self._is_running = False

    @pyqtSlot()
    def run(self):
        try:
            self._is_running = True
            if not self.protocol.is_connected():
                self.protocol.connect()

            while self._is_running:
                # Read raw line "val1,val2\r\n"
                raw = self.protocol.read_raw()
                if raw:
                    try:
                        line = raw.decode('utf-8').strip()
                        parts = line.split(',')
                        
                        # Validate count
                        if len(parts) == len(self.channels):
                            current_time = time.time()
                            values = {}
                            for i, ch_config in enumerate(self.channels):
                                val_str = parts[i]
                                values[ch_config.name] = float(val_str)
                            
                            self.buffer.push(current_time, values)
                    except ValueError:
                        pass  # Ignore parse errors
                else:
                    # No data or closed
                    if not self.protocol.is_connected():
                        break
                    time.sleep(0.001)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.protocol.disconnect()
            self.finished.emit()

    def stop(self):
        self._is_running = False
