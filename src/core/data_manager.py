import mne
import itertools


class DataManager:
    """Singleton for managing loaded EEG datasets."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance.signals = {}  # Dict[str, mne.io.BaseRaw]
            cls._instance.listeners = [] # List of callables
        return cls._instance

    def add_signal(self, name: str, raw: mne.io.BaseRaw) -> str:
        """Add a signal with automatic name deduplication."""
        base_name = name
        counter = itertools.count(1)
        
        while name in self.signals:
            name = f"{base_name} ({next(counter)})"
            
        self.signals[name] = raw
        self.notify_listeners()
        return name

    def get_signal(self, name: str) -> mne.io.BaseRaw | None:
        return self.signals.get(name)

    def get_signal_names(self) -> list[str]:
        return list(self.signals.keys())

    def remove_signal(self, name: str):
        if name in self.signals:
            del self.signals[name]
            self.notify_listeners()

    def add_listener(self, callback):
        self.listeners.append(callback)

    def get_all_channels(self) -> list[tuple[str, int, str]]:
        """
        Returns flattened list of all available channels:
        [(filename, channel_index, channel_name), ...]
        """
        channels = []
        for name, raw in self.signals.items():
            if raw.ch_names:
                for i, ch_name in enumerate(raw.ch_names):
                    channels.append((name, i, ch_name))
        return channels

    def notify_listeners(self):
        """Notify all registered listeners of data changes."""
        for callback in self.listeners:
            callback()
