import mne

class DataManager:
    """
    Singleton-like class or shared state to manage loaded datasets.
    Stores MNE Raw objects keyed by a unique identifier (filename or custom name).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance.signals = {}  # Dict[str, mne.io.BaseRaw]
            cls._instance.listeners = [] # List of callables to notify on change
        return cls._instance

    def add_signal(self, name: str, raw: mne.io.BaseRaw):
        """Add a new signal to the manager."""
        # Ensure unique name if needed
        base_name = name
        counter = 1
        while name in self.signals:
            name = f"{base_name} ({counter})"
            counter += 1
            
        self.signals[name] = raw
        self.notify_listeners()
        return name

    def get_signal(self, name: str) -> mne.io.BaseRaw:
        """Retrieve a signal by name."""
        return self.signals.get(name)

    def get_signal_names(self) -> list[str]:
        """Get list of all stored signal names."""
        return list(self.signals.keys())

    def remove_signal(self, name: str):
        if name in self.signals:
            del self.signals[name]
            self.notify_listeners()

    def add_listener(self, callback):
        """Register a callback to be called when signal list changes."""
        self.listeners.append(callback)

    def notify_listeners(self):
        names = self.get_signal_names()
        for callback in self.listeners:
            callback(names)
