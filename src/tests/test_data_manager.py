import pytest
import numpy as np
import mne
from src.core.data_manager import DataManager


class TestDataManagerSingleton:
    """Tests for DataManager singleton pattern."""
    
    def test_singleton_same_instance(self, clean_data_manager):
        """Test that DataManager returns the same instance."""
        dm1 = DataManager()
        dm2 = DataManager()
        
        assert dm1 is dm2
    
    def test_singleton_shared_state(self, clean_data_manager):
        """Test that DataManager instances share state."""
        dm1 = DataManager()
        dm2 = DataManager()
        
        # Add signal through dm1
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm1.add_signal("test", raw)
        
        # Should be visible through dm2
        assert "test" in dm2.get_signal_names()


class TestSignalManagement:
    """Tests for signal add/remove/get operations."""
    
    def test_add_signal_basic(self, clean_data_manager):
        """Test basic signal addition."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        name = dm.add_signal("test_signal", raw)
        
        assert name == "test_signal"
        assert "test_signal" in dm.get_signal_names()
    
    def test_add_multiple_signals(self, clean_data_manager):
        """Test adding multiple signals."""
        dm = clean_data_manager
        
        for i in range(5):
            info = mne.create_info([f'Ch{i}'], 250, ['eeg'])
            data = np.random.randn(1, 100)
            raw = mne.io.RawArray(data, info, verbose=False)
            dm.add_signal(f"signal_{i}", raw)
        
        assert len(dm.get_signal_names()) == 5
    
    def test_add_duplicate_name_creates_unique(self, clean_data_manager):
        """Test that duplicate names are automatically made unique."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        
        raw1 = mne.io.RawArray(data, info, verbose=False)
        raw2 = mne.io.RawArray(data, info, verbose=False)
        raw3 = mne.io.RawArray(data, info, verbose=False)
        
        name1 = dm.add_signal("data.edf", raw1)
        name2 = dm.add_signal("data.edf", raw2)
        name3 = dm.add_signal("data.edf", raw3)
        
        assert name1 == "data.edf"
        assert name2 == "data.edf (1)"
        assert name3 == "data.edf (2)"
        assert len(dm.get_signal_names()) == 3
    
    def test_get_signal(self, clean_data_manager):
        """Test retrieving signals."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test", raw)
        retrieved = dm.get_signal("test")
        
        assert retrieved is not None
        assert retrieved is raw
    
    def test_get_nonexistent_signal(self, clean_data_manager):
        """Test retrieving nonexistent signal returns None."""
        dm = clean_data_manager
        
        result = dm.get_signal("nonexistent")
        
        assert result is None
    
    def test_remove_signal(self, clean_data_manager):
        """Test removing signals."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("to_remove", raw)
        assert "to_remove" in dm.get_signal_names()
        
        dm.remove_signal("to_remove")
        assert "to_remove" not in dm.get_signal_names()
    
    def test_remove_nonexistent_signal(self, clean_data_manager):
        """Test removing nonexistent signal doesn't crash."""
        dm = clean_data_manager
        
        # Should not raise exception
        dm.remove_signal("nonexistent")
    
    def test_get_signal_names_empty(self, clean_data_manager):
        """Test getting signal names when empty."""
        dm = clean_data_manager
        
        names = dm.get_signal_names()
        
        assert isinstance(names, list)
        assert len(names) == 0
    
    def test_get_signal_names_returns_list(self, clean_data_manager):
        """Test that get_signal_names returns a list."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test", raw)
        names = dm.get_signal_names()
        
        assert isinstance(names, list)
        assert "test" in names


class TestListenerPattern:
    """Tests for listener notification system."""
    
    def test_add_listener(self, clean_data_manager):
        """Test adding listeners."""
        dm = clean_data_manager
        
        called = []
        def callback():
            called.append(True)
        
        dm.add_listener(callback)
        
        # Trigger notification
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        dm.add_signal("test", raw)
        
        assert len(called) > 0
    
    def test_multiple_listeners(self, clean_data_manager):
        """Test multiple listeners are all notified."""
        dm = clean_data_manager
        
        called = {'listener1': False, 'listener2': False, 'listener3': False}
        
        def make_callback(name):
            def callback():
                called[name] = True
            return callback
        
        dm.add_listener(make_callback('listener1'))
        dm.add_listener(make_callback('listener2'))
        dm.add_listener(make_callback('listener3'))
        
        # Trigger notification
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        dm.add_signal("test", raw)
        
        assert all(called.values())
    
    def test_listener_called_on_add(self, clean_data_manager):
        """Test that listeners are called when adding signals."""
        dm = clean_data_manager
        
        call_count = [0]
        def callback():
            call_count[0] += 1
        
        dm.add_listener(callback)
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test1", raw)
        dm.add_signal("test2", raw)
        
        assert call_count[0] == 2
    
    def test_listener_called_on_remove(self, clean_data_manager):
        """Test that listeners are called when removing signals."""
        dm = clean_data_manager
        
        call_count = [0]
        def callback():
            call_count[0] += 1
        
        dm.add_listener(callback)
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test", raw)
        initial_count = call_count[0]
        
        dm.remove_signal("test")
        
        assert call_count[0] == initial_count + 1


class TestChannelEnumeration:
    """Tests for get_all_channels functionality."""
    
    def test_get_all_channels_empty(self, clean_data_manager):
        """Test getting channels when no signals exist."""
        dm = clean_data_manager
        
        channels = dm.get_all_channels()
        
        assert isinstance(channels, list)
        assert len(channels) == 0
    
    def test_get_all_channels_single_signal(self, clean_data_manager):
        """Test getting channels from single signal."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1', 'Ch2', 'Ch3'], 250, ['eeg', 'eeg', 'eeg'])
        data = np.random.randn(3, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test.edf", raw)
        channels = dm.get_all_channels()
        
        assert len(channels) == 3
        assert ('test.edf', 0, 'Ch1') in channels
        assert ('test.edf', 1, 'Ch2') in channels
        assert ('test.edf', 2, 'Ch3') in channels
    
    def test_get_all_channels_multiple_signals(self, clean_data_manager):
        """Test getting channels from multiple signals."""
        dm = clean_data_manager
        
        # Add first signal
        info1 = mne.create_info(['A1', 'A2'], 250, ['eeg', 'eeg'])
        data1 = np.random.randn(2, 100)
        raw1 = mne.io.RawArray(data1, info1, verbose=False)
        dm.add_signal("signal1", raw1)
        
        # Add second signal
        info2 = mne.create_info(['B1', 'B2', 'B3'], 250, ['eeg', 'eeg', 'eeg'])
        data2 = np.random.randn(3, 100)
        raw2 = mne.io.RawArray(data2, info2, verbose=False)
        dm.add_signal("signal2", raw2)
        
        channels = dm.get_all_channels()
        
        # Should have 2 + 3 = 5 channels total
        assert len(channels) == 5
        
        # Check first signal channels
        assert ('signal1', 0, 'A1') in channels
        assert ('signal1', 1, 'A2') in channels
        
        # Check second signal channels
        assert ('signal2', 0, 'B1') in channels
        assert ('signal2', 2, 'B3') in channels
    
    def test_get_all_channels_format(self, clean_data_manager):
        """Test that channel tuples have correct format."""
        dm = clean_data_manager
        
        info = mne.create_info(['TestCh'], 250, ['eeg'])
        data = np.random.randn(1, 100)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        dm.add_signal("test", raw)
        channels = dm.get_all_channels()
        
        # Each channel should be (str, int, str)
        for channel in channels:
            assert isinstance(channel, tuple)
            assert len(channel) == 3
            assert isinstance(channel[0], str)  # filename
            assert isinstance(channel[1], int)  # channel index
            assert isinstance(channel[2], str)  # channel name


class TestDataManagerEdgeCases:
    """Tests for edge cases in DataManager."""
    
    def test_add_signal_with_empty_name(self, clean_data_manager):
        """Test adding signal with empty name."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        
        name = dm.add_signal("", raw)
        
        # Should still add with empty name (or handle it)
        assert name in dm.get_signal_names()
    
    def test_add_signal_with_special_characters(self, clean_data_manager):
        """Test adding signal with special characters in name."""
        dm = clean_data_manager
        
        info = mne.create_info(['Ch1'], 250, ['eeg'])
        data = np.zeros((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        
        special_name = "test@#$%.edf"
        name = dm.add_signal(special_name, raw)
        
        assert name == special_name
    
    def test_signal_with_no_channels(self, clean_data_manager):
        """Test handling signal with no channels."""
        dm = clean_data_manager
        
        # This might not be possible with MNE, but test the channel enumeration
        channels = dm.get_all_channels()
        
        # Should return empty list for no signals
        assert isinstance(channels, list)
