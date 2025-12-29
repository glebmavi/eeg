from PyQt6.QtWidgets import QMainWindow, QTabWidget, QApplication
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import sys

from src.gui.tabs.record_tab import RecordTabWidget
from src.gui.tabs.analyze_tab import AnalysisTabWidget
from src.gui.theme_manager import ThemeManager
from src.core.protocols import ProtocolFactory
from src.core.storage import PandasStorage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVisor - EEG Analysis Environment")
        self.resize(1600, 900)

        # Initialize Services
        self.storage = PandasStorage()
        self.protocol_factory = ProtocolFactory()

        # Central Widget: TabWidget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Tabs
        self.record_tab = RecordTabWidget(
            storage=self.storage, 
            protocol_factory=self.protocol_factory
        )
        self.analysis_tab = AnalysisTabWidget()

        self.tab_widget.addTab(self.record_tab, "🔴 Recording")
        self.tab_widget.addTab(self.analysis_tab, "📊 Analysis")

        # Toolbar
        self.toolbar = self.addToolBar("Main Toolbar")
        
        # Load Data Button (Delegate to Analysis Tab)
        self.action_load = QAction("Load Data", self)
        self.action_load.triggered.connect(self.analysis_tab.load_data_file)
        self.toolbar.addAction(self.action_load)

        # Dark Mode Action
        self.action_theme = QAction("Dark Mode", self)
        self.action_theme.setCheckable(True)
        self.action_theme.setChecked(True) # Default Dark
        self.action_theme.toggled.connect(self.toggle_theme)
        self.toolbar.addAction(self.action_theme)

        # Apply initial theme
        self.toggle_theme(True)

    def toggle_theme(self, is_dark):
        app = QApplication.instance()
        theme = "dark" if is_dark else "light"
        ThemeManager.apply_theme(app, theme)
        
        self.analysis_tab.apply_theme(is_dark)

    def closeEvent(self, event):
        self.record_tab.shutdown()
        self.analysis_tab.shutdown()
        super().closeEvent(event)