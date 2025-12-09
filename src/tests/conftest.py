import pytest
from PyQt6.QtWidgets import QApplication
import sys

@pytest.fixture(scope="session")
def qapp():
    """
    Fixture to ensure a QApplication exists for the entire test session.
    Required for any PyQt widgets.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app