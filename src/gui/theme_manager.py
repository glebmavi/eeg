import os
import sys

from PyQt6.QtWidgets import QApplication

class ThemeManager:
    """
    Manages application themes using QSS stylesheets.
    """

    @staticmethod
    def get_resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_path, relative_path)

    @staticmethod
    def apply_theme(app: QApplication, theme_name: str = "dark"):
        """
        Apply a theme to the application.
        
        Args:
            app (QApplication): The application instance.
            theme_name (str): Name of the theme (e.g., 'dark', 'light').
        """
        qss_path = ThemeManager.get_resource_path(os.path.join("gui", "styles", f"{theme_name}.qss"))
        
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                qss = f.read()
                app.setStyleSheet(qss)
        else:
            print(f"Warning: Theme file not found at {qss_path}")
            # Fallback or clear
            app.setStyleSheet("")
