import os
from PyQt6.QtWidgets import QApplication

class ThemeManager:
    """
    Manages application themes using QSS stylesheets.
    """
    THEME_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "styles")

    @staticmethod
    def apply_theme(app: QApplication, theme_name: str = "dark"):
        """
        Apply a theme to the application.
        
        Args:
            app (QApplication): The application instance.
            theme_name (str): Name of the theme (e.g., 'dark', 'light').
        """
        qss_path = os.path.join(ThemeManager.THEME_DIR, f"{theme_name}.qss")
        
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                qss = f.read()
                app.setStyleSheet(qss)
        else:
            print(f"Warning: Theme file not found at {qss_path}")
            # Fallback or clear
            app.setStyleSheet("")
