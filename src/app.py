import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.gui.theme_manager import ThemeManager


def main() -> None:
    app = QApplication(sys.argv)
    ThemeManager.apply_theme(app, "dark")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
