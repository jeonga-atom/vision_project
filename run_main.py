# ---------------------------
# main
# ---------------------------
import sys
from PySide6 import QtWidgets
from .main_window import MainWindow

def main():
    """메인 함수"""
    # 기본 엔드포인트 설정
    endpoints = [
        "tcp://192.168.31.72:5555",  # 메인 카메라
    ]
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(endpoints)
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()