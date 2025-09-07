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
        "tcp://127.0.0.1:5555",  # 메인 카메라
        # "tcp://127.0.0.1:5556",  # 보급상자 감지
        # "tcp://127.0.0.1:5558",  # 사람 감지
        # "tcp://127.0.0.1:5559",  # 완료 감지
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