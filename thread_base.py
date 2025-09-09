from PySide6 import QtCore

class BaseSubscriberThread(QtCore.QThread):
    """공통 구독자 스레드 베이스 클래스"""
    text_received = QtCore.Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def emit_log(self, message: str):
        """로그 메시지 발송"""
        self.text_received.emit(message)