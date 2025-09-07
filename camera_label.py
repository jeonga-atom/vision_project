import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import cv2

class CameraLabel(QtWidgets.QLabel):
    def __init__(self, placeholder_text="No Image", parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(placeholder_text)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #111; color: #eee;")
        self._current_pixmap = None
        self._last_size = self.size()
        
    def show_frame(self, frame_bgr: np.ndarray):
        """프레임 표시 최적화"""
        if not self._is_valid_frame(frame_bgr):
            self._show_no_image()
            return
        
        try:
            qimg = self._convert_frame_to_qimage(frame_bgr)
            if qimg.isNull():
                self._show_no_image()
                return
            
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self._current_pixmap = pixmap
            self._update_display()
            
        except Exception as e:
            print(f"[CameraLabel] Error: {e}", file=sys.stderr)
            self._show_no_image()
    
    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """프레임 유효성 검사"""
        return (frame is not None and 
                isinstance(frame, np.ndarray) and 
                frame.size > 0)
    
    def _convert_frame_to_qimage(self, frame_bgr: np.ndarray) -> QtGui.QImage:
        """BGR 프레임을 QImage로 변환"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
    
    def _show_no_image(self):
        """이미지 없음 표시"""
        self.clear()
        self.setText("No Image")
        self._current_pixmap = None
    
    def _update_display(self):
        if self._current_pixmap and not self._current_pixmap.isNull():
            if self.size() != self._last_size:
                self._scaled_pixmap = self._current_pixmap.scaled(
                    self.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation)
                self._last_size = self.size()
            self.setPixmap(getattr(self, "_scaled_pixmap", self._current_pixmap))
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        self._last_size = QtCore.QSize(0,0) # 다음 _update_display에서 재스케일 강제
        self._update_display()
        super().resizeEvent(event)