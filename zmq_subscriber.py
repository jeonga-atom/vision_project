from typing import List, Optional
import base64, zmq, numpy as np, cv2
from PySide6 import QtCore
from .thread_base import BaseSubscriberThread

class ZmqSubscriberThread(BaseSubscriberThread):
    frame_received = QtCore.Signal(object)
    
    def __init__(self, endpoints: List[str], context=None, parent=None):
        super().__init__(parent)
        self.endpoints = list(endpoints)
        self._ctx_owned = context is None
        self.context = context or zmq.Context(io_threads=1)
        
    def _decode_frame(self, payload_bytes: bytes) -> Optional[np.ndarray]:
        """프레임 디코딩 최적화"""
        if not payload_bytes:
            return None
            
        # Base64 디코딩 시도
        try:
            jpg_bytes = base64.b64decode(payload_bytes, validate=False)
            np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception:
            pass
        
        # 직접 바이너리 디코딩 시도
        try:
            np_arr = np.frombuffer(payload_bytes, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.emit_log(f"[ZMQ] Frame decode failed: {e}")
            return None
    
    def run(self):
        sockets = []
        poller = zmq.Poller()
        
        try:
            for ep in self.endpoints:   # 소켓 초기화
                try:
                    sock = self.context.socket(zmq.SUB)
                    sock.setsockopt(zmq.RCVHWM, 50)
                    sock.linger = 0
                    sock.connect(ep)
                    sock.setsockopt_string(zmq.SUBSCRIBE, "")
                    poller.register(sock, zmq.POLLIN)
                    sock.setsockopt(zmq.CONFLATE, 1)  # 최신 1개만 유지 (옵션)
                    sock.setsockopt(zmq.RCVHWM, 1)    # HWM 최소화
                    sockets.append(sock)
                    self.emit_log(f"[ZMQ] Connected to {ep}")
                except Exception as e:
                    self.emit_log(f"[ZMQ] Connection error {ep}: {e}")
                    continue
            
            while not self._stop:     # 메인 루프
                try:
                    events = dict(poller.poll(timeout=200))
                    if not events:
                        continue
                    
                    for sock in events:
                        if events[sock] & zmq.POLLIN:
                            try:
                                msg = sock.recv_multipart(flags=zmq.NOBLOCK)
                                payload = msg[-1] if msg else b""
                                
                                frame = self._decode_frame(payload)
                                if frame is not None:
                                    self.frame_received.emit(("camera", frame))
                                    
                            except zmq.Again:
                                continue
                            except Exception as e:
                                self.emit_log(f"[ZMQ] Receive error: {e}")
                                
                except Exception as e:
                    self.emit_log(f"[ZMQ] Poll error: {e}")
                    
        finally:
            self._cleanup_sockets(sockets, poller)
            self.emit_log("[ZMQ] Thread terminated")
    
    def _cleanup_sockets(self, sockets: List, poller: zmq.Poller):
        """소켓 정리"""
        for sock in sockets:
            try:
                poller.unregister(sock)
                sock.close(0)
            except:
                pass
        
        if self._ctx_owned:
            try:
                self.context.term()
            except:
                pass