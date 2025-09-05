import sys
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from PySide6 import QtCore, QtGui, QtWidgets
import zmq
import base64
import numpy as np
import cv2

# ---------------------------
# Configuration & Constants
# ---------------------------
@dataclass
class ButtonConfig:
    """버튼 설정 클래스"""
    flag_name: str
    button_name: str
    text_idle: str
    text_active: str
    button_type: str = 'detection'  # 'detection' or 'robot'

@dataclass
class UIStyles:
    """UI 스타일 상수"""
    DET_IDLE: str = "color: white; background: darkred;"
    DET_ACTIVE: str = "color: black; background: yellow;"
    ROBOT_IDLE: str = "color: black; background: white; border:1px solid #8b1f1f; padding:6px;"
    ROBOT_ACTIVE: str = "color: white; background: #2ecc71; border:1px solid #1e8449; padding:6px;"

class MissionType(Enum):
    MISSION1 = 0
    MISSION2 = 1  
    MISSION3 = 2
    MISSION4 = 3

# 설정 집중화
UI_STYLES = UIStyles()

BUTTON_CONFIGS = {
    'SOS': ButtonConfig('sos_flag', 'btn_m1_sos', 'SOS: NO', 'SOS: YES'),
    'Button': ButtonConfig('button_flag', 'btn_m3_button', 'BUTTON: NO', 'BUTTON: YES'),
    'Fire': ButtonConfig('fire_flag', 'btn_m3_fire', 'FIRE: NO', 'FIRE: YES'),
    'Door': ButtonConfig('door_flag', 'btn_m3_door', 'DOOR: NO', 'DOOR: YES'),
    'Safebox': ButtonConfig('safebox_flag', 'btn_m4_safebox', 'SAFEBOX: NO', 'SAFEBOX: YES'),
    'Human': ButtonConfig('human_flag', 'btn_m4_human', 'HUMAN: NO', 'HUMAN: YES'),
    'Finish': ButtonConfig('finish_flag', 'btn_m4_finish', 'FINISH: NO', 'FINISH: YES'),
    'OK': ButtonConfig('robot_ok_flag', 'btn_ok', 'OK', 'OK', 'robot'),
    'Open': ButtonConfig('robot_open_flag', 'btn_m3_open', 'Open', 'Open', 'robot'),
    'Pick': ButtonConfig('robot_pick_flag', 'btn_m4_pick', 'Pick', 'Pick', 'robot'),
}

DEFAULT_ROS_TOPICS = {
    'ok': '/robot/ok',
    'open': '/robot/open',
    'pick': '/robot/pick'
}

# ---------------------------
# 공통 베이스 클래스
# ---------------------------
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

# ---------------------------
# ZMQ 구독자 (최적화)
# ---------------------------
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
            # 소켓 초기화
            for ep in self.endpoints:
                try:
                    sock = self.context.socket(zmq.SUB)
                    sock.setsockopt(zmq.RCVHWM, 50)
                    sock.linger = 0
                    sock.connect(ep)
                    sock.setsockopt_string(zmq.SUBSCRIBE, "")
                    poller.register(sock, zmq.POLLIN)
                    sockets.append(sock)
                    self.emit_log(f"[ZMQ] Connected to {ep}")
                except Exception as e:
                    self.emit_log(f"[ZMQ] Connection error {ep}: {e}")
                    continue
            
            # 메인 루프
            while not self._stop:
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

# ---------------------------
# ROS2 구독자 (최적화)
# ---------------------------
class Ros2SubscriberThread(BaseSubscriberThread):
    # 탐지 시그널들
    sos_detected = QtCore.Signal()
    button_detected = QtCore.Signal()
    fire_detected = QtCore.Signal()
    door_detected = QtCore.Signal()
    safebox_detected = QtCore.Signal()
    human_detected = QtCore.Signal()
    finish_detected = QtCore.Signal()
    
    # 로봇 상태 시그널들
    robot_ok = QtCore.Signal()
    robot_open = QtCore.Signal()
    robot_pick = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._node = None
        self._subs = []
        
    def _create_subscriptions(self):
        """구독 생성 (중복 코드 제거)"""
        # 탐지 토픽들
        detection_topics = {
            '/object_detection/SOS': self.sos_detected,
            '/object_detection/button': self.button_detected,
            '/object_detection/fire': self.fire_detected,
            '/object_detection/door': self.door_detected,
            '/object_detection/safebox': self.safebox_detected,
            '/object_detection/human': self.human_detected,
            '/object_detection/finish': self.finish_detected,
        }
        
        # 로봇 상태 토픽들
        robot_topics = {
            '/robot/ok': self.robot_ok,
            '/robot/open': self.robot_open,
            '/robot/pick': self.robot_pick,
        }
        
        # 모든 토픽 구독
        all_topics = {**detection_topics, **robot_topics}
        
        for topic, signal in all_topics.items():
            try:
                from std_msgs.msg import Empty
                sub = self._node.create_subscription(
                    Empty, topic,
                    lambda msg, sig=signal: sig.emit(),
                    10
                )
                self._subs.append(sub)
                self.emit_log(f"[ROS2] Subscribed to {topic}")
            except Exception as e:
                self.emit_log(f"[ROS2] Subscription error {topic}: {e}")
    
    def run(self):
        try:
            import rclpy
            from std_msgs.msg import Empty
        except ImportError as e:
            self.emit_log(f"[ROS2] Import error: {e}")
            return
        
        try:
            # ROS2 초기화
            try:
                rclpy.init(args=None)
            except:
                pass  # 이미 초기화된 경우
            
            self._node = rclpy.create_node('ui_subscriber')
            self._create_subscriptions()
            
            # 메인 루프
            while not self._stop and rclpy.ok():
                try:
                    rclpy.spin_once(self._node, timeout_sec=0.1)
                except Exception as e:
                    self.emit_log(f"[ROS2] Spin error: {e}")
                    
        except Exception as e:
            self.emit_log(f"[ROS2] Runtime error: {e}")
        finally:
            self._cleanup()
            self.emit_log("[ROS2] Thread terminated")
    
    def _cleanup(self):
        """리소스 정리"""
        if self._node:
            try:
                self._node.destroy_node()
            except:
                pass
        
        try:
            import rclpy
            rclpy.shutdown()
        except:
            pass

# ---------------------------
# 최적화된 카메라 레이블
# ---------------------------
class CameraLabel(QtWidgets.QLabel):
    def __init__(self, placeholder_text="No Image", parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(placeholder_text)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #111; color: #eee;")
        self._current_pixmap = None
        
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
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    
    def _show_no_image(self):
        """이미지 없음 표시"""
        self.clear()
        self.setText("No Image")
        self._current_pixmap = None
    
    def _update_display(self):
        """디스플레이 업데이트"""
        if self._current_pixmap and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.size(), 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        self._update_display()
        super().resizeEvent(event)

# ---------------------------
# UI 상태 관리자
# ---------------------------
class StateManager:
    """UI 상태 관리 클래스"""
    
    def __init__(self):
        # 감지 플래그들
        self.detection_flags = {
            'sos_flag': False,
            'button_flag': False, 
            'fire_flag': False,
            'door_flag': False,
            'safebox_flag': False,
            'human_flag': False,
            'finish_flag': False
        }
        
        # 로봇 상태 플래그들
        self.robot_flags = {
            'robot_ok_flag': False,
            # 'robot_press_flag': False,
            'robot_open_flag': False,
            'robot_pick_flag': False
        }
        
        # 카운터들
        self.counters = {
            'fire_count': 0,
            'human_count': 0
        }
    
    def set_flag(self, flag_name: str, value: bool):
        """플래그 설정"""
        if flag_name in self.detection_flags:
            self.detection_flags[flag_name] = value
        elif flag_name in self.robot_flags:
            self.robot_flags[flag_name] = value
    
    def get_flag(self, flag_name: str) -> bool:
        """플래그 조회"""
        return (self.detection_flags.get(flag_name, False) or 
                self.robot_flags.get(flag_name, False))
    
    def increment_counter(self, counter_name: str) -> int:
        """카운터 증가"""
        if counter_name in self.counters:
            self.counters[counter_name] += 1
            return self.counters[counter_name]
        return 0
    
    def reset_counter(self, counter_name: str):
        """카운터 리셋"""
        if counter_name in self.counters:
            self.counters[counter_name] = 0
    
    def reset_all(self):
        """모든 상태 리셋"""
        for key in self.detection_flags:
            self.detection_flags[key] = False
        for key in self.robot_flags:
            self.robot_flags[key] = False
        for key in self.counters:
            self.counters[key] = 0

# ---------------------------
# 메인 윈도우 (최적화)
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, zmq_endpoints: List[str], ros_topics=None):
        super().__init__()
        self.setWindowTitle("ZMQ + ROS2 Viewer (Optimized)")
        self.resize(1000, 700)
        
        # 상태 관리자
        self.state = StateManager()
        
        # 설정
        self.zmq_endpoints = zmq_endpoints
        self.ros_topics = ros_topics or DEFAULT_ROS_TOPICS
        
        # 스레드
        self.zmq_thread = None
        self.ros_thread = None
        
        # UI 초기화
        self._init_ui()
        self._connect_signals()
        
        # 종료 시 정리
        QtCore.QCoreApplication.instance().aboutToQuit.connect(self._cleanup)
    
    def _init_ui(self):
        """UI 초기화"""
        # 중앙 위젯
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # 왼쪽: 카메라 + 컨트롤
        left_widget = self._create_left_panel()
        
        # 오른쪽: 탭 + 미션 + 로그
        right_widget = self._create_right_panel()
        
        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_widget, stretch=1)
        self.setCentralWidget(central)
    
    def _create_left_panel(self) -> QtWidgets.QWidget:
        """왼쪽 패널 생성"""
        self.main_cam = CameraLabel("Main Camera")
        
        # 컨트롤 버튼들
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.main_cam, stretch=1)
        layout.addLayout(btn_layout)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget
    
    def _create_right_panel(self) -> QtWidgets.QWidget:
        """오른쪽 패널 생성"""
        layout = QtWidgets.QVBoxLayout()
        
        # 탭바
        self.tab_bar = self._create_tab_bar()
        layout.addWidget(self.tab_bar, alignment=QtCore.Qt.AlignLeft)
        
        # 미션 스택
        self.mission_stack = self._create_mission_stack()
        layout.addWidget(self.mission_stack, alignment=QtCore.Qt.AlignLeft)
        
        layout.addStretch()
        
        # 로그
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setMinimumWidth(300)
        self.log_list.setMaximumWidth(400)
        layout.addWidget(self.log_list)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setMinimumWidth(420)
        return widget
    
    def _create_tab_bar(self) -> QtWidgets.QTabBar:
        """탭바 생성"""
        tab_bar = QtWidgets.QTabBar()
        tab_bar.setExpanding(False)
        tab_bar.setDrawBase(False)
        tab_bar.setFixedHeight(34)
        
        for i in range(1, 5):
            tab_bar.addTab(f"mission{i}")
        
        return tab_bar
    
    def _create_mission_stack(self) -> QtWidgets.QStackedWidget:
        """미션 스택 생성"""
        stack = QtWidgets.QStackedWidget()
        stack.setFixedSize(336, 220)
        
        for mission_type in MissionType:
            page = self._create_mission_page(mission_type)
            stack.addWidget(page)
        
        return stack
    
    def _create_mission_page(self, mission_type: MissionType) -> QtWidgets.QWidget:
        """미션 페이지 생성"""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        if mission_type == MissionType.MISSION1:
            # Mission 1: SOS만
            widget = self._create_mission1_content()
            layout.addWidget(widget, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            
        elif mission_type == MissionType.MISSION3:
            # Mission 3: 감지 버튼들 + 로봇 상태
            left_widget = self._create_mission3_detection_buttons()
            right_widget = self._create_mission3_robot_buttons()
            
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.setSpacing(12)
            h_layout.addWidget(left_widget, alignment=QtCore.Qt.AlignTop)
            h_layout.addWidget(right_widget, alignment=QtCore.Qt.AlignTop)
            
            container = QtWidgets.QWidget()
            container.setLayout(h_layout)
            layout.addWidget(container, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            
        elif mission_type == MissionType.MISSION4:
            # Mission 4: 감지 버튼들 + 로봇 상태
            left_widget = self._create_mission4_detection_buttons()
            right_widget = self._create_mission4_robot_buttons()
            
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.setSpacing(12)
            h_layout.addWidget(left_widget, alignment=QtCore.Qt.AlignTop)
            h_layout.addWidget(right_widget, alignment=QtCore.Qt.AlignTop)
            
            container = QtWidgets.QWidget()
            container.setLayout(h_layout)
            layout.addWidget(container, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
        
        page.setLayout(layout)
        return page
    
    def _create_button(self, text: str, button_type: str = 'detection') -> QtWidgets.QPushButton:
        """공통 버튼 생성 함수"""
        btn = QtWidgets.QPushButton(text)
        btn.setFixedSize(150, 44)
        
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        btn.setFont(font)
        
        if button_type == 'detection':
            btn.setStyleSheet(UI_STYLES.DET_IDLE)
        else:  # robot
            btn.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        
        return btn
    
    def _create_mission1_content(self) -> QtWidgets.QWidget:
        """Mission 1 컨텐츠 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.btn_m1_sos = self._create_button("SOS: NO")
        self.btn_m1_sos.clicked.connect(lambda: self._reset_flag('sos_flag', self.btn_m1_sos, "SOS: NO"))
        
        layout.addWidget(self.btn_m1_sos, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission3_detection_buttons(self) -> QtWidgets.QWidget:
        """Mission 3 감지 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # 버튼들
        self.btn_m3_button = self._create_button("BUTTON: NO")
        self.btn_m3_fire = self._create_button("FIRE: NO")
        self.btn_m3_door = self._create_button("DOOR: NO")
        
        # 이벤트 연결
        self.btn_m3_button.clicked.connect(lambda: self._reset_flag('button_flag', self.btn_m3_button, "BUTTON: NO"))
        self.btn_m3_fire.clicked.connect(self._reset_fire_flag)
        self.btn_m3_door.clicked.connect(lambda: self._reset_flag('door_flag', self.btn_m3_door, "DOOR: NO"))
        
        layout.addWidget(self.btn_m3_button, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_fire, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_door, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        # 불 카운터 레이블
        self.lbl_m3_fire_count = QtWidgets.QLabel("불의 개수: 0 개 (0/3)")
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.lbl_m3_fire_count.setFont(font)
        self.lbl_m3_fire_count.setFixedSize(150, 44)
        self.lbl_m3_fire_count.setWordWrap(True)
        self.lbl_m3_fire_count.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_m3_fire_count, alignment=QtCore.Qt.AlignLeft)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission3_robot_buttons(self) -> QtWidgets.QWidget:
        """Mission 3 로봇 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.btn_m3_ok_1 = self._create_button("OK", 'robot')
        self.btn_m3_ok_2 = self._create_button("OK", 'robot')
        self.btn_m3_open = self._create_button("Open", 'robot')
        
        # 이벤트 연결
        self.btn_m3_ok_1.clicked.connect(lambda: self._reset_robot_flag('robot_ok_flag', [self.btn_m3_ok_1, getattr(self, 'btn_m4_ok', None)]))
        self.btn_m3_ok_2.clicked.connect(lambda: self._reset_robot_flag('robot_ok_flag', [self.btn_m3_ok_2]))
        self.btn_m3_open.clicked.connect(lambda: self._reset_robot_flag('robot_open_flag', [self.btn_m3_open]))
        
        layout.addWidget(self.btn_m3_ok_1, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_ok_2, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_open, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission4_detection_buttons(self) -> QtWidgets.QWidget:
        """Mission 4 감지 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(45)
        
        # 버튼들
        self.btn_m4_safebox = self._create_button("SAFEBOX: NO")
        self.btn_m4_human = self._create_button("HUMAN: NO")
        self.btn_m4_finish = self._create_button("FINISH: NO")
        
        # 이벤트 연결
        self.btn_m4_safebox.clicked.connect(lambda: self._reset_flag('safebox_flag', self.btn_m4_safebox, "SAFEBOX: NO"))
        self.btn_m4_human.clicked.connect(self._reset_human_flag)
        self.btn_m4_finish.clicked.connect(lambda: self._reset_flag('finish_flag', self.btn_m4_finish, "FINISH: NO"))
        
        layout.addWidget(self.btn_m4_safebox, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m4_human, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m4_finish, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        # 사람 카운터 레이블들
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        
        self.lbl_m4_human_count = QtWidgets.QLabel("사람: 0 명 (0/3)")
        self.lbl_m4_human_count.setFont(font)
        self.lbl_m4_human_count.setFixedSize(150, 44)
        self.lbl_m4_human_count.setWordWrap(True)
        self.lbl_m4_human_count.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_m4_human_count, alignment=QtCore.Qt.AlignLeft)
        
        self.lbl_m4_human_state = QtWidgets.QLabel("상태: 생존 / 사망")
        self.lbl_m4_human_state.setFont(font)
        self.lbl_m4_human_state.setFixedSize(150, 44)
        self.lbl_m4_human_state.setWordWrap(True)
        self.lbl_m4_human_state.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_m4_human_state, alignment=QtCore.Qt.AlignLeft)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission4_robot_buttons(self) -> QtWidgets.QWidget:
        """Mission 4 로봇 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.btn_m4_ok = self._create_button("OK", 'robot')
        self.btn_m4_pick = self._create_button("Pick", 'robot')
        
        # 이벤트 연결
        self.btn_m4_ok.clicked.connect(lambda: self._reset_robot_flag('robot_ok_flag', [getattr(self, 'btn_m3_ok_1', None), self.btn_m4_ok]))
        self.btn_m4_pick.clicked.connect(lambda: self._reset_robot_flag('robot_pick_flag', [self.btn_m4_pick]))
        
        layout.addWidget(self.btn_m4_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m4_pick, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _connect_signals(self):
        """시그널 연결"""
        # 컨트롤 버튼들
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        
        # 탭 동기화
        self.tab_bar.currentChanged.connect(self.mission_stack.setCurrentIndex)
        self.mission_stack.currentChanged.connect(self.tab_bar.setCurrentIndex)
        
        # 키보드 단축키
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """키보드 단축키 설정"""
        left_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        right_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        
        left_shortcut.activated.connect(self._go_prev_tab)
        right_shortcut.activated.connect(self._go_next_tab)
    
    # ---------------------------
    # 이벤트 핸들러들 (최적화)
    # ---------------------------
    def _reset_flag(self, flag_name: str, button: QtWidgets.QPushButton, idle_text: str):
        """공통 플래그 리셋 함수"""
        self.state.set_flag(flag_name, False)
        button.setText(idle_text)
        button.setStyleSheet(UI_STYLES.DET_IDLE)
        self._append_log(f"[UI] {flag_name} reset")
    
    def _reset_robot_flag(self, flag_name: str, buttons: List[QtWidgets.QPushButton]):
        """로봇 플래그 리셋 함수"""
        self.state.set_flag(flag_name, False)
        for btn in buttons:
            if btn is not None:
                btn.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        self._append_log(f"[UI] {flag_name} reset")
    
    def _reset_fire_flag(self):
        """불 플래그 특별 리셋"""
        self.state.set_flag('fire_flag', False)
        self.state.reset_counter('fire_count')
        self.btn_m3_fire.setText("FIRE: NO")
        self.btn_m3_fire.setStyleSheet(UI_STYLES.DET_IDLE)
        self.lbl_m3_fire_count.setText("불의 개수: 0 개 (0/3)")
        self._append_log("[UI] Fire flag and counter reset")
    
    def _reset_human_flag(self):
        """사람 플래그 특별 리셋"""
        self.state.set_flag('human_flag', False)
        self.state.reset_counter('human_count')
        self.btn_m4_human.setText("HUMAN: NO")
        self.btn_m4_human.setStyleSheet(UI_STYLES.DET_IDLE)
        self.lbl_m4_human_count.setText("사람: 0 명 (0/3)")
        self._append_log("[UI] Human flag and counter reset")
    
    def _go_next_tab(self):
        """다음 탭으로"""
        current = self.tab_bar.currentIndex()
        count = self.tab_bar.count()
        next_idx = (current + 1) % count
        self.tab_bar.setCurrentIndex(next_idx)
        self._append_log(f"[UI] Mission -> {next_idx + 1}")
    
    def _go_prev_tab(self):
        """이전 탭으로"""
        current = self.tab_bar.currentIndex()
        count = self.tab_bar.count()
        prev_idx = (current - 1) % count
        self.tab_bar.setCurrentIndex(prev_idx)
        self._append_log(f"[UI] Mission -> {prev_idx + 1}")
    
    @QtCore.Slot(str)
    def _append_log(self, message: str):
        """로그 추가 (스레드 안전)"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if self.log_list is not None:
                self.log_list.addItem(f"{timestamp} {message}")
                self.log_list.scrollToBottom()
            else:
                print(f"{timestamp} {message}")
        except Exception as e:
            print(f"[UI] Log error: {e}")
    
    # -------------------------------------------------------------
    # ROS2 콜백 함수들, flag를 설정 
    # -------------------------------------------------------------
     ################### mission1의 sos 인식 #######################
    @QtCore.Slot()
    def _on_sos_detected(self):
        """SOS 감지"""
        if not self.state.get_flag('sos_flag'):
            self.state.set_flag('sos_flag', True)
            self.btn_m1_sos.setText("SOS: YES")
            self.btn_m1_sos.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] SOS detected!")
    
    ################# mission3의 버튼, 불, 문 인식 ##################
    @QtCore.Slot() 
    def _on_button_detected(self):
        """버튼 감지"""
        if not self.state.get_flag('button_flag'):
            self.state.set_flag('button_flag', True)
            self.btn_m3_button.setText("BUTTON: YES")
            self.btn_m3_button.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Button detected!")
    
    @QtCore.Slot()
    def _on_fire_detected(self):
        """불 감지"""
        count = self.state.increment_counter('fire_count')
        self.lbl_m3_fire_count.setText(f"불의 개수: {count} 개 ({count}/3)")
        
        if not self.state.get_flag('fire_flag'):
            self.state.set_flag('fire_flag', True)
            self.btn_m3_fire.setText("FIRE: YES")
            self.btn_m3_fire.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Fire detected!")
    
    @QtCore.Slot()
    def _on_door_detected(self):
        """문 감지"""
        if not self.state.get_flag('door_flag'):
            self.state.set_flag('door_flag', True)
            self.btn_m3_door.setText("DOOR: YES")
            self.btn_m3_door.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Door detected!")
    
    ############ mission4의 보급상자, 사람, finish 라인 인식 ###########
    @QtCore.Slot()
    def _on_safebox_detected(self):
        """보급상자"""
        if not self.state.get_flag('safebox_flag'):
            self.state.set_flag('safebox_flag', True)
            self.btn_m4_safebox.setText("SAFEBOX: YES")
            self.btn_m4_safebox.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Safebox detected!")
    
    @QtCore.Slot()
    def _on_human_detected(self):
        """사람 감지"""
        count = self.state.increment_counter('human_count')
        self.lbl_m4_human_count.setText(f"사람: {count} 명 ({count}/3)")
        
        if not self.state.get_flag('human_flag'):
            self.state.set_flag('human_flag', True)
            self.btn_m4_human.setText("HUMAN: YES")
            self.btn_m4_human.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Human detected!")
    
    @QtCore.Slot()
    def _on_finish_detected(self):
        """완료 감지"""
        if not self.state.get_flag('finish_flag'):
            self.state.set_flag('finish_flag', True)
            self.btn_m4_finish.setText("FINISH: YES")
            self.btn_m4_finish.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Finish detected!")
    
    ############# 버튼 인식, 보급 상자 인식이 되면 터미널에 로그를 출력 ############
    @QtCore.Slot()
    def _on_robot_ok(self):
        """로봇 OK"""
        if not self.state.get_flag('robot_ok_flag'):
            self.state.set_flag('robot_ok_flag', True)
            if hasattr(self, 'btn_m3_ok_1'):
                self.btn_m3_ok_1.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            if hasattr(self, 'btn_m3_ok_2'):
                self.btn_m3_ok_2.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            if hasattr(self, 'btn_m4_ok'):
                self.btn_m4_ok.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] Robot OK!")
 
    @QtCore.Slot()
    def _on_robot_open(self):
        """로봇 Open"""
        if not self.state.get_flag('robot_open_flag'):
            self.state.set_flag('robot_open_flag', True)
            self.btn_m3_open.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] Robot Open!")
    
    @QtCore.Slot()
    def _on_robot_pick(self):
        """로봇 Pick"""
        if not self.state.get_flag('robot_pick_flag'):
            self.state.set_flag('robot_pick_flag', True)
            self.btn_m4_pick.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] Robot Pick!")
    
    @QtCore.Slot(object)
    def _on_frame_received(self, data):
        """프레임 수신"""
        try:
            _, frame = data
            self.main_cam.show_frame(frame)
        except Exception as e:
            self._append_log(f"[UI] Frame handler error: {e}")
    
    # ------------------------------------------------------------
    # 스트림 제어
    # ------------------------------------------------------------
    def start_stream(self):
        """스트림 시작"""
        if self._is_running():
            self._append_log("[UI] Already running.")
            return
        
        # ZMQ 스레드 시작
        self._start_zmq_thread()
        
        # ROS2 스레드 시작
        self._start_ros_thread()
    
    def stop_stream(self):
        """스트림 중지"""
        self._stop_threads()
    
    def _is_running(self) -> bool:
        """실행 중인지 확인"""
        zmq_running = self.zmq_thread and self.zmq_thread.isRunning()
        ros_running = self.ros_thread and self.ros_thread.isRunning()
        return zmq_running or ros_running
    
    def _start_zmq_thread(self):
        """ZMQ 스레드 시작"""
        try:
            self.zmq_thread = ZmqSubscriberThread(self.zmq_endpoints)
            self.zmq_thread.text_received.connect(self._append_log)
            self.zmq_thread.frame_received.connect(self._on_frame_received)
            self.zmq_thread.start()
            self._append_log("[ZMQ] Image subscriber started")
        except Exception as e:
            self._append_log(f"[ZMQ] Start error: {e}")
    
    def _start_ros_thread(self):
        """ROS2 스레드 시작"""
        try:
            self.ros_thread = Ros2SubscriberThread()
            self.ros_thread.text_received.connect(self._append_log)
            
            # 탐지 시그널 연결
            self.ros_thread.sos_detected.connect(self._on_sos_detected)
            self.ros_thread.button_detected.connect(self._on_button_detected)
            self.ros_thread.fire_detected.connect(self._on_fire_detected)
            self.ros_thread.door_detected.connect(self._on_door_detected)
            self.ros_thread.safebox_detected.connect(self._on_safebox_detected)
            self.ros_thread.human_detected.connect(self._on_human_detected)
            self.ros_thread.finish_detected.connect(self._on_finish_detected)
            
            # 로봇 상태 시그널 연결
            self.ros_thread.robot_ok.connect(self._on_robot_ok)
            self.ros_thread.robot_open.connect(self._on_robot_open)
            self.ros_thread.robot_pick.connect(self._on_robot_pick)
            
            self.ros_thread.start()
            self._append_log("[ROS2] All-topic subscriber started")
        except Exception as e:
            self._append_log(f"[ROS2] Start error: {e}")
    
    def _stop_threads(self):
        """모든 스레드 중지"""
        if self.zmq_thread and self.zmq_thread.isRunning():
            self.zmq_thread.stop()
            self.zmq_thread.wait(1000)
            self._append_log("[ZMQ] Stopped")
        
        if self.ros_thread and self.ros_thread.isRunning():
            self.ros_thread.stop()
            self.ros_thread.wait(1000)
            self._append_log("[ROS2] Stopped")
    
    def _cleanup(self):
        """리소스 정리"""
        self._stop_threads()

# ---------------------------
# main
# ---------------------------
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