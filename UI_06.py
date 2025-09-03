import sys
import os
import time
from datetime import datetime
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets
import zmq
import base64
import numpy as np
import cv2

## 로봇팔 토픽 ##
DEFAULT_ROS_TOPICS = {
    'ok': '/robot/ok',
    'press': '/robot/press',
    'open': '/robot/open',
    'pick': '/robot/pick'
}

# ---------------------------
# ZMQ Subscriber QThread (정확 매칭 로직)
# ---------------------------
class ZmqSubscriberThread(QtCore.QThread):
    frame_received = QtCore.Signal(object)   # emit((topic, frame_ndarray))
    text_received = QtCore.Signal(str)
    detection_received = QtCore.Signal(str)  # emit(canonical_topic_string) e.g. "button_detected"

    def __init__(self, endpoints, topic_filters=None, context=None, ctx_owned=False, parent=None):
        super().__init__(parent)
        self.endpoints = list(endpoints)
        self.topic_filters = topic_filters or ["" for _ in self.endpoints]
        self._stop = False
        self._ctx_owned = ctx_owned
        if context is None:
            self.context = zmq.Context(io_threads=1)
            self._ctx_owned = True
        else:
            self.context = context

        # 인식 토픽 목록
        self._detection_keywords = {
            'sos_detected',
            'button_detected',
            'fire_detected',
            'door_detected',
            'safebox_detected',
            'human_detected',
            'finish_detected',
        }

        self._sock_to_ep = {}

    def stop(self): # 외부에서 종료 요청을 할 때 호출
        self._stop = True

    def canonicalize(self, s: str) -> str:
        if not s:
            return ""
        t = s.strip().lower()
        if t.startswith('/'):
            t = t.lstrip('/')
        return t

    def run(self):
        sockets = []
        poller = zmq.Poller()
        try:
            for ep, tf in zip(self.endpoints, self.topic_filters):
                try:
                    sock = self.context.socket(zmq.SUB)
                except Exception as e:
                    self.text_received.emit(f"[ZMQ] Failed to create socket: {e}")
                    return
                sock.setsockopt(zmq.RCVHWM, 50)
                sock.linger = 0
                try:
                    sock.connect(ep)
                except Exception as e:
                    self.text_received.emit(f"[ZMQ] connect error {ep}: {e}")
                    return
                sock.setsockopt_string(zmq.SUBSCRIBE, tf or "")
                poller.register(sock, zmq.POLLIN)
                sockets.append(sock)
                self._sock_to_ep[sock] = ep
                self.text_received.emit(f"[ZMQ] connected {ep} (filter='{tf}')")

            while not self._stop:
                events = dict(poller.poll(timeout=200))
                if not events:
                    continue

                for sock in list(events.keys()):
                    if events[sock] & zmq.POLLIN:
                        try:
                            msg = sock.recv_multipart(flags=0)
                            ep = self._sock_to_ep.get(sock, "unknown_ep")

                            if len(msg) >= 2:
                                topic_part = msg[0]
                                payload = msg[1]
                                try:
                                    topic_text = topic_part.decode('utf-8', errors='ignore').strip()
                                except Exception:
                                    topic_text = str(topic_part)
                                canonical = self.canonicalize(topic_text)
                                if canonical in self._detection_keywords:
                                    self.detection_received.emit(canonical)
                                    continue
                                payload_bytes = payload
                            else:
                                payload_bytes = msg[0]
                                try:
                                    payload_text = payload_bytes.decode('utf-8', errors='ignore').strip()
                                    canonical = self.canonicalize(payload_text)
                                    if canonical in self._detection_keywords:
                                        self.detection_received.emit(canonical)
                                        continue
                                except Exception:
                                    pass

                            # 이미지 디코딩 (base64 기대)
                            frame = None
                            try:
                                jpg_bytes = base64.b64decode(payload_bytes, validate=False)
                                np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            except Exception:
                                try:
                                    np_arr = np.frombuffer(payload_bytes, dtype=np.uint8)
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                except Exception as e2:
                                    self.text_received.emit(f"[ZMQ][{ep}] frame decode error: {e2}")

                            if frame is not None:
                                self.frame_received.emit(("camera", frame))
                                continue

                        except zmq.Again:
                            continue
                        except Exception as e:
                            self.text_received.emit(f"[ZMQ] recv error: {e}")
        finally:
            for s in sockets:
                try:
                    poller.unregister(s)
                except Exception:
                    pass
                try:
                    s.close(0)
                except Exception:
                    pass
            if self._ctx_owned:
                try:
                    self.context.term()
                except Exception:
                    pass
            self.text_received.emit("[ZMQ] subscriber thread terminated.")


# ---------------------------
# ROS2 (rclpy) Subscriber QThread (Humble 전용)
# ---------------------------
class Ros2SubscriberThread(QtCore.QThread):
    ok_received = QtCore.Signal()
    press_received = QtCore.Signal()
    open_received = QtCore.Signal()
    pick_received = QtCore.Signal()
    text_received = QtCore.Signal(str)

    def __init__(self, topics=None, parent=None):
        super().__init__(parent)
        self._stop = False
        self.topics = topics or DEFAULT_ROS_TOPICS
        self._rclpy_inited = False
        self._node = None

    def stop(self):
        self._stop = True

    def _make_cb_bool(self, signal):
        def cb(msg):
            try:
                if getattr(msg, 'data', False):
                    signal.emit()
            except Exception:
                pass
        return cb

    def _make_cb_string(self, signal, accept_values=('1', 'true', 'yes', 'ok', 'done', 'open', 'pick')):
        def cb(msg):
            try:
                v = getattr(msg, 'data', None)
                if v is None:
                    return
                if isinstance(v, str):
                    if v.lower() in accept_values:
                        signal.emit()
                else:
                    if bool(v):
                        signal.emit()
            except Exception:
                pass
        return cb

    def run(self):
        try:
            import rclpy
            from std_msgs.msg import String, Bool
        except Exception as e:
            self.text_received.emit(f"[ROS2] rclpy not available: {e}")
            return

        try:
            try:
                rclpy.init(args=None)
                self._rclpy_inited = True
            except Exception:
                self._rclpy_inited = True

            try:
                self._node = rclpy.create_node('gui_ros2_subscriber')
            except Exception as e:
                self.text_received.emit(f"[ROS2] create_node failed: {e}")
                self._node = None

            if self._node is None:
                return

            # try Bool subscriptions first, fallback to String -> 값 확인해서 하나로 하나만 남기기
            try:
                self._node.create_subscription(String, self.topics['ok'], self._make_cb_bool(self.ok_received), 10)
                self._node.create_subscription(String, self.topics['press'], self._make_cb_bool(self.press_received), 10)
                self._node.create_subscription(String, self.topics['open'], self._make_cb_bool(self.open_received), 10)
                self._node.create_subscription(String, self.topics['pick'], self._make_cb_bool(self.pick_received), 10)
            except Exception:
                try:
                    self._node.create_subscription(Bool, self.topics['ok'], self._make_cb_string(self.ok_received), 10)
                    self._node.create_subscription(Bool, self.topics['press'], self._make_cb_string(self.press_received), 10)
                    self._node.create_subscription(Bool, self.topics['open'], self._make_cb_string(self.open_received), 10)
                    self._node.create_subscription(Bool, self.topics['pick'], self._make_cb_string(self.pick_received), 10)
                except Exception as e:
                    self.text_received.emit(f"[ROS2] subscription creation error: {e}")

            while not self._stop and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)

        except Exception as e:
            self.text_received.emit(f"[ROS2] runtime error: {e}")
        finally:
            try:
                if self._node is not None:
                    try:
                        self._node.destroy_node()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if self._rclpy_inited:
                    import rclpy
                    try:
                        rclpy.shutdown()
                    except Exception:
                        pass
            except Exception:
                pass
            self.text_received.emit("[ROS2] subscriber thread terminated.")


# ---------------------------
# GUI widgets
# ---------------------------
class CameraLabel(QtWidgets.QLabel):
    def __init__(self, placeholder_text="No Image", parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(placeholder_text)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #111; color: #eee;")

    def show_frame(self, frame_bgr: np.ndarray):
        # 안전검사
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            self.clear()
            self.setText("No Image")
            return

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            if qimg.isNull():
                self.clear()
                self.setText("No Image")
                return
            pix = QtGui.QPixmap.fromImage(qimg.copy())
            if pix.isNull():
                self.clear()
                self.setText("No Image")
                return
            scaled = pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)
        except Exception as e:
            # 개발 시 디버깅 로그
            print(f"[CameraLabel] show_frame error: {e}", file=sys.stderr)
            self.clear()
            self.setText("No Image")

    def resizeEvent(self, ev):
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            scaled = pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)
        super().resizeEvent(ev)


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    DEFAULT_RESET_FLAG = {
        'SOS'     : {'flags': 'sos_flag',              'btn': 'btn_m1_sos',   'text': 'SOS: NO',    'type': 'detection'},
        'Button'  : {'flags': 'button_flag',           'btn': 'btn_m3_button','text': 'BUTTON: NO', 'type': 'detection'},
        'Fire'    : {'flags': 'fire_flag',             'btn': 'btn_m3_fire',  'text': 'FIRE: NO',   'type': 'detection'},
        'Door'    : {'flags': 'door_flag',             'btn': 'btn_m3_door',  'text': 'DOOR: NO',   'type': 'detection'},
        'Safebox' : {'flags': 'safebox_flag',          'btn': 'btn_m4_safebox','text': 'SAFEBOX: NO','type': 'detection'},
        'Human'   : {'flags': 'human_flag',            'btn': 'btn_m4_human', 'text': 'HUMAN: NO',  'type': 'detection'},
        'Finish'  : {'flags': 'finish_flag',           'btn': 'btn_m4_finish','text': 'FINISH: NO', 'type': 'detection'},

        'OK'      : {'flags': 'robot_m3_ok_flag',      'btn': 'btn_m3_ok',  'type': 'robot'},
        'Press'   : {'flags': 'robot_m3_press_flag',   'btn': 'btn_m3_press', 'type': 'robot'},
        'Open'    : {'flags': 'robot_m3_open_flag',    'btn': 'btn_m3_open',  'type': 'robot'},
        'Pick'    : {'flags': 'robot_m4_pick_flag',    'btn': 'btn_m4_pick', 'type': 'robot'},
    }
    ## 여기서 값들 수정 ##

    def __init__(self, zmq_endpoints, ros_topics=None):
        super().__init__()
        self.setWindowTitle("ZMQ + ROS2 Humble Viewer (Aligned) - with EMERGENCY STOP")
        self.resize(1000, 700)

        # Left: camera + control
        self.main_cam = CameraLabel("Main camera")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        # self.stop_btn.setEnabled(False)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.main_cam, stretch=1)
        left_layout.addLayout(btn_layout)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)

        # Right: tabbar + mission stack + log + emergency stop
        right_layout = QtWidgets.QVBoxLayout()

        # Tabs
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setFixedHeight(34)
        for name in ("mission1", "mission2", "mission3", "mission4"):
            self.tab_bar.addTab(name)
        self.tab_bar.setCurrentIndex(0)

        self.mission_stack = QtWidgets.QStackedWidget()

        # common font (same for detection and robot buttons)
        common_font = QtGui.QFont()
        common_font.setPointSize(12)
        common_font.setBold(True)

        # styles
        self.det_idle_style = "color: white; background: darkred;"
        self.det_active_style = "color: black; background: yellow;"
        # robot initial: white background, dark border, same font/size; active: green
        self.robot_idle_style = "color: black; background: white; border:1px solid #8b1f1f; padding:6px;"
        self.robot_active_style = "color: white; background: #2ecc71; border:1px solid #1e8449; padding:6px;"

        # create pages
        for i in range(4):
            pg = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout()
            l.setContentsMargins(0, 0, 0, 0)

            if i == 0:
                # mission1: SOS positioned like mission3 buttons (left/top aligned)
                vbox = QtWidgets.QVBoxLayout()
                vbox.setContentsMargins(0,0,0,0)
                vbox.setSpacing(6)

                self.btn_m1_sos = QtWidgets.QPushButton("SOS: NO")
                self.btn_m1_sos.setEnabled(True)
                self.btn_m1_sos.setFixedSize(150, 44)
                self.btn_m1_sos.setFont(common_font)
                self.btn_m1_sos.setStyleSheet(self.det_idle_style)
                self.btn_m1_sos.clicked.connect(partial(self.reset_flag, 'SOS'))
                vbox.addWidget(self.btn_m1_sos, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                wrapper = QtWidgets.QWidget()
                wrapper.setLayout(vbox)
                wrapper.setFixedWidth(150)
                wrapper.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                l.addWidget(wrapper, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

            elif i == 2:
                # mission3: left column detection buttons + right column robot status buttons
                left_vbox_m3 = QtWidgets.QVBoxLayout()      # 세로 방향으로 위젯 쌓는 레이아웃 객체 생성
                left_vbox_m3.setContentsMargins(0,0,0,0)    # 상하좌우 여백을 0으로 설정
                left_vbox_m3.setSpacing(6)                  # 간격 6으로 설정

                self.btn_m3_button = QtWidgets.QPushButton("BUTTON: NO")    # 버튼 생성
                self.btn_m3_button.setFixedSize(150, 44)                    # 버튼 크기: 150x44
                self.btn_m3_button.setFont(common_font)                     # 버튼 폰트
                self.btn_m3_button.setStyleSheet(self.det_idle_style)       # 버튼 글씨체, 색깔 등
                self.btn_m3_button.clicked.connect(partial(self.reset_flag, 'Button'))         # 기능
                left_vbox_m3.addWidget(self.btn_m3_button, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  #생성한 버튼을 레이아웃에 추가, alignment로 왼쪽 상단 정렬, 각 위젯이 수직으로 쌓임.

                self.btn_m3_fire = QtWidgets.QPushButton("FIRE: NO")
                self.btn_m3_fire.setFixedSize(150, 44)
                self.btn_m3_fire.setFont(common_font)
                self.btn_m3_fire.setStyleSheet(self.det_idle_style)
                self.btn_m3_fire.clicked.connect(partial(self.reset_flag, 'Fire'))
                left_vbox_m3.addWidget(self.btn_m3_fire, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_door = QtWidgets.QPushButton("DOOR: NO")
                self.btn_m3_door.setFixedSize(150, 44)
                self.btn_m3_door.setFont(common_font)
                self.btn_m3_door.setStyleSheet(self.det_idle_style)
                self.btn_m3_door.clicked.connect(partial(self.reset_flag, 'Door'))
                left_vbox_m3.addWidget(self.btn_m3_door, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                # 불의 개수 세는 코드
                self.fire_count = 0  # 불 감지 카운터 (초기값)
                self.lbl_m3_fire_count = QtWidgets.QLabel("불의 개수: 0 개 (0/3)")
                self.lbl_m3_fire_count.setFont(common_font)
                self.lbl_m3_fire_count.setFixedSize(150, 44)
                self.lbl_m3_fire_count.setWordWrap(True)
                self.lbl_m3_fire_count.setAlignment(QtCore.Qt.AlignCenter)
                left_vbox_m3.addWidget(self.lbl_m3_fire_count, alignment=QtCore.Qt.AlignLeft)
                ####################

                left_wrapper_m3 = QtWidgets.QWidget()
                left_wrapper_m3.setLayout(left_vbox_m3)
                left_wrapper_m3.setFixedWidth(150)
                left_wrapper_m3.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

                # right column (robot status) for mission3
                robot_vbox_m3 = QtWidgets.QVBoxLayout()
                robot_vbox_m3.setContentsMargins(0,0,0,0)
                robot_vbox_m3.setSpacing(6)
                robot_vbox_m3.setAlignment(QtCore.Qt.AlignTop)  # 상단에 강제로 고정

                self.btn_m3_ok = QtWidgets.QPushButton("OK")
                self.btn_m3_ok.setFixedSize(150, 44)
                self.btn_m3_ok.setFont(common_font)
                self.btn_m3_ok.setStyleSheet(self.robot_idle_style)
                self.btn_m3_ok.clicked.connect(partial(self.reset_flag, 'OK', 'btn_m3_ok'))
                robot_vbox_m3.addWidget(self.btn_m3_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_press = QtWidgets.QPushButton("Press")
                self.btn_m3_press.setFixedSize(150, 44)
                self.btn_m3_press.setFont(common_font)
                self.btn_m3_press.setStyleSheet(self.robot_idle_style)
                self.btn_m3_press.clicked.connect(partial(self.reset_flag, 'Press'))
                robot_vbox_m3.addWidget(self.btn_m3_press, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_open = QtWidgets.QPushButton("Open")
                self.btn_m3_open.setFixedSize(150, 44)
                self.btn_m3_open.setFont(common_font)
                self.btn_m3_open.setStyleSheet(self.robot_idle_style)
                self.btn_m3_open.clicked.connect(partial(self.reset_flag, 'Open'))
                robot_vbox_m3.addWidget(self.btn_m3_open, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                right_wrapper_m3 = QtWidgets.QWidget()
                right_wrapper_m3.setLayout(robot_vbox_m3)
                right_wrapper_m3.setFixedWidth(150)
                right_wrapper_m3.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)  # Fixed -> Minimum

                # Put left and right wrappers into an HBox
                h_inner_m3 = QtWidgets.QHBoxLayout()
                h_inner_m3.setContentsMargins(0,0,0,0)
                h_inner_m3.setSpacing(12)
                h_inner_m3.addWidget(left_wrapper_m3, 0, QtCore.Qt.AlignTop)    # 수정
                h_inner_m3.addWidget(right_wrapper_m3, 0, QtCore.Qt.AlignTop)   # 수정

                container_m3 = QtWidgets.QWidget()
                container_m3.setLayout(h_inner_m3)
                # container_m3.setFixedWidth(150 + 12 + 150)
                container_m3.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)  # Fixed -> Minimum

                l.addWidget(container_m3, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            elif i == 3:
                # mission4: left detection (SAFEBOX/HUMAN/FINISH) + right robot status (OK/PICK)
                left_vbox_m4 = QtWidgets.QVBoxLayout()
                left_vbox_m4.setContentsMargins(0,0,0,0)
                left_vbox_m4.setSpacing(6)

                self.btn_m4_safebox = QtWidgets.QPushButton("SAFEBOX: NO")
                self.btn_m4_safebox.setFixedSize(150, 44)
                self.btn_m4_safebox.setFont(common_font)
                self.btn_m4_safebox.setStyleSheet(self.det_idle_style)
                self.btn_m4_safebox.clicked.connect(partial(self.reset_flag, 'Safebox'))
                left_vbox_m4.addWidget(self.btn_m4_safebox, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_human = QtWidgets.QPushButton("HUMAN: NO")
                self.btn_m4_human.setFixedSize(150, 44)
                self.btn_m4_human.setFont(common_font)
                self.btn_m4_human.setStyleSheet(self.det_idle_style)
                self.btn_m4_human.clicked.connect(partial(self.reset_flag, 'Human'))
                left_vbox_m4.addWidget(self.btn_m4_human, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_finish = QtWidgets.QPushButton("FINISH: NO")
                self.btn_m4_finish.setFixedSize(150, 44)
                self.btn_m4_finish.setFont(common_font)
                self.btn_m4_finish.setStyleSheet(self.det_idle_style)
                self.btn_m4_finish.clicked.connect(partial(self.reset_flag, 'Finish'))
                left_vbox_m4.addWidget(self.btn_m4_finish, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                # 사람 명수 세는 코드
                self.human_count = 0  # 사람 감지 카운터 (초기값)
                self.lbl_m4_human_count = QtWidgets.QLabel("사람: 0 명 (0/3)")
                self.lbl_m4_human_count.setFont(common_font)
                self.lbl_m4_human_count.setFixedSize(150, 44)
                self.lbl_m4_human_count.setWordWrap(True)
                self.lbl_m4_human_count.setAlignment(QtCore.Qt.AlignCenter)
                left_vbox_m4.addWidget(self.lbl_m4_human_count, alignment=QtCore.Qt.AlignLeft)

                self.human_state = 0
                self.lbl_m4_human_state = QtWidgets.QLabel("상태: 생존 / 사망")
                self.lbl_m4_human_state.setFont(common_font)
                self.lbl_m4_human_state.setFixedSize(150, 44)
                self.lbl_m4_human_state.setWordWrap(True)
                self.lbl_m4_human_state.setAlignment(QtCore.Qt.AlignCenter)
                left_vbox_m4.addWidget(self.lbl_m4_human_state, alignment=QtCore.Qt.AlignLeft)
                ####################

                left_wrapper_m4 = QtWidgets.QWidget()
                left_wrapper_m4.setLayout(left_vbox_m4)
                left_wrapper_m4.setFixedWidth(150)
                left_wrapper_m4.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

                # right column for mission4 robot status
                robot_vbox_m4 = QtWidgets.QVBoxLayout()
                robot_vbox_m4.setContentsMargins(0,0,0,0)
                robot_vbox_m4.setSpacing(6)
                robot_vbox_m4.setAlignment(QtCore.Qt.AlignTop)

                self.btn_m4_ok = QtWidgets.QPushButton("OK")
                self.btn_m4_ok.setFixedSize(150, 44)
                self.btn_m4_ok.setFont(common_font)
                self.btn_m4_ok.setStyleSheet(self.robot_idle_style)
                self.btn_m4_ok.clicked.connect(partial(self.reset_flag, 'OK', 'btn_m4_ok'))
                robot_vbox_m4.addWidget(self.btn_m4_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_pick = QtWidgets.QPushButton("Pick")
                self.btn_m4_pick.setFixedSize(150, 44)
                self.btn_m4_pick.setFont(common_font)
                self.btn_m4_pick.setStyleSheet(self.robot_idle_style)
                self.btn_m4_pick.clicked.connect(partial(self.reset_flag, 'Pick'))
                robot_vbox_m4.addWidget(self.btn_m4_pick, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                right_wrapper_m4 = QtWidgets.QWidget()
                right_wrapper_m4.setLayout(robot_vbox_m4)
                right_wrapper_m4.setFixedWidth(150)
                right_wrapper_m4.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)

                # Put left and right wrappers into an HBox
                h_inner_m4 = QtWidgets.QHBoxLayout()
                h_inner_m4.setContentsMargins(0,0,0,0)
                h_inner_m4.setSpacing(12)
                h_inner_m4.addWidget(left_wrapper_m4, 0, QtCore.Qt.AlignTop)    # 수정
                h_inner_m4.addWidget(right_wrapper_m4, 0, QtCore.Qt.AlignTop)

                container_m4 = QtWidgets.QWidget()
                container_m4.setLayout(h_inner_m4)
                # container_m4.setFixedWidth(150 + 12 + 150)
                container_m4.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)

                l.addWidget(container_m4, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            else:
                placeholder = QtWidgets.QLabel("")
                placeholder.setFixedSize(150, 48)
                l.addWidget(placeholder)

            pg.setLayout(l)
            self.mission_stack.addWidget(pg)

        # sync tabs <-> stack
        self.tab_bar.currentChanged.connect(self.mission_stack.setCurrentIndex)
        self.mission_stack.currentChanged.connect(self.tab_bar.setCurrentIndex)

        # top layout: tabbar
        top_h = QtWidgets.QHBoxLayout()
        top_h.addWidget(self.tab_bar, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        top_h.addStretch()
        right_layout.addLayout(top_h)

        # mission_stack size increased so two columns fit and align with mission4
        self.mission_stack.setFixedSize(336, 220)
        right_layout.addWidget(self.mission_stack, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        right_layout.addStretch()

        # log
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setMinimumWidth(300)
        self.log_list.setMaximumWidth(400)
        log_container = QtWidgets.QWidget()
        log_container_layout = QtWidgets.QVBoxLayout()
        log_container_layout.addStretch()
        log_container_layout.addWidget(self.log_list, alignment=QtCore.Qt.AlignBottom)
        log_container.setLayout(log_container_layout)
        right_layout.addWidget(log_container)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMinimumWidth(420)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_widget, stretch=1)
        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # internal
        self.zmq_thread = None
        self.ros_thread = None
        self.zmq_endpoints = zmq_endpoints
        self.ros_topics = ros_topics or DEFAULT_ROS_TOPICS

        # detection flags (keep original names)
        self.sos_flag = False
        self.button_flag = False
        self.fire_flag = False
        self.door_flag = False
        self.safebox_flag = False
        self.human_flag = False
        self.finish_flag = False

        # robot flags (page-aware)
        self.robot_m3_ok_flag = False
        self.robot_m3_press_flag = False
        self.robot_m3_open_flag = False

        self.robot_m4_ok_flag = False
        self.robot_m4_pick_flag = False

        # hookups
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self._left_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self._right_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self._left_shortcut.activated.connect(self.go_prev_tab)
        self._right_shortcut.activated.connect(self.go_next_tab)

        QtCore.QCoreApplication.instance().aboutToQuit.connect(self.on_app_quit)

    # -----------------------------
    # ZMQ / ROS control
    # -----------------------------
    def start_stream(self):
        if self.zmq_thread is None or not self.zmq_thread.isRunning():
            self.zmq_thread = ZmqSubscriberThread(self.zmq_endpoints)
            self.zmq_thread.frame_received.connect(self.on_frame_received)
            self.zmq_thread.text_received.connect(self.append_log)
            self.zmq_thread.detection_received.connect(self.on_detection_received)
            self.zmq_thread.start()
            self.append_log("[UI] ZMQ Subscriber started.")
        else:
            self.append_log("[UI] ZMQ already running.")

        if self.ros_thread is None or not self.ros_thread.isRunning():
            self.ros_thread = Ros2SubscriberThread(topics=self.ros_topics)
            self.ros_thread.ok_received.connect(self.on_robot_ok)
            self.ros_thread.press_received.connect(self.on_robot_press)
            self.ros_thread.open_received.connect(self.on_robot_open)
            self.ros_thread.pick_received.connect(self.on_robot_pick)
            self.ros_thread.text_received.connect(self.append_log)
            self.ros_thread.start()
            self.append_log("[UI] ROS2 Subscriber started.")
        else:
            self.append_log("[UI] ROS2 already running.")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_stream(self):
        # 정상적인 정지 시도 (start_stream의 역)
        if self.zmq_thread is not None and self.zmq_thread.isRunning():
            self.zmq_thread.stop()
            self.zmq_thread.wait(1000)
            self.append_log("[UI] ZMQ Subscriber stopped.")
        self.zmq_thread = None

        if self.ros_thread is not None and self.ros_thread.isRunning():
            self.ros_thread.stop()
            self.ros_thread.wait(1000)
            self.append_log("[UI] ROS2 Subscriber stopped.")
        self.ros_thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @QtCore.Slot(object)
    def on_frame_received(self, item):
        camera_id, frame = item
        self.main_cam.show_frame(frame)

    # ZMQ detection handling
    @QtCore.Slot(str)
    def on_detection_received(self, canonical_topic: str):
        t = (canonical_topic or "").strip().lower()

        DET_MAP = {
            'sos_detected': 'SOS',
            'button_detected': 'Button',
            'fire_detected': 'Fire',
            'door_detected': 'Door',
            'safebox_detected': 'Safebox',
            'human_detected': 'Human',
            'finish_detected': 'Finish',
        }

        key = DET_MAP.get(t)
        if not key:
            self.append_log(f"[DETECT-UNKNOWN] {canonical_topic}")
            return

        cfg = self.DEFAULT_RESET_FLAG.get(key, {})
        # 지원: cfg에 'flags' 또는 과거 'flag' 둘 다 가능
        flags = self._normalize_flags(cfg.get('flags') or cfg.get('flag'))

        # --- 중요한 변경: FIRE 카운트는 이미 활성 상태와 무관하게 항상 증가시킨다 ---
        if key == 'Fire':
            try:
                self.fire_count = getattr(self, 'fire_count', 0) + 1
                # 라벨 업데이트
                try:
                    if hasattr(self, 'lbl_m3_fire_count') and self.lbl_m3_fire_count is not None:
                        self.lbl_m3_fire_count.setText(f"불의 개수: {self.fire_count} 개 ({self.fire_count}/3)")
                except Exception:
                    pass
            except Exception:
                pass
        # ------------------------------------------------------------------

        activated = False
        for f in flags:
            try:
                if not getattr(self, f, False):
                    setattr(self, f, True)
                    activated = True
            except Exception:
                pass

        # 이미 활성화되어 있으면 아무 처리 안함 (기존 동작 유지)
        if not activated:
            # 단, FIRE 카운트는 위에서 처리했기 때문에 여기서 바로 리턴하지 않아도 무방.
            # 기존 동작 유지(스타일/텍스트 재설정은 하지 않음)
            return

        # 버튼 UI 업데이트 (탐지 버튼은 텍스트 변경 + det_active_style)
        btn_entry = cfg.get('btn')
        # btn이 리스트일 수도 있으니 일관 처리
        btn_names = btn_entry if isinstance(btn_entry, (list, tuple)) else [btn_entry] if btn_entry else []
        for btn_name in btn_names:
            try:
                btn = getattr(self, btn_name, None)
                if btn is None:
                    continue
                # active text 우선: cfg의 'text' 사용(예: "SOS: NO") -> "SOS: YES" 생성
                txt_idle = cfg.get('text') or cfg.get('Text') or ""
                if txt_idle:
                    if ':' in txt_idle:
                        prefix = txt_idle.split(':', 1)[0].strip()
                        txt_active = f"{prefix}: YES"
                    else:
                        txt_active = f"{txt_idle}: YES"
                else:
                    txt_active = f"{key}: YES"
                try:
                    btn.setText(txt_active)
                except Exception:
                    pass
                try:
                    btn.setStyleSheet(self.det_active_style)
                except Exception:
                    pass
            except Exception:
                pass

        # 로그 형식은 기존과 동일하게 유지
        self.append_log(f"/{t}")

    # ROS2 callbacks -> set corresponding robot button to green (active)
    @QtCore.Slot()
    def on_robot_ok(self):
        try:
            # mission3 OK
            self.robot_m3_ok_flag = True
            if hasattr(self, 'btn_m3_ok'):
                self.btn_m3_ok.setStyleSheet(self.robot_active_style)
            # mission4 OK as well
            self.robot_m4_ok_flag = True
            if hasattr(self, 'btn_m4_ok'):
                self.btn_m4_ok.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] robot OK")
        except Exception:
            pass

    @QtCore.Slot()
    def on_robot_press(self):
        try:
            self.robot_m3_press_flag = True
            if hasattr(self, 'btn_m3_press'):
                self.btn_m3_press.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] robot Press")
        except Exception:
            pass

    @QtCore.Slot()
    def on_robot_open(self):
        try:
            self.robot_m3_open_flag = True
            if hasattr(self, 'btn_m3_open'):
                self.btn_m3_open.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] robot Open")
        except Exception:
            pass

    @QtCore.Slot()
    def on_robot_pick(self):
        try:
            # mission4 PICK (and optionally mission3 if you had a pick there)
            self.robot_m4_pick_flag = True
            if hasattr(self, 'btn_m4_pick'):
                self.btn_m4_pick.setStyleSheet(self.robot_active_style)
            # if there is a mission3 pick (not present by default), set it too
            if hasattr(self, 'btn_m3_pick'):
                self.btn_m3_pick.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] robot Pick")
        except Exception:
            pass

    @QtCore.Slot(str)
    def append_log(self, s: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_list.addItem(f"{ts} {s}")
        self.log_list.scrollToBottom()

    # -----------------------------
    # resets (buttons on click reset only their flags / styles)
    # -----------------------------
    def _normalize_flags(self, flags):
        """helper: flags가 문자열이면 리스트로 바꿔서 반환"""
        if flags is None:
            return []
        if isinstance(flags, str):
            return [flags]
        try:
            return list(flags)
        except Exception:
            return [flags]

    def reset_flag(self, key: str, btn_override: str = None, checked=None):
        """범용 리셋.
        - key: DEFAULT_RESET_FLAG 키
        - btn_override: (선택) 실제로 스타일/text를 바꿀 버튼 이름(예: 'btn_m4_ok')
        - checked: Qt clicked 시그널이 전달하는 값(무시)
        """
        cfg = self.DEFAULT_RESET_FLAG.get(key)
        if not cfg:
            self.append_log(f"[UI] reset_flag: unknown key '{key}'")
            return

        flags = self._normalize_flags(cfg.get('flags'))
        was_any = False
        for f in flags:
            was = bool(getattr(self, f, False))
            was_any = was_any or was
            try:
                setattr(self, f, False)
            except Exception:
                pass

        # 버튼 UI 복구: btn_override가 있으면 우선 사용, 없으면 cfg의 btn 사용
        btn_name = btn_override or cfg.get('btn')
        try:
            btn = getattr(self, btn_name, None)
            if btn is not None:
                tp = cfg.get('type', 'detection')
                if tp == 'detection':
                    txt = cfg.get('text')
                    if txt is not None:
                        btn.setText(txt)
                    btn.setStyleSheet(self.det_idle_style)
                else:
                    btn.setStyleSheet(self.robot_idle_style)
        except Exception:
            pass

        # --- 추가: Fire 리셋 시 카운터도 초기화 ---
        if key == 'Fire':
            try:
                self.fire_count = 0
                if hasattr(self, 'lbl_m3_fire_count') and self.lbl_m3_fire_count is not None:
                    self.lbl_m3_fire_count.setText(f"불의 개수: 0 개 (0/3)")
            except Exception:
                pass
        # -------------------------------------------------

        if was_any:
            self.append_log(f"[UI] {key} flag reset (by {key} click)")
        else:
            self.append_log(f"[UI] {key} clicked (no active flag)")

    # tabs
    def go_next_tab(self):
        idx = self.tab_bar.currentIndex()
        cnt = self.tab_bar.count()
        self.tab_bar.setCurrentIndex((idx + 1) % cnt)
        self.append_log(f"[UI] mission -> {self.tab_bar.currentIndex()+1}")

    def go_prev_tab(self):
        idx = self.tab_bar.currentIndex()
        cnt = self.tab_bar.count()
        self.tab_bar.setCurrentIndex((idx - 1) % cnt)
        self.append_log(f"[UI] mission -> {self.tab_bar.currentIndex()+1}")

    def on_app_quit(self):
        # ensure threads stopped (앱 종료 시 안전하게 stop 요청)
        if self.zmq_thread is not None and self.zmq_thread.isRunning():
            self.zmq_thread.stop()
            self.zmq_thread.wait(1000)
        if self.ros_thread is not None and self.ros_thread.isRunning():
            self.ros_thread.stop()
            self.ros_thread.wait(1000)

# ---------------------------
# Entry point
# ---------------------------
def main():
    endpoints = [
        "tcp://127.0.0.1:5555",  # image
        "tcp://127.0.0.1:5556",  # safebox detection
        "tcp://127.0.0.1:5558",  # human detection
        "tcp://127.0.0.1:5559",  # finish detection
    ]

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(endpoints)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
