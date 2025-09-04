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
# ZMQ 메시지를 백그라운드에서 수신 (정확 매칭 로직)
# ---------------------------
class ZmqSubscriberThread(QtCore.QThread):
    frame_received = QtCore.Signal(object)   # emit((topic, frame_ndarray))
    text_received = QtCore.Signal(str)

    def __init__(self, endpoints, context=None, parent=None):
        super().__init__(parent)
        self.endpoints = list(endpoints)
        self._stop = False
        self._ctx_owned = False
        if context is None:
            self.context = zmq.Context(io_threads=1)
            self._ctx_owned = True
        else:
            self.context = context
        self._sock_to_ep = {}

    def stop(self): # 외부에서 종료 요청을 할 때 호출
        self._stop = True

    def run(self):
        sockets = []
        poller = zmq.Poller()
        try:
            for ep in self.endpoints:
                try:
                    sock = self.context.socket(zmq.SUB)
                    sock.setsockopt(zmq.RCVHWM, 50)
                    sock.linger = 0
                    sock.connect(ep)
                    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # 모든 프레임 수신
                    poller.register(sock, zmq.POLLIN)
                    sockets.append(sock)
                    self._sock_to_ep[sock] = ep
                    self.text_received.emit(f"[ZMQ] connected {ep} (image only)")
                except Exception as e:
                    self.text_received.emit(f"[ZMQ] connect error {ep}: {e}")
                    return

            while not self._stop:
                events = dict(poller.poll(timeout=200))
                if not events:
                    continue

                for sock in list(events.keys()):
                    if events[sock] & zmq.POLLIN:
                        try:
                            msg = sock.recv_multipart(flags=0)
                            payload_bytes = msg[-1] if len(msg) >= 1 else b""

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
                                    self.text_received.emit(f"[ZMQ] frame decode error: {e2}")

                            if frame is not None:
                                self.frame_received.emit(("camera", frame))
                        except Exception as e:
                            self.text_received.emit(f"[ZMQ] recv error: {e}")
        finally:
            for s in sockets:
                try: poller.unregister(s)
                except: pass
                try: s.close(0)
                except:pass
            if self._ctx_owned:
                try: self.context.term()
                except: pass
            self.text_received.emit("[ZMQ] subscriber thread terminated.")


# ---------------------------
# ROS2 (rclpy) ROS2 토픽을 백그라운드에서 구독
# ---------------------------
class Ros2SubscriberThread(QtCore.QThread):
    # 탐지 시그널
    sos_detected     = QtCore.Signal()
    button_detected  = QtCore.Signal()
    fire_detected    = QtCore.Signal()
    door_detected    = QtCore.Signal()
    safebox_detected = QtCore.Signal()
    human_detected   = QtCore.Signal()
    finish_detected  = QtCore.Signal()

    # 로봇 상태 시그널
    robot_ok    = QtCore.Signal()
    robot_press = QtCore.Signal()
    robot_open  = QtCore.Signal()
    robot_pick  = QtCore.Signal()

    text_received = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self._rclpy_inited = False
        self._node = None
        self._subs = []

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import rclpy
            from std_msgs.msg import Empty  # ← Empty 타입 필수
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
                self._node = rclpy.create_node('ui_subscriber')
            except Exception as e:
                self.text_received.emit(f"[ROS2] create_node failed: {e}")
                return

            # -------------------------
            # Object detection 토픽들 (모두 Empty)
            # -------------------------
            topics_det = {
                '/object_detection/SOS'     : self.sos_detected,
                '/object_detection/button'  : self.button_detected,
                '/object_detection/fire'    : self.fire_detected,
                '/object_detection/door'    : self.door_detected,
                '/object_detection/safebox' : self.safebox_detected,
                '/object_detection/human'   : self.human_detected,
                '/object_detection/finish'  : self.finish_detected,
            }

            for t, sig in topics_det.items():
                try:
                    sub = self._node.create_subscription(
                        Empty, t, lambda msg, s=sig, tn=t: s.emit(), 10
                    )
                    self._subs.append(sub)
                    self.text_received.emit(f"[ROS2] Subscribed to {t}")
                except Exception as e:
                    self.text_received.emit(f"[ROS2] subscription error {t}: {e}")

            # -------------------------
            # Robot 상태 토픽들 (모두 Empty)
            # -------------------------
            topics_robot = {
                '/robot/ok'   : self.robot_ok,
                '/robot/press': self.robot_press,
                '/robot/open' : self.robot_open,
                '/robot/pick' : self.robot_pick,
            }
            for t, sig in topics_robot.items():
                try:
                    sub = self._node.create_subscription(
                        Empty, t, lambda msg, s=sig, tn=t: s.emit(), 10
                    )
                    self._subs.append(sub)
                    self.text_received.emit(f"[ROS2] Subscribed to {t}")
                except Exception as e:
                    self.text_received.emit(f"[ROS2] subscription error {t}: {e}")

            while not self._stop and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)

        except Exception as e:
            self.text_received.emit(f"[ROS2] runtime error: {e}")
        finally:
            try:
                if self._node is not None:
                    try: self._node.destroy_node()
                    except: pass
            except: pass
            try:
                if self._rclpy_inited:
                    import rclpy
                    try: rclpy.shutdown()
                    except: pass
            except: pass
            self.text_received.emit("[ROS2] subscriber thread terminated.")

# ---------------------------
# GUI widgets, 수신한 카메라 영상 데이터를 화면에 표시
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
# Main Window, 프로그램의 메인 창 UI를 만들고 백그라운드 스레드 관리
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
                self.btn_m1_sos.clicked.connect(self.reset_sos_flag)
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
                self.btn_m3_button.clicked.connect(self.reset_button_flag)        # 기능
                left_vbox_m3.addWidget(self.btn_m3_button, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  #생성한 버튼을 레이아웃에 추가, alignment로 왼쪽 상단 정렬, 각 위젯이 수직으로 쌓임.

                self.btn_m3_fire = QtWidgets.QPushButton("FIRE: NO")
                self.btn_m3_fire.setFixedSize(150, 44)
                self.btn_m3_fire.setFont(common_font)
                self.btn_m3_fire.setStyleSheet(self.det_idle_style)
                self.btn_m3_fire.clicked.connect(self.reset_fire_flag)
                left_vbox_m3.addWidget(self.btn_m3_fire, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_door = QtWidgets.QPushButton("DOOR: NO")
                self.btn_m3_door.setFixedSize(150, 44)
                self.btn_m3_door.setFont(common_font)
                self.btn_m3_door.setStyleSheet(self.det_idle_style)
                self.btn_m3_door.clicked.connect(self.reset_door_flag)
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
                self.btn_m3_ok.clicked.connect(self.reset_robot_ok_flag)
                robot_vbox_m3.addWidget(self.btn_m3_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_press = QtWidgets.QPushButton("Press")
                self.btn_m3_press.setFixedSize(150, 44)
                self.btn_m3_press.setFont(common_font)
                self.btn_m3_press.setStyleSheet(self.robot_idle_style)
                self.btn_m3_press.clicked.connect(self.reset_robot_press_flag)
                robot_vbox_m3.addWidget(self.btn_m3_press, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m3_open = QtWidgets.QPushButton("Open")
                self.btn_m3_open.setFixedSize(150, 44)
                self.btn_m3_open.setFont(common_font)
                self.btn_m3_open.setStyleSheet(self.robot_idle_style)
                self.btn_m3_open.clicked.connect(self.reset_robot_open_flag)
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
                self.btn_m4_safebox.clicked.connect(self.reset_safebox_flag)
                left_vbox_m4.addWidget(self.btn_m4_safebox, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_human = QtWidgets.QPushButton("HUMAN: NO")
                self.btn_m4_human.setFixedSize(150, 44)
                self.btn_m4_human.setFont(common_font)
                self.btn_m4_human.setStyleSheet(self.det_idle_style)
                self.btn_m4_human.clicked.connect(self.reset_human_flag)
                left_vbox_m4.addWidget(self.btn_m4_human, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_finish = QtWidgets.QPushButton("FINISH: NO")
                self.btn_m4_finish.setFixedSize(150, 44)
                self.btn_m4_finish.setFont(common_font)
                self.btn_m4_finish.setStyleSheet(self.det_idle_style)
                self.btn_m4_finish.clicked.connect(self.reset_finish_flag)
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
                self.btn_m4_ok.clicked.connect(self.reset_robot_ok_flag)
                robot_vbox_m4.addWidget(self.btn_m4_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

                self.btn_m4_pick = QtWidgets.QPushButton("Pick")
                self.btn_m4_pick.setFixedSize(150, 44)
                self.btn_m4_pick.setFont(common_font)
                self.btn_m4_pick.setStyleSheet(self.robot_idle_style)
                self.btn_m4_pick.clicked.connect(self.reset_robot_pick_flag)
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
        # self.robot_m3_ok_flag = False # 필요 없을듯
        self.robot_m3_press_flag = False
        self.robot_m3_open_flag = False

        # self.robot_m4_ok_flag = False
        self.robot_m4_pick_flag = False

        self.robot_ok_flag = False
        self.robot_press_flag = False
        self.robot_open_flag = False
        self.robot_pick_flag = False

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
    @QtCore.Slot()
    def on_sos_detected(self):
        """SOS 감지 토픽을 받았을 때"""
        if not self.sos_flag:
            self.sos_flag = True
            self.btn_m1_sos.setText("SOS: YES")
            self.btn_m1_sos.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] SOS detected!")

    # Mission3 콜백들
    @QtCore.Slot()
    def on_button_detected(self):
        if not self.button_flag:
            self.button_flag = True
            self.btn_m3_button.setText("BUTTON: YES")
            self.btn_m3_button.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Button detected!")

    @QtCore.Slot()
    def on_fire_detected(self):
        self.fire_count += 1
        if hasattr(self, 'lbl_m3_fire_count'):
            self.lbl_m3_fire_count.setText(f"불의 개수: {self.fire_count} 개 ({self.fire_count}/3)")
        
        if not self.fire_flag:
            self.fire_flag = True
            self.btn_m3_fire.setText("FIRE: YES")
            self.btn_m3_fire.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Fire detected!")

    @QtCore.Slot()
    def on_door_detected(self):
        if not self.door_flag:
            self.door_flag = True
            self.btn_m3_door.setText("DOOR: YES")
            self.btn_m3_door.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Door detected!")

    @QtCore.Slot()
    def on_robot_ok(self):
        if not self.robot_ok_flag:
            self.robot_ok_flag = True
            if hasattr(self, 'btn_m3_ok'):
                self.btn_m3_ok.setStyleSheet(self.robot_active_style)
            if hasattr(self, 'btn_m4_ok'):
                self.btn_m4_ok.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] Robot OK!")

    @QtCore.Slot()
    def on_robot_press(self):
        if not self.robot_press_flag:
            self.robot_press_flag = True
            self.btn_m3_press.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] Robot Press!")

    @QtCore.Slot()
    def on_robot_open(self):
        if not self.robot_open_flag:
            self.robot_open_flag = True
            self.btn_m3_open.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] Robot Open!")

    # Mission4 콜백들
    @QtCore.Slot()
    def on_safebox_detected(self):
        if not self.safebox_flag:
            self.safebox_flag = True
            self.btn_m4_safebox.setText("SAFEBOX: YES")
            self.btn_m4_safebox.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Safebox detected!")

    @QtCore.Slot()
    def on_human_detected(self):
        self.human_count += 1
        if hasattr(self, 'lbl_m4_human_count'):
            self.lbl_m4_human_count.setText(f"사람: {self.human_count} 명 ({self.human_count}/3)")
        
        if not self.human_flag:
            self.human_flag = True
            self.btn_m4_human.setText("HUMAN: YES")
            self.btn_m4_human.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Human detected!")

    @QtCore.Slot()
    def on_finish_detected(self):
        if not self.finish_flag:
            self.finish_flag = True
            self.btn_m4_finish.setText("FINISH: YES")
            self.btn_m4_finish.setStyleSheet(self.det_active_style)
            self.append_log("[ROS2] Finish detected!")

    @QtCore.Slot()
    def on_robot_pick(self):
        if not self.robot_pick_flag:
            self.robot_pick_flag = True
            self.btn_m4_pick.setStyleSheet(self.robot_active_style)
            self.append_log("[ROS2] Robot Pick!")
    
    @QtCore.Slot(object)
    def on_frame_received(self, data):
        # data: (topic, frame_ndarray)
        try:
            _, frame = data
            self.main_cam.show_frame(frame)
        except Exception as e:
            self.append_log(f"[UI] frame handler err: {e}")

    @QtCore.Slot(str)
    def on_zmq_detection(self, canonical):
        # 'sos_detected', 'button_detected', ...
        if canonical == "sos_detected":
            self.on_sos_detected()
        elif canonical == "button_detected":
            self.on_button_detected()
        elif canonical == "fire_detected":
            self.on_fire_detected()
        elif canonical == "door_detected":
            self.on_door_detected()
        elif canonical == "safebox_detected":
            self.on_safebox_detected()
        elif canonical == "human_detected":
            self.on_human_detected()
        elif canonical == "finish_detected":
            self.on_finish_detected()

    # MainWindow 클래스 내부 어딘가(예: reset_* 함수들 위/아래)에 추가
    @QtCore.Slot(str)
    def append_log(self, s: str):
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, "log_list") and self.log_list is not None:
                self.log_list.addItem(f"{ts} {s}")
                self.log_list.scrollToBottom()
            else:
                # 혹시 log_list가 아직 없거나 None이면 콘솔로라도 출력
                print(f"{ts} {s}")
        except Exception as e:
            print(f"[UI] append_log error: {e}")

    # 리셋 함수들
    def reset_sos_flag(self):
        self.sos_flag = False
        self.btn_m1_sos.setText("SOS: NO")
        self.btn_m1_sos.setStyleSheet(self.det_idle_style)
        self.append_log("[UI] SOS flag reset")

    # Mission3 리셋 함수들
    def reset_button_flag(self):
        self.button_flag = False
        self.btn_m3_button.setText("BUTTON: NO")
        self.btn_m3_button.setStyleSheet(self.det_idle_style)
        self.append_log("[UI] Button flag reset")

    def reset_fire_flag(self):
        self.fire_flag = False
        self.fire_count = 0
        self.btn_m3_fire.setText("FIRE: NO")
        self.btn_m3_fire.setStyleSheet(self.det_idle_style)
        if hasattr(self, 'lbl_m3_fire_count'):
            self.lbl_m3_fire_count.setText("불의 개수: 0 개 (0/3)")
        self.append_log("[UI] Fire flag reset")

    def reset_door_flag(self):
        self.door_flag = False
        self.btn_m3_door.setText("DOOR: NO")
        self.btn_m3_door.setStyleSheet(self.det_idle_style)
        self.append_log("[UI] Door flag reset")

    def reset_robot_ok_flag(self):
        self.robot_ok_flag = False
        if hasattr(self, 'btn_m3_ok'):
            self.btn_m3_ok.setStyleSheet(self.robot_idle_style)
        if hasattr(self, 'btn_m4_ok'):
            self.btn_m4_ok.setStyleSheet(self.robot_idle_style)
        self.append_log("[UI] Robot OK flag reset")

    def reset_robot_press_flag(self):
        self.robot_press_flag = False
        self.btn_m3_press.setStyleSheet(self.robot_idle_style)
        self.append_log("[UI] Robot Press flag reset")

    def reset_robot_open_flag(self):
        self.robot_open_flag = False
        self.btn_m3_open.setStyleSheet(self.robot_idle_style)
        self.append_log("[UI] Robot Open flag reset")

    # Mission4 리셋 함수들
    def reset_safebox_flag(self):
        self.safebox_flag = False
        self.btn_m4_safebox.setText("SAFEBOX: NO")
        self.btn_m4_safebox.setStyleSheet(self.det_idle_style)
        self.append_log("[UI] Safebox flag reset")

    def reset_human_flag(self):
        self.human_flag = False
        self.human_count = 0
        self.btn_m4_human.setText("HUMAN: NO")
        self.btn_m4_human.setStyleSheet(self.det_idle_style)
        if hasattr(self, 'lbl_m4_human_count'):
            self.lbl_m4_human_count.setText("사람: 0 명 (0/3)")
        self.append_log("[UI] Human flag reset")

    def reset_finish_flag(self):
        self.finish_flag = False
        self.btn_m4_finish.setText("FINISH: NO")
        self.btn_m4_finish.setStyleSheet(self.det_idle_style)
        self.append_log("[UI] Finish flag reset")

    def reset_robot_pick_flag(self):
        self.robot_pick_flag = False
        self.btn_m4_pick.setStyleSheet(self.robot_idle_style)
        self.append_log("[UI] Robot Pick flag reset")

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

    # ZQM는 프레임만 -> frame_received만 연결 
    # ROS2는 모든 탐지/상태 -> 각 시그널을 기존 슬롯에 연결
    def start_stream(self):
        # 이미 실행 중이면 무시
        if (self.zmq_thread is not None and self.zmq_thread.isRunning()) or \
        (self.ros_thread is not None and self.ros_thread.isRunning()):
            self.append_log("[UI] Already running.")
            return

        # --- ZMQ (영상 전용) ---
        try:
            self.zmq_thread = ZmqSubscriberThread(
                endpoints=self.zmq_endpoints,
            )
            self.zmq_thread.text_received.connect(self.append_log)
            self.zmq_thread.frame_received.connect(self.on_frame_received)
            self.zmq_thread.start()
            self.append_log("[ZMQ] image subscriber started.")
        except Exception as e:
            self.append_log(f"[ZMQ] start error: {e}")

        # --- ROS2 (탐지/상태 전부) ---
        try:
            self.ros_thread = Ros2SubscriberThread()
            self.ros_thread.text_received.connect(self.append_log)

            # 탐지
            self.ros_thread.sos_detected.connect(self.on_sos_detected)
            self.ros_thread.button_detected.connect(self.on_button_detected)
            self.ros_thread.fire_detected.connect(self.on_fire_detected)
            self.ros_thread.door_detected.connect(self.on_door_detected)
            self.ros_thread.safebox_detected.connect(self.on_safebox_detected)
            self.ros_thread.human_detected.connect(self.on_human_detected)
            self.ros_thread.finish_detected.connect(self.on_finish_detected)

            # 로봇 상태
            self.ros_thread.robot_ok.connect(self.on_robot_ok)
            self.ros_thread.robot_press.connect(self.on_robot_press)
            self.ros_thread.robot_open.connect(self.on_robot_open)
            self.ros_thread.robot_pick.connect(self.on_robot_pick)

            self.ros_thread.start()
            self.append_log("[ROS2] all-topic subscriber started.")
        except Exception as e:
            self.append_log(f"[ROS2] start error: {e}")

    def stop_stream(self):
        # ZMQ stop
        if self.zmq_thread is not None and self.zmq_thread.isRunning():
            self.zmq_thread.stop()
            self.zmq_thread.wait(1000)
            self.append_log("[ZMQ] stopped.")
        # ROS stop
        if self.ros_thread is not None and self.ros_thread.isRunning():
            self.ros_thread.stop()
            self.ros_thread.wait(1000)
            self.append_log("[ROS2] stopped.")

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
