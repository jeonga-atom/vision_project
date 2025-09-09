from datetime import datetime
from typing import List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import base64, zmq, numpy as np, cv2

from .config import DEFAULT_ROS_TOPICS, UI_STYLES, MissionType
from .state import StateManager
from .camera_label import CameraLabel
from .zmq_subscriber import ZmqSubscriberThread
from .ros2_subscriber import Ros2SubscriberThread


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, zmq_endpoints: List[str], ros_topics=None):
        super().__init__()
        self.setWindowTitle("ZMQ + ROS2 Viewer (Optimized)")
        self.resize(1000, 700)
        self.state          = StateManager()            # 상태 관리자
        self.zmq_endpoints = zmq_endpoints              # 설정
        self.ros_topics = ros_topics or DEFAULT_ROS_TOPICS
        self.zmq_thread     = None
        self.ros_thread     = None
        self._init_ui()                                 # UI 초기화
        self._connect_signals()
        self._frame_pending = False
        self._fire_on       = 0
        self._fire_off      = 0
        self._human_red     = 0
        self._human_blue    = 0
        
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
            layout.addWidget(self._create_mission4_grid(), alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
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
        self.btn_m3_fire = self._create_button("FIRE: (0/4)")
        self.btn_m3_door = self._create_button("DOOR: NO")
        
        # 이벤트 연결
        self.btn_m3_button.clicked.connect(lambda: self._reset_flag('button_flag', self.btn_m3_button, 
                                                                    "BUTTON: NO", self.btn_m3_button_ok,
                                                                    'robot_ok_flag'))
        self.btn_m3_fire.clicked.connect(self._reset_fire_flag)
        self.btn_m3_door.clicked.connect(lambda: self._reset_flag('door_flag', self.btn_m3_door, 
                                                                  "DOOR: NO", self.btn_m3_open,
                                                                  'robot_open_flag'))
        layout.addWidget(self.btn_m3_button, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_fire,   alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_door,   alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission3_robot_buttons(self) -> QtWidgets.QWidget:
        """Mission 3 로봇 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.btn_m3_button_ok = self._create_button("ButtonEnd", 'robot')
        self.btn_m3_fire_done = self._create_button("FireEnd", 'robot')
        self.btn_m3_open      = self._create_button("OpenDoorEnd", 'robot')
        
        # 이벤트 연결
        self.btn_m3_button_ok.clicked.connect(lambda: self._reset_robot_flag('robot_ok_flag', [self.btn_m3_button_ok, getattr(self, 'btn_m4_ok', None)]))
        self.btn_m3_fire_done.clicked.connect(lambda: self._reset_robot_flag('robot_fire_done_flag', [self.btn_m3_fire_done]))
        self.btn_m3_open.clicked.connect(lambda: self._reset_robot_flag('robot_open_flag', [self.btn_m3_open]))
        
        layout.addWidget(self.btn_m3_button_ok, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_fire_done, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m3_open,      alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    def _create_mission4_robot_buttons(self) -> QtWidgets.QWidget:
        """Mission 4 로봇 버튼들 생성"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.btn_m4_pick_place = self._create_button("PickAndPlaceEnd", 'robot')
        self.btn_m4_human_end = self._create_button("HumanEnd", 'robot')
        
        # 이벤트 연결
        self.btn_m4_pick_place.clicked.connect(lambda: self._reset_robot_flag('robot_pick_place_flag', [self.btn_m4_pick_place]))
        self.btn_m4_human_end.clicked.connect(lambda: self._reset_robot_flag('robot_human_done_flag', [self.btn_m4_human_end]))
        
        layout.addWidget(self.btn_m4_pick_place, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.btn_m4_human_end, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(150)
        return widget
    
    # 버튼의 위치를 만들기 위한 함수
    def _create_mission4_grid(self) -> QtWidgets.QWidget: 
        g = QtWidgets.QGridLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)

        # 빨간(감지) 버튼들
        self.btn_m4_safebox  = self._create_button("SAFEBOX: NO")
        self.btn_m4_safebox2 = self._create_button("SAFEBOX_2: NO")
        self.btn_m4_human    = self._create_button("HUMAN: (0/3)")
        self.btn_m4_finish   = self._create_button("FINISH: NO")

        # 흰(로봇 완료) 버튼들
        self.btn_m4_pick_place = self._create_button("PickAndPlaceEnd", 'robot')
        self.btn_m4_human_end  = self._create_button("HumanEnd", 'robot')

        # ----- 배치: 같은 행(row)에 짝 배치 -----
        g.addWidget(self.btn_m4_safebox,    0, 0)
        g.addWidget(self.btn_m4_safebox2,   1, 0)
        g.addWidget(self.btn_m4_pick_place, 1, 1)

        g.addWidget(self.btn_m4_human,      2, 0)
        g.addWidget(self.btn_m4_human_end,  2, 1)

        g.addWidget(self.btn_m4_finish,     3, 0)

        # ----- 클릭(리셋) 연결: 기존 로직 그대로 -----
        self.btn_m4_safebox.clicked.connect(
            lambda: self._reset_flag('safebox_flag',  self.btn_m4_safebox,
                                    "SAFEBOX: NO",   self.btn_m4_pick_place,
                                    'robot_pick_place_flag')
        )
        self.btn_m4_safebox2.clicked.connect(
            lambda: self._reset_flag('safebox2_flag', self.btn_m4_safebox2,
                                    "SAFEBOX_2: NO", self.btn_m4_pick_place,
                                    'robot_pick_place_flag')
        )
        self.btn_m4_human.clicked.connect(self._reset_human_flag)  # 이미 쓰고 있는 함수 이름 유지
        self.btn_m4_finish.clicked.connect(
            lambda: self._reset_flag('finish_flag', self.btn_m4_finish, "FINISH: NO")
        )
        self.btn_m4_pick_place.clicked.connect(
            lambda: self._reset_robot_flag('robot_pick_place_flag', [self.btn_m4_pick_place])
        )
        self.btn_m4_human_end.clicked.connect(
            lambda: self._reset_robot_flag('robot_human_done_flag', [self.btn_m4_human_end])
        )

        w = QtWidgets.QWidget()
        w.setLayout(g)
        w.setFixedWidth(336)  # mission3 폭과 맞추고 싶으면 유지
        return w
    
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
    def _reset_flag(self, flag_name: str, button: QtWidgets.QPushButton, idle_text: str, 
                    partner_robot_btn: QtWidgets.QPushButton = None,
                    partner_robot_flag: str = None):
        """공통 플래그 리셋 함수"""
        self.state.set_flag(flag_name, False)
        button.setText(idle_text)
        button.setStyleSheet(UI_STYLES.DET_IDLE)
        if partner_robot_btn is not None:
            partner_robot_btn.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        if partner_robot_flag:
            self.state.set_flag(partner_robot_flag, False)
        self._append_log(f"[UI] {flag_name} reset")
    
    def _reset_robot_flag(self, flag_name: str, buttons: List[QtWidgets.QPushButton]):
        """로봇 플래그 리셋 함수"""
        self.state.set_flag(flag_name, False)
        for btn in buttons:
            if btn is not None:
                btn.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        self._append_log(f"[UI] {flag_name} reset")
    
    def _reset_fire_flag(self):
        """mission3 불 플래그/카운터 리셋"""
        self.state.set_flag('fire_flag', False)
        self.state.reset_counter('fire_count')
        self.btn_m3_fire.setText("FIRE: (0/4)")
        self.btn_m3_fire.setStyleSheet(UI_STYLES.DET_IDLE)

        # on/off 카운터 초기화
        self._fire_on = 0
        self._fire_off = 0

        self.state.set_flag('robot_fire_done_flag', False)
        if hasattr(self, 'btn_m3_fire_done'):
            self.btn_m3_fire_done.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        self._append_log("[UI] Fire flag and counter reset")
    
    def _reset_human_flag(self):
        """mission4 사람 플래그 특별 리셋"""
        self.state.set_flag('human_flag', False)
        self.btn_m4_human.setText("HUMAN: (0/3)")
        self.btn_m4_human.setStyleSheet(UI_STYLES.DET_IDLE)

        self._human_red = 0
        self._human_blue = 0

        self.state.set_flag('robot_human_done_flag', False)
        if hasattr(self, 'btn_m4_human_end'):
            self.btn_m4_human_end.setStyleSheet(UI_STYLES.ROBOT_IDLE)
        self._append_log("[UI] Human reset")
    
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
    
    def _decode_frame(self, payload_bytes: bytes) -> Optional[np.ndarray]:
        if not payload_bytes:
            return None
        # 아주 가벼운 휴리스틱: ASCII 비율과 '=' 패딩 존재 확인
        is_base64ish = b'=' in payload_bytes and all((32 <= b <= 122) or b in (9,10,13) for b in payload_bytes[:64])
        try:
            if is_base64ish:
                jpg_bytes = base64.b64decode(payload_bytes, validate=False)
            else:
                jpg_bytes = payload_bytes
            np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.emit_log(f"[ZMQ] Frame decode failed: {e}")
            return None
    
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

    @QtCore.Slot(object)
    def _on_frame_received(self, data):
        if self._frame_pending:
            return
        self._frame_pending = True
        QtCore.QTimer.singleShot(0, lambda: self._consume_frame(data))

    def _consume_frame(self, data):
        try:
            _, frame = data
            self.main_cam.show_frame(frame)
        finally:
            self._frame_pending = False

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
            self._append_log("[ROS2] SOS인식 완료! SOS detected!")
    
    ################# mission3의 버튼, 불, 문 인식 ##################
    @QtCore.Slot() 
    def _on_button_detected(self):
        """버튼 감지"""
        if not self.state.get_flag('button_flag'):
            self.state.set_flag('button_flag', True)
            self.btn_m3_button.setText("BUTTON: YES")
            self.btn_m3_button.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] 버튼 인식 완료! Button detected!")
    
    @QtCore.Slot()
    def _on_fire_on(self):
        # 총합이 4 미만일 때만 증가 (중복 방지)
        if self._fire_on + self._fire_off < 4:
            self._fire_on += 1
        self._after_fire_update()

    @QtCore.Slot()
    def _on_fire_off(self):
        if self._fire_on + self._fire_off < 4:
            self._fire_off += 1
        self._after_fire_update()

    def _after_fire_update(self):
        total = self._fire_on + self._fire_off

        # 감지 시작 시 버튼을 활성(노랑)으로
        if not self.state.get_flag('fire_flag'):
            self.state.set_flag('fire_flag', True)
            self.btn_m3_fire.setStyleSheet(UI_STYLES.DET_ACTIVE)

        # 버튼 텍스트는 on+off 합산으로 표시
        self.btn_m3_fire.setText(f"FIRE: ({total}/4)")

        # 터미널 로그: 켜진/꺼진 각각 출력
        self._append_log(f"[ROS2] on: {self._fire_on}      off:{self._fire_off}")

        # 완료 처리
        if total >= 4 and not self.state.get_flag('robot_fire_done_flag'):
            self.state.set_flag('robot_fire_done_flag', True)
            if hasattr(self, 'btn_m3_fire_done'):
                self.btn_m3_fire_done.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] 불 인식 완료!(4/4)-> Robot Fire Done!")

    @QtCore.Slot()
    def _on_door_detected(self):
        """문 감지"""
        if not self.state.get_flag('door_flag'):
            self.state.set_flag('door_flag', True)
            self.btn_m3_door.setText("DOOR: YES")
            self.btn_m3_door.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] 문 인식 완료! Door detected!")
    
    ############ mission4의 보급상자, 사람, finish 라인 인식 ###########
    @QtCore.Slot()
    def _on_safebox_detected(self):
        """보급상자"""
        if not self.state.get_flag('safebox_flag'):
            self.state.set_flag('safebox_flag', True)
            self.btn_m4_safebox.setText("SAFEBOX: YES")
            self.btn_m4_safebox.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] 보급박스 1차 인식 완료! Safebox detected!")

    @QtCore.Slot()
    def _on_safebox2_detected(self):
        """보급상자"""
        if not self.state.get_flag('safebox2_flag'):
            self.state.set_flag('safebox2_flag', True)
            self.btn_m4_safebox2.setText("SAFEBOX_2: YES")
            self.btn_m4_safebox2.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] 보급박스 2차 인식 완료! Safebox_2 detected!")
    
    @QtCore.Slot()
    def _on_human_red(self):
        # 총합 3 미만일 때만 증가
        if self._human_red + self._human_blue < 3:
            self._human_red += 1
        self._after_human_update()

    @QtCore.Slot()
    def _on_human_blue(self):
        if self._human_red + self._human_blue < 3:
            self._human_blue += 1
        self._after_human_update()

    def _after_human_update(self):
        total = self._human_red + self._human_blue

        # 최초 감지 시 버튼 활성(노랑)
        if not self.state.get_flag('human_flag'):
            self.state.set_flag('human_flag', True)
            self.btn_m4_human.setStyleSheet(UI_STYLES.DET_ACTIVE)

        # 버튼 텍스트는 red+blue 합산으로
        self.btn_m4_human.setText(f"HUMAN: ({total}/3)")

        # 터미널 로그: 빨강/파랑 각각 출력
        self._append_log(f"[ROS2] 사망: {self._human_red}      생존: {self._human_blue}")

        # 완료 처리(3/3 도달)
        if total >= 3 and not self.state.get_flag('robot_human_done_flag'):
            self.state.set_flag('robot_human_done_flag', True)
            if hasattr(self, 'btn_m4_human_end'):
                self.btn_m4_human_end.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] 사람 인식 완료! (3/3) -> Human detect done")
    
    @QtCore.Slot()
    def _on_pick_place_end(self):
        if not self.state.get_flag('robot_pick_place_flag'):
            self.state.set_flag('robot_pick_place_flag', True)
            if hasattr(self, 'btn_m4_pick_place'):
                self.btn_m4_pick_place.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] 보급상자 로봇팔 pick and place 완료!")
    
    @QtCore.Slot()
    def _on_finish_detected(self):
        """완료 감지"""
        if not self.state.get_flag('finish_flag'):
            self.state.set_flag('finish_flag', True)
            self.btn_m4_finish.setText("FINISH: YES")
            self.btn_m4_finish.setStyleSheet(UI_STYLES.DET_ACTIVE)
            self._append_log("[ROS2] Finish라인 인식 완료! Finish detected!")

    @QtCore.Slot()
    def _on_robot_button_end(self):
        """로봇암: 버튼 작업 완료 → ButtonEnd 초록"""
        if not self.state.get_flag('robot_ok_flag'):
            self.state.set_flag('robot_ok_flag', True)
            if hasattr(self, 'btn_m3_button_ok'):
                self.btn_m3_button_ok.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            # (mission4 OK 박스도 함께 초록으로 하고 싶으면 아래 켜기)
            # if hasattr(self, 'btn_m4_ok'):
            #     self.btn_m4_ok.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] 로봇팔 버튼 누르기 완료! -> Button Press End")

    @QtCore.Slot()
    def _on_robot_open_door_end(self):
        """로봇암: 문 열기 완료 → OpenDoor 초록"""
        if not self.state.get_flag('robot_open_flag'):
            self.state.set_flag('robot_open_flag', True)
            if hasattr(self, 'btn_m3_open'):
                self.btn_m3_open.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
            self._append_log("[ROS2] 로봇팔 문 열기 완료! -> Door Open End")
    
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
            self.btn_m4_pick_place.setStyleSheet(UI_STYLES.ROBOT_ACTIVE)
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
            self.ros_thread.fire_on_detected.connect(self._on_fire_on)
            self.ros_thread.fire_off_detected.connect(self._on_fire_off)
            self.ros_thread.door_detected.connect(self._on_door_detected)
            self.ros_thread.safebox_detected.connect(self._on_safebox_detected)
            self.ros_thread.safebox2_detected.connect(self._on_safebox2_detected)
            self.ros_thread.human_red_detected.connect(self._on_human_red)
            self.ros_thread.human_blue_detected.connect(self._on_human_blue)
            self.ros_thread.finish_detected.connect(self._on_finish_detected)
            # self.ros_thread.fire_on.connect(self._on_fire_on)
            # self.ros_thread.fire_off.connect(self._on_fire_off)
            
            # 로봇 상태 시그널 연결
            self.ros_thread.robot_ok.connect(self._on_robot_ok)
            self.ros_thread.robot_open.connect(self._on_robot_open)
            self.ros_thread.robot_pick.connect(self._on_robot_pick)

            # --- mission3 전용 END 시그널 ---
            self.ros_thread.robot_button_end.connect(self._on_robot_button_end)
            self.ros_thread.robot_open_door_end.connect(self._on_robot_open_door_end)
            
            # --- mission4 전용 END 시그널 ---
            self.ros_thread.pick_place_end.connect(self._on_pick_place_end)

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