from dataclasses import dataclass
from enum import Enum

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