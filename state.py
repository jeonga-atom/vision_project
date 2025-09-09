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
            'safebox2_flag': False,
            'human_flag': False,
            'finish_flag': False
        }
        
        # 로봇 상태 플래그들
        self.robot_flags = {
            'robot_ok_flag': False,
            'robot_open_flag': False,
            'robot_pick_flag': False,
            'robot_pick_place_flag': False,
            'robot_fire_done_flag': False,
            'robot_human_done_flag': False,
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