from PySide6 import QtCore
from .thread_base import BaseSubscriberThread
from std_msgs.msg import Empty
# import rclpy


class Ros2SubscriberThread(BaseSubscriberThread):
    # 탐지 시그널들
    sos_detected        = QtCore.Signal()
    button_detected     = QtCore.Signal()
    fire_detected       = QtCore.Signal()
    door_detected       = QtCore.Signal()
    safebox_detected    = QtCore.Signal()
    safebox2_detected   = QtCore.Signal()
    human_tick          = QtCore.Signal()
    finish_detected     = QtCore.Signal()
    
    # 로봇 상태 시그널들
    robot_ok            = QtCore.Signal()
    robot_open          = QtCore.Signal()
    robot_pick          = QtCore.Signal()
    robot_button_end    = QtCore.Signal()
    robot_open_door_end = QtCore.Signal()
    pick_place_end      = QtCore.Signal()

    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._node = None
        self._subs = []
        
    def _create_subscriptions(self):
        """구독 생성 """
        # qos = rclpy.qos.QoSProfile(depth=1)
        # 탐지 토픽들
        detection_topics = {
            # mission 1
            '/object_detection/SOS': self.sos_detected,
            # mission 3
            '/object_detection/button': self.button_detected,
            '/object_detection/fire/on': self.fire_detected,
            '/object_detection/fire/off': self.fire_detected,
            '/object_detection/door': self.door_detected,
            #mission 4
            '/object_detection/safebox': self.safebox_detected,
            '/object_detection/safebox_2': self.safebox2_detected,
            '/object_detection/human/red': self.human_tick,
            '/object_detection/human/blue': self.human_tick,
            '/object_detection/finish': self.finish_detected,
        }
        # ----------- 로봇 상태 토픽들 -----------------
        robot_topics = {
            '/robot/ok': self.robot_ok,
            '/robot/open': self.robot_open,
            '/robot/pick': self.robot_pick,
        }
        # ---------- mission3 로봇암 완료 토픽 ----------
        robot_arm_topics = {
            '/robot_arm/button_end'   : self.robot_button_end,
            '/robot_arm/open_door_end': self.robot_open_door_end,
            '/robot_arm/pick_and_place_end': self.pick_place_end,
        }

        # 모든 토픽 구독
        all_topics = {**detection_topics, **robot_topics, **robot_arm_topics}
        
        for topic, signal in all_topics.items():
            try:
                sub = self._node.create_subscription(
                    Empty, topic,
                    lambda _msg, sig=signal: sig.emit(),
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