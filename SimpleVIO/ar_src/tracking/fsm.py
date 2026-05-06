import enum
import time

class SystemState(enum.Enum):
    INIT = 0        # 初始化：正在寻找特征点
    TRACKING = 1    # 正常追踪：视觉定位很稳
    LOST = 2        # 视觉丢失：靠 IMU 盲操续命
    RECOVERING = 3  # 重定位：尝试在历史记忆里找回位置

class TrackingFSM:
    def __init__(self):
        self.state = SystemState.INIT
        self.lost_time = 0
        self.max_lost_duration = 5.0  # 丢失超过5秒就彻底重置系统

    def update_state(self, tracking_ok, recovery_ok=False):
        """
        根据当前每一帧的追踪结果，决定系统状态的跳转
        :param tracking_ok: 本帧视觉 PnP 是否成功
        :param recovery_ok: 本帧回环检测/重定位是否找回
        """
        old_state = self.state

        # ---------------- 状态跳转逻辑 ----------------
        if self.state == SystemState.INIT:
            if tracking_ok:
                self.state = SystemState.TRACKING
            
        elif self.state == SystemState.TRACKING:
            if not tracking_ok:
                self.state = SystemState.LOST
                self.lost_time = time.time()
                
        elif self.state == SystemState.LOST:
            if tracking_ok: 
                self.state = SystemState.TRACKING
            elif time.time() - self.lost_time > self.max_lost_duration:
                # 丢太久了，IMU也救不回来，重置
                self.state = SystemState.INIT
            else:
                # 只要还没超时，就进入恢复模式尝试找回
                self.state = SystemState.RECOVERING
                
        elif self.state == SystemState.RECOVERING:
            if tracking_ok or recovery_ok:
                self.state = SystemState.TRACKING
            else:
                # 还没找回来，继续回 LOST 状态
                self.state = SystemState.LOST
        
        # 如果状态变了，打印一下，方便演示
        if old_state != self.state:
            print(f">>> [状态切换]: {old_state.name} -> {self.state.name}")
            
        return self.state

    def get_state(self):
        return self.state