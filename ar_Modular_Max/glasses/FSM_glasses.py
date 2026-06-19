class GlassesFSM:
    """
    眼镜端状态机（Finite State Machine）类
    
    这个类管理AR眼镜的显示状态，根据SLAM和AI事件决定渲染内容
    三个状态：
    1. DISPLAY: 正常显示状态，渲染AR内容
    2. LOADING: 加载状态，显示等待提示
    3. TRACKING_LOST: 跟踪丢失状态，清空画面
    
    这个状态机确保用户在不同情况下有清晰的视觉反馈
    """
    
    # 定义状态常量（类属性）
    STATE_DISPLAY = "DISPLAY"       # 正在渲染（正常显示）
    STATE_LOADING = "LOADING"       # 等待中（显示加载动画，如转圈）
    STATE_LOST = "TRACKING_LOST"    # 丢帧（SLAM丢失空间，清空画面）

    def __init__(self):
        """
        初始化眼镜状态机
        
        创建新的状态机实例，初始状态为DISPLAY
        表示眼镜刚启动，正常显示AR内容
        """
        # 设置当前状态为显示状态
        # 当状态机创建时，它处于显示状态，可以正常渲染AR内容
        self.state = self.STATE_DISPLAY

    def update_render(self, event):
        """
        根据事件决定渲染策略
        
        参数:
        event: 触发事件，字符串类型，可以是：
               'SLAM_SUCCESS' - SLAM成功，跟踪正常
               'SLAM_FAILED' - SLAM失败，跟踪丢失
               'AI_WAITING' - AI处理中，需要等待
        
        返回:
        要执行的渲染指令（字符串），如果事件在当前状态下无效，返回None
        
        这个方法根据事件来切换状态并决定渲染策略
        这确保了用户在不同情况下都有合适的视觉反馈
        """
        # 如果收到SLAM_SUCCESS事件
        if event == 'SLAM_SUCCESS':
            # 切换到显示状态
            self.state = self.STATE_DISPLAY
            # 返回渲染AR物体的指令
            return 'RENDER_AR_OBJECT'
        
        # 如果收到SLAM_FAILED事件
        elif event == 'SLAM_FAILED':
            # 切换到跟踪丢失状态
            self.state = self.STATE_LOST
            # 返回清空屏幕的指令
            return 'CLEAR_SCREEN'
            
        # 如果收到AI_WAITING事件
        elif event == 'AI_WAITING':
            # 切换到加载状态
            self.state = self.STATE_LOADING
            # 返回显示加载指示器的指令
            return 'SHOW_LOADING_INDICATOR'
        
        # 如果事件在当前状态下无效，返回None
        return None