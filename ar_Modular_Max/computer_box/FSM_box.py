class BoxFSM:
    """
    计算盒状态机（Finite State Machine）类
    
    这个类管理计算盒的工作状态，实现事件驱动的状态转换
    主要解决网络延迟和异常情况下的降级处理
    
    四个状态：
    1. IDLE: 空闲状态，等待相机帧到来
    2. SENDING: 发送状态，正在向云端发送图片并等待结果
    3. PROCESSING: 本地处理状态，网络延迟过高时切换到本地轻量模型
    4. RETRY: 重试状态，网络异常时尝试重新连接
    
    这个状态机确保计算盒在不同的网络条件下都能正常工作
    """
    
    # 定义状态常量（类属性）
    STATE_IDLE = "IDLE"           # 空闲，等待相机帧
    STATE_SENDING = "SENDING"     # 正在向云端发送图片，等待结果
    STATE_PROCESSING = "LOCAL_AI" # 网络延迟过高，切换到本地轻量模型处理
    STATE_RETRY = "RETRY"         # 网络异常，尝试重连

    def __init__(self):
        """
        初始化计算盒状态机
        
        创建新的状态机实例，初始状态为IDLE
        表示计算盒刚启动，等待接收相机帧
        """
        # 设置当前状态为空闲状态
        # 当状态机创建时，它处于空闲状态，可以接收相机帧
        self.state = self.STATE_IDLE

    def handle_event(self, event, *args):
        """
        事件驱动的状态转换逻辑
        
        参数:
        event: 触发事件，字符串类型，如 'FRAME_ARRIVED', 'NET_TIMEOUT', 'RESULT_RECEIVED'
        *args: 可变参数，传递事件的额外参数
        
        返回:
        要执行的动作（字符串），如果事件在当前状态下无效，返回None
        
        这个方法根据当前状态和事件来决定状态转换和要执行的动作
        这是状态机的核心逻辑
        """
        # 打印当前状态和收到的事件，用于调试
        print(f"[FSM] 当前状态: {self.state}, 收到事件: {event}")

        # 状态1: IDLE（空闲状态）
        if self.state == self.STATE_IDLE:
            # 在空闲状态下，如果收到FRAME_ARRIVED事件
            if event == 'FRAME_ARRIVED':
                # 状态转换：从IDLE转换到SENDING
                self.state = self.STATE_SENDING
                # 返回要执行的动作：发送到云端
                return 'SEND_TO_CLOUD'
            # 注意：这里只处理了FRAME_ARRIVED事件
            # 空闲状态下可能还会收到其他事件，如手动启动处理等
        
        # 状态2: SENDING（发送中状态）
        elif self.state == self.STATE_SENDING:
            # 在发送状态下，如果收到RESULT_RECEIVED事件
            if event == 'RESULT_RECEIVED':
                # 状态转换：从SENDING转换到IDLE
                self.state = self.STATE_IDLE
                # 返回要执行的动作：显示结果
                return 'DISPLAY_RESULT'
            
            # 在发送状态下，如果收到NET_TIMEOUT事件
            elif event == 'NET_TIMEOUT':
                # 状态转换：从SENDING转换到PROCESSING
                self.state = self.STATE_PROCESSING
                # 返回要执行的动作：运行本地AI
                return 'RUN_LOCAL_AI'
            # 注意：这里没有处理RETRY状态
            # 可能需要添加：elif event == 'NET_ERROR': 然后转换到RETRY状态
        
        # 状态3: PROCESSING（本地处理状态）
        elif self.state == self.STATE_PROCESSING:
            # 在处理状态下，如果收到COMPUTATION_DONE事件
            if event == 'COMPUTATION_DONE':
                # 状态转换：从PROCESSING转换到IDLE
                self.state = self.STATE_IDLE
                # 返回要执行的动作：显示结果
                return 'DISPLAY_RESULT'
            # 注意：本地处理可能失败，可能需要添加错误处理
        
        # 状态4: RETRY（重试状态）
        # 注意：原代码中没有实现RETRY状态的处理逻辑
        # 如果需要支持重试，可以在这里添加
        # elif self.state == self.STATE_RETRY:
        #     if event == 'RECONNECT_SUCCESS':
        #         self.state = self.STATE_IDLE
        #         return 'CONTINUE_NORMAL'
        #     elif event == 'RECONNECT_FAILED':
        #         self.state = self.STATE_PROCESSING
        #         return 'RUN_LOCAL_AI'

        # 如果事件在当前状态下无效，返回None
        return None