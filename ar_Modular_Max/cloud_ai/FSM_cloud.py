class CloudFSM:
    """
    云端状态机（Finite State Machine）类
    
    这个类管理云端AI服务的状态，确保服务的稳定性和可靠性
    状态机有三个状态：
    1. READY: 准备就绪，可以接收新任务
    2. INFERENCE: 推理中，正在处理任务
    3. ERROR: 错误状态，需要人工干预
    
    使用状态机的好处：
    1. 防止并发请求导致的问题
    2. 清晰的状态管理
    3. 错误恢复机制
    """
    
    # 定义状态常量（类属性）
    STATE_READY = "READY"         # 可以接收图片并进行推理
    STATE_INFERENCE = "INFERENCE" # 正在计算，此时不应接收新图片
    STATE_ERROR = "ERROR"         # 模型出错或硬件故障

    def __init__(self):
        """
        初始化状态机
        
        创建一个新的状态机实例，初始状态为READY
        表示服务刚启动，可以接收任务
        """
        # 设置当前状态为就绪状态
        # 当状态机创建时，它处于就绪状态，可以立即开始工作
        self.current_state = self.STATE_READY

    def set_state(self, new_state):
        """
        切换状态机的状态
        
        参数:
        new_state: 新的状态，必须是类中定义的STATE_常量之一
        
        这个方法不仅改变内部状态，还会记录状态切换日志
        这对于调试和监控非常重要
        """
        # 更新当前状态
        self.current_state = new_state
        
        # 打印状态切换日志
        # 在实际应用中，这可以替换为更复杂的日志记录
        print(f"[FSM] 云端状态切换为: {self.current_state}")

    def can_process(self):
        """
        判断是否可以接受新任务
        
        返回:
        bool: 如果当前状态是READY，返回True，否则返回False
        
        这个方法用于检查状态机是否能够处理新的请求
        只有处于READY状态时才能接收新任务
        """
        # 检查当前状态是否为就绪状态
        return self.current_state == self.STATE_READY

    def process_request(self, callback_func, *args):
        """
        状态机驱动的推理函数
        
        参数:
        callback_func: 实际的AI推理函数，将被调用的函数
        *args: 传递给callback_func的参数
        
        返回:
        推理函数的返回值，如果状态机忙碌则返回None
        
        这个方法封装了推理过程，确保：
        1. 只有在就绪状态时才处理请求
        2. 处理过程中状态变为忙碌
        3. 处理完成后状态恢复为就绪
        4. 发生错误时进入错误状态
        
        这个设计防止了：
        1. 同时处理多个请求导致的资源竞争
        2. 在错误状态下继续处理请求
        3. 状态不一致导致的问题
        """
        # 首先检查是否可以处理新请求
        if not self.can_process():
            # 如果状态机不在就绪状态，打印警告并返回None
            print("[FSM] 警告：云端忙碌中，拒绝请求")
            return None

        # 如果可以处理，切换到忙碌状态
        # 这确保在推理过程中不会接收新请求
        self.set_state(self.STATE_INFERENCE)
        
        try:
            # 执行实际推理
            # callback_func是用户提供的推理函数
            # *args是传递给这个函数的参数
            result = callback_func(*args)
            
            # 返回推理结果
            return result
            
        except Exception as e:
            # 如果在推理过程中发生异常
            # 将状态设置为错误状态
            self.set_state(self.STATE_ERROR)
            
            # 重新抛出异常，让调用者处理
            raise e
            
        finally:
            # 无论是否发生异常，都会执行这个代码块
            # 这是Python的finally语句，确保总是执行
            
            # 注意：如果发生异常，状态已经设置为ERROR
            # 所以这里我们需要检查当前状态
            if self.current_state != self.STATE_ERROR:
                # 如果状态不是ERROR，说明推理成功完成
                # 恢复就绪状态，准备接收新请求
                self.set_state(self.STATE_READY)