# 从自定义模块导入组件
from ai_engine import AIEngine      # AI引擎，负责模型推理
from FSM_cloud import CloudFSM      # 状态机，管理云端服务状态
from Transport_cloud import CloudTransport  # 传输层，管理网络通信

class CloudRuntime:
    """
    云端运行时主类，负责协调AI引擎、状态机和传输层
    
    这个类是云端AI服务的核心，它：
    1. 初始化各个子系统
    2. 协调子系统之间的交互
    3. 提供请求处理逻辑
    4. 启动整个云端服务
    """
    
    def __init__(self, model_path):
        """
        初始化云端运行时系统
        
        参数:
        model_path: ONNX模型文件路径
        """
        # 1. 实例化各个核心模块
        # 创建AI引擎实例，加载ONNX模型
        # 这个引擎负责处理实际的AI推理任务
        self.engine = AIEngine(model_path)
        
        # 创建状态机实例
        # 状态机管理云端服务的状态（就绪、忙碌、错误等）
        # 防止并发请求冲突，提供状态管理和错误恢复
        self.fsm = CloudFSM()
        
        # 创建传输层实例，监听8888端口
        # 传输层负责网络通信，接收客户端连接和数据
        self.transport = CloudTransport(port=8888)
        
        # 打印初始化完成信息
        print("[Runtime] 云端系统初始化完成...")

    def handle_request(self, data):
        """
        这个函数会被 Transport 层调用 (回调)
        它通过 FSM 协调 Engine 进行处理
        
        参数:
        data: 接收到的二进制图片数据
        
        返回:
        处理结果的JSON字符串
        """
        # 利用 FSM 处理推理请求
        # process_request 是状态机的方法，它确保：
        # 1. 只有就绪状态才处理请求
        # 2. 处理过程中状态为忙碌
        # 3. 处理完成后恢复就绪状态
        # 4. 发生错误时进入错误状态
        
        # 参数说明:
        # self.engine.predict: 要执行的实际推理函数
        # data: 传递给predict函数的参数（图片数据）
        result = self.fsm.process_request(self.engine.predict, data)
        
        # 如果结果不为None，返回结果
        if result:
            return result
        else:
            # 如果 FSM 返回 None，说明云端正忙，返回错误提示给计算盒
            # 这种情况发生在状态机不在就绪状态时
            return '{"error": "Cloud busy, please retry"}'

    def run(self):
        """
        启动云端运行时系统
        
        这个方法启动传输层服务器，开始监听客户端连接
        传输层会为每个连接创建独立线程，并调用handle_request处理请求
        """
        # 启动 Transport 开始监听，并将 handle_request 传给它
        # 当传输层接收到数据时，会调用handle_request方法
        self.transport.start_server(on_receive_callback=self.handle_request)

# 主程序入口
# 当这个文件被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建CloudRuntime实例
    # 参数: 'models/mobilenet.onnx' - ONNX模型文件路径
    runtime = CloudRuntime('models/mobilenet.onnx')
    
    # 启动云端服务
    runtime.run()