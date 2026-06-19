# 导入必要的Python库
import socket  # 提供网络通信功能
import struct  # 用于处理二进制数据的打包和解包

class BoxTransport:
    """
    计算盒传输层类，负责与云端服务器的网络通信
    
    这个类封装了客户端socket功能，用于：
    1. 连接到云端服务器
    2. 发送图片数据到云端
    3. 接收云端返回的AI推理结果
    """
    
    def __init__(self, host='127.0.0.1', port=8888):
        """
        初始化传输层
        
        参数:
        host: 云端服务器的主机地址，默认127.0.0.1表示本机
        port: 云端服务器的端口号，默认8888
        """
        # 服务器地址
        self.host = host
        
        # 服务器端口
        self.port = port
        
        # 客户端socket对象，初始为None
        # 这个socket用于与云端服务器通信
        self.client_socket = None

    def connect(self):
        """
        尝试建立连接到云端服务器
        
        返回:
        bool: 连接成功返回True，失败返回False
        """
        try:
            # 创建TCP socket对象
            # AF_INET: 使用IPv4地址族
            # SOCK_STREAM: 使用TCP协议
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 连接到指定的服务器地址和端口
            # 这是一个阻塞调用，会一直尝试连接直到成功或失败
            self.client_socket.connect((self.host, self.port))
            
            # 打印连接成功信息
            print(f"[Transport] 成功连接至云端 {self.host}:{self.port}")
            return True
            
        except Exception as e:
            # 捕获连接过程中的所有异常
            print(f"[Transport] 连接云端失败: {e}")
            return False

    def send_and_receive(self, data):
        """
        核心函数：发送图片并同步等待结果
        
        参数:
        data: 要发送的图片二进制数据
        
        返回:
        str: 云端返回的JSON结果字符串，如果失败则返回None
        
        这个函数实现了完整的请求-响应流程：
        1. 发送数据长度
        2. 发送实际数据
        3. 接收云端响应
        4. 处理异常和重连
        """
        # 检查socket是否存在，如果不存在则尝试连接
        if not self.client_socket:
            # 尝试连接，如果连接失败则返回None
            if not self.connect(): 
                return None

        try:
            # 1. 发送数据长度 (防粘包，先发4字节长度)
            # 计算数据的字节长度
            data_len = len(data)
            
            # 使用struct.pack将整数打包为4字节的大端字节序
            # '>I': 大端字节序的无符号整数
            # 这4个字节表示后面要发送的数据的长度
            # 这种设计解决了TCP粘包问题
            self.client_socket.sendall(struct.pack('>I', data_len))
            
            # 2. 发送实际图片数据
            # sendall确保所有数据都被发送
            # 这比send更可靠，send可能只发送部分数据
            self.client_socket.sendall(data)
            
            # 3. 等待接收云端返回的JSON结果
            # 接收最多1024字节的数据
            # 假设云端返回的数据不超过1024字节
            # 注意：这里可能接收不全，特别是当响应数据较大时
            result = self.client_socket.recv(1024).decode('utf-8')
            
            # 返回解码后的字符串
            return result
            
        except Exception as e:
            # 捕获发送/接收过程中的所有异常
            print(f"[Transport] 发送失败，尝试重连: {e}")
            
            # 清空连接，以便下次重连
            # 这样下次调用send_and_receive时会自动尝试重新连接
            self.client_socket = None
            
            return None