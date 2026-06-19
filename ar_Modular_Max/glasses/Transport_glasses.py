# 导入必要的Python库
import socket  # 提供网络通信功能
import struct  # 用于处理二进制数据的打包和解包

class GlassesTransport:
    """
    眼镜传输层类，负责与计算盒的网络通信
    
    这个类封装了客户端socket功能，用于：
    1. 连接到计算盒服务器
    2. 接收计算盒发送的实时数据（位姿+AI结果）
    3. 处理网络异常和重连
    
    眼镜作为客户端，计算盒作为服务器
    数据流：计算盒 → 眼镜
    """
    
    def __init__(self, host='127.0.0.1', port=9999):
        """
        初始化眼镜传输层
        
        参数:
        host: 计算盒服务器的主机地址，默认127.0.0.1表示本机
        port: 计算盒服务器的端口号，默认9999
        """
        # 计算盒服务器地址
        self.host = host
        
        # 计算盒服务器端口
        self.port = port
        
        # 客户端socket对象，初始为None
        # 这个socket用于与计算盒服务器通信
        self.client_socket = None

    def connect(self):
        """
        与计算盒建立连接
        
        尝试连接到计算盒服务器
        如果连接成功，打印成功消息
        如果连接失败，打印错误信息
        
        返回:
        无返回值，但会设置self.client_socket
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
            print(f"[Transport] 眼镜已与计算盒 {self.host} 握手成功")
            
        except Exception as e:
            # 捕获连接过程中的所有异常
            print(f"[Transport] 无法连接到计算盒: {e}")
            
            # 注意：这里没有重试逻辑
            # 在实际应用中，可能需要添加自动重连机制

    def receive_data(self):
        """
        核心函数：从计算盒接收实时数据包
        
        返回:
        str: 解码后的JSON字符串，包含渲染用的位姿和AI识别结果
             如果接收失败，返回None
        
        这个函数实现了完整的数据接收流程：
        1. 接收4字节的数据长度头部
        2. 根据长度接收完整的JSON数据
        3. 解码为UTF-8字符串
        4. 返回JSON字符串
        
        数据包格式：[4字节JSON长度（大端）] + [JSON数据]
        这种设计解决了TCP粘包问题
        """
        try:
            # 1. 读取包头（接收4字节的JSON长度）
            # 假设数据包格式：4字节长度 + JSON数据
            # 使用struct.unpack解包4字节的无符号整数
            # '>I' 表示大端字节序的无符号整数
            header = self.client_socket.recv(4)
            
            # 如果接收到的数据为空，说明连接已断开
            if not header: 
                return None
                
            # 将4字节的长度数据转换为整数
            data_len = struct.unpack('>I', header)[0]
            
            # 2. 读取完整的JSON数据包
            # 初始化空字节串
            data = b''
            
            # 循环接收，直到收到完整的data_len字节数据
            # 这个循环解决了TCP数据分片问题
            while len(data) < data_len:
                # 计算还需要接收的字节数
                remaining = data_len - len(data)
                
                # 接收数据
                packet = self.client_socket.recv(remaining)
                
                # 如果接收到的数据为空，说明连接断开
                if not packet: 
                    break
                    
                # 将接收到的数据添加到data
                data += packet
                
            # 注意：这里没有检查是否收到了完整的数据
            # 如果循环因为break退出，data长度可能小于data_len
            
            # 将字节数据解码为UTF-8字符串
            return data.decode('utf-8')
            
        except Exception as e:
            # 捕获接收过程中的所有异常
            print(f"[Transport] 数据接收异常: {e}")
            return None