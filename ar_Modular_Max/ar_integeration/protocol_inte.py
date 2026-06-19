# 导入必要的Python库
import struct  # 用于处理二进制数据的打包和解包

class ProtocolHelper:
    """
    协议助手类，定义了自定义的二进制通信协议
    
    协议格式：[ 4字节类型ID ] + [ 4字节数据长度 ] + [ 真实数据 ]
    总共 8 字节的定长包头，让接收端永远不会迷路
    
    这个协议的设计目的是解决TCP流式传输的"粘包"问题：
    - TCP是字节流协议，不保证数据包边界
    - 这个协议添加了包头，让接收方知道每个数据包的开始和结束
    """
    
    # 定义协议头部的格式字符串
    # 格式字符串说明：
    #   '!' 表示网络字节序（大端字节序，big-endian）
    #   'I' 表示无符号整数（4字节）
    #   "!II" 表示两个4字节的无符号整数，共8字节
    # 第一个I是数据类型，第二个I是数据长度
    HEADER_FORMAT = "!II"
    
    # 包头的大小：两个4字节整数 = 8字节
    HEADER_SIZE = 8

    @staticmethod
    def pack(data_type, data):
        """
        将数据打包成带头的二进制流
        
        参数:
        data_type: 数据类型标识（整数），用于区分不同的消息类型
        data: 要发送的实际数据（字节类型）
        
        返回:
        打包好的二进制数据，格式为：[包头(8字节) + 数据]
        """
        # 计算数据的长度
        length = len(data)
        
        # 使用struct.pack打包包头
        # 参数说明：
        #   HEADER_FORMAT: 格式字符串"!II"
        #   data_type: 第一个整数，数据类型
        #   length: 第二个整数，数据长度
        # 返回8字节的二进制数据
        header = struct.pack(ProtocolHelper.HEADER_FORMAT, data_type, length)
        
        # 将包头和数据拼接起来
        return header + data

    @staticmethod
    def read_packet(sock):
        """
        从 Socket 中精准读取一个完整包
        
        参数:
        sock: socket对象，用于接收数据
        
        返回:
        一个元组 (data_type, payload):
        - data_type: 数据类型标识（整数）
        - payload: 实际数据（字节类型），如果读取失败则为None
        
        这个方法能解决TCP"粘包"问题，即使网络把数据切碎了，
        它也能等到凑齐长度再返回
        """
        # 1. 先读 8 字节包头
        # 调用_recv_all方法确保读取完整的8字节包头
        header_data = ProtocolHelper._recv_all(sock, ProtocolHelper.HEADER_SIZE)
        
        # 如果header_data是None，说明连接已断开
        if not header_data:
            return None, None  # 返回(None, None)表示失败
            
        # 使用struct.unpack解包头部数据
        # 参数说明：
        #   HEADER_FORMAT: 格式字符串"!II"
        #   header_data: 8字节的包头数据
        # 返回一个元组 (data_type, length)
        data_type, length = struct.unpack(ProtocolHelper.HEADER_FORMAT, header_data)
        
        # 2. 根据长度读取真实数据
        # 读取指定长度的数据
        payload = ProtocolHelper._recv_all(sock, length)
        
        # 返回数据类型和实际数据
        return data_type, payload

    @staticmethod
    def _recv_all(sock, count):
        """
        底层保障：循环读取直到凑齐 count 字节
        
        参数:
        sock: socket对象
        count: 需要读取的字节数
        
        返回:
        读取到的字节数据，如果连接断开则返回None
        
        这个方法确保即使TCP将数据分割成多个小包，
        也能完整读取指定数量的字节
        """
        # 创建一个空的字节串用于累积数据
        buf = b''
        
        # 循环读取，直到读取到count个字节
        while len(buf) < count:
            # 计算还需要读取的字节数
            # count - len(buf): 剩余的字节数
            # 注意：socket.recv的参数是"最多读取多少字节"，不保证读取这么多
            packet = sock.recv(count - len(buf))
            
            # 如果接收到的数据为空，说明连接已断开
            if not packet:
                return None
                
            # 将接收到的数据添加到缓冲区
            buf += packet
            
        # 返回完整的数据
        return buf