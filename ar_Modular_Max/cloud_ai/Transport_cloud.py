# 导入必要的Python库
import socket      # 提供网络通信功能
import threading   # 多线程支持库

class CloudTransport:
    """
    云端传输层类，负责管理网络连接和通信
    
    这个类封装了TCP服务器功能，用于：
    1. 接收来自计算盒的连接
    2. 接收图片数据
    3. 将数据传递给AI处理
    4. 返回处理结果给计算盒
    """
    
    def __init__(self, host='0.0.0.0', port=8888):
        """
        初始化传输层
        
        参数:
        host: 监听的主机地址，默认0.0.0.0表示监听所有网络接口
        port: 监听的端口号，默认8888
        """
        # 服务器监听地址
        self.host = host
        
        # 服务器监听端口
        self.port = port
        
        # 服务器socket对象，初始为None
        self.server_socket = None

    def start_server(self, on_receive_callback):
        """
        启动监听服务
        
        参数:
        on_receive_callback: 当收到数据时，回调给runtime处理的函数
        
        这个函数会：
        1. 创建TCP服务器
        2. 监听客户端连接
        3. 为每个客户端连接创建独立线程处理
        4. 持续运行直到程序终止
        """
        # 创建TCP socket对象
        # AF_INET: 使用IPv4地址族
        # SOCK_STREAM: 使用TCP协议
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 绑定socket到指定地址和端口
        # 0.0.0.0: 监听所有可用的网络接口
        # 这意味着服务器会接受来自所有IP地址的连接
        self.server_socket.bind((self.host, self.port))
        
        # 开始监听客户端连接
        # 参数5: 最大连接队列长度
        # 当有多个客户端同时连接时，操作系统会将超过处理能力的连接放入队列
        # 队列最多可以存放5个等待处理的连接
        self.server_socket.listen(5)
        
        # 打印服务器启动信息
        print(f"[Transport] 云端服务在 {self.port} 端口等待连接...")

        # 服务器主循环，持续运行直到程序终止
        while True:
            # 等待客户端连接
            # accept()是阻塞调用，会一直等待直到有客户端连接
            # 返回值：
            #   conn: 与客户端通信的新socket对象
            #   addr: 客户端的地址信息（IP地址, 端口号）
            conn, addr = self.server_socket.accept()
            
            # 打印客户端连接信息
            print(f"[Transport] 收到来自 {addr} 的连接")
            
            # 为每个连接启动一个独立处理线程
            # threading.Thread: 创建新线程
            # target=self._handle_connection: 线程要执行的函数
            # args=(conn, on_receive_callback): 传递给函数的参数
            #   conn: 客户端连接socket
            #   on_receive_callback: 回调函数
            # daemon=True: 设置为守护线程
            #   守护线程在主程序退出时会自动结束
            threading.Thread(target=self._handle_connection, args=(conn, on_receive_callback), daemon=True).start()

    def _handle_connection(self, conn, callback):
        """
        处理单个客户端连接的私有方法
        
        参数:
        conn: 客户端连接socket
        callback: 数据接收回调函数
        
        这个函数在一个独立的线程中运行，处理与单个客户端的完整通信流程
        流程：
        1. 接收数据长度
        2. 接收实际数据
        3. 调用回调函数处理数据
        4. 返回处理结果
        5. 重复直到连接断开
        """
        # 为每个客户端连接创建独立的处理循环
        while True:
            try:
                # 接收数据长度 (定长 4 字节)
                # 假设客户端发送数据的格式：[4字节长度][实际数据]
                # 这种设计解决了TCP粘包问题
                len_bytes = conn.recv(4)
                
                # 如果接收到的数据为空，说明客户端已断开连接
                if not len_bytes: 
                    break  # 退出循环，结束线程
                
                # 将4字节的长度数据转换为整数
                # int.from_bytes: 将字节转换为整数
                # byteorder='big': 使用大端字节序（网络字节序）
                data_len = int.from_bytes(len_bytes, byteorder='big')
                
                # 接收实际图片数据
                # 初始化空字节串
                data = b''
                
                # 循环接收，直到收到完整的data_len字节数据
                # 这个循环解决了TCP数据分片问题
                while len(data) < data_len:
                    # 计算还需要接收的字节数
                    remaining = data_len - len(data)
                    
                    # 接收数据
                    packet = conn.recv(remaining)
                    
                    # 如果接收到的数据为空，说明连接断开
                    if not packet: 
                        break
                    
                    # 将接收到的数据添加到data
                    data += packet
                
                # 检查是否收到了完整的数据
                if len(data) < data_len:
                    # 数据不完整，连接可能已断开
                    print(f"[Transport] 数据接收不完整，期望{data_len}字节，实际收到{len(data)}字节")
                    break
                
                # 回调：把数据扔给 runtime 处理
                # callback是外部传入的处理函数
                # 它将接收到的数据作为参数，返回处理结果
                result = callback(data)
                
                # 回传结果
                if result:
                    # 将处理结果发送回客户端
                    # 注意：这里假设result是字符串，需要编码为字节
                    conn.sendall(result.encode('utf-8'))
                    
            except Exception as e:
                # 捕获并处理所有异常
                print(f"[Transport] 通信错误: {e}")
                break  # 退出循环，结束线程
                
        # 关闭客户端连接
        conn.close()