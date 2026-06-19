# 导入必要的Python库
import socket      # 提供网络通信功能
import threading   # 多线程支持库

# 定义网络助手类
class NetworkHelper:
    """
    网络助手类，提供服务器创建和数据发送的静态方法
    这个类被设计为工具类，不需要创建实例即可使用
    """
    
    # 使用@staticmethod装饰器表示这是一个静态方法
    # 静态方法属于类，但不访问类的实例属性
    @staticmethod
    def create_server(port, handler_func):
        """
        创建一个 TCP 服务器并自动开启多线程监听
        
        参数:
        port: 要监听的端口号（整数）
        handler_func: 接收到客户端连接后要执行的函数（业务逻辑处理函数）
                      这个函数应该接收一个参数：client_socket
        
        返回:
        server_s: 创建的服务器socket对象，如果创建失败则返回None
        """
        # 创建一个TCP socket对象
        # AF_INET: 使用IPv4地址族
        # SOCK_STREAM: 使用TCP协议（面向连接的流式socket）
        server_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 设置socket选项，允许端口快速重用
        # SOL_SOCKET: 表示在socket级别设置选项
        # SO_REUSEADDR: 允许重用本地地址
        # 1: 启用此选项（True）
        # 这个选项在调试时特别有用，避免"Address already in use"错误
        server_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # 绑定socket到指定端口
            # "0.0.0.0": 绑定到所有可用的网络接口
            # 这意味着服务器会监听所有网络接口（包括本地回环、局域网、公网）
            # port: 要监听的端口号
            server_s.bind(("0.0.0.0", port))
            
            # 开始监听客户端连接
            # 5: 最大连接队列长度
            # 当有多个客户端同时连接时，操作系统会将超过处理能力的连接放入队列
            # 这个数字指定队列的最大长度
            server_s.listen(5)
            
            # 打印服务器启动成功的信息
            print(f"[NETWORK] 服务器已启动，监听端口: {port}")
            
        except socket.error as e:  # 如果绑定或监听失败
            # 打印错误信息
            print(f"[NETWORK] 绑定失败: {e}")
            # 返回None表示创建服务器失败
            return None

        # 定义一个内部函数，用于处理客户端连接请求
        def accept_loop():
            """
            接收客户端的连接循环
            这个函数在单独的线程中运行，专门负责"接待"新客户端
            """
            while True:  # 无限循环，持续接收客户端连接
                # 等待客户端连接
                # accept()是阻塞调用，会一直等待直到有客户端连接
                # 返回值：
                #   client_sock: 与客户端通信的新socket对象
                #   addr: 客户端的地址信息（IP, 端口）
                client_sock, addr = server_s.accept()
                
                # 打印新客户端连接信息
                print(f"[NETWORK] 新设备连接: {addr}")
                
                # 为每个客户端连接创建一个新的线程来处理
                # threading.Thread: 创建新线程
                # target=handler_func: 线程要执行的函数
                # args=(client_sock,): 传递给handler_func的参数
                # daemon=True: 设置为守护线程
                #   守护线程在主程序退出时会自动结束
                #   这意味着如果主程序退出，所有客户端处理线程也会自动结束
                threading.Thread(target=handler_func, args=(client_sock,), daemon=True).start()
                # 注意：这里只创建了线程但没有调用start()，实际代码中应该调用start()

        # 创建一个线程来运行accept_loop函数
        # 这样做的好处是：主线程不会被阻塞，可以继续执行其他任务
        threading.Thread(target=accept_loop, daemon=True).start()
        # 返回创建的服务器socket对象
        return server_s

    @staticmethod
    def send_data(sock, data):
        """
        简单封装发送逻辑
        
        参数:
        sock: 要发送数据的socket对象
        data: 要发送的数据（字节类型）
        
        这个方法封装了sendall操作，添加了错误处理
        """
        try:
            # 发送数据
            # sendall()会确保所有数据都被发送，或者抛出异常
            # 与send()不同，send()可能只发送部分数据
            sock.sendall(data)
        except Exception as e:  # 捕获所有异常
            # 打印发送失败的信息
            print(f"[NETWORK] 发送失败: {e}")
            # 注意：这里只是打印错误，没有重新抛出异常
            # 调用者可能想知道发送是否成功，但这里没有返回值指示