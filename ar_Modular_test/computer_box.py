# 导入必要的Python库
import socket      # 提供网络通信功能，用于建立TCP连接
import threading   # 多线程支持库，允许同时处理多个客户端连接
import time        # 时间相关功能，用于延时、超时控制
import json        # JSON数据处理库，用于处理AI节点的JSON响应

# 定义计算盒类
class ComputeBox:
    # 初始化方法，设置计算盒的基础配置
    def __init__(self, listen_port=8888, ai_ip="127.0.0.1", ai_port=9999):
        # 监听端口，眼镜端会连接这个端口
        self.listen_port = listen_port
        # AI服务器的IP地址，默认是本机
        self.ai_ip = ai_ip
        # AI服务器的端口号，默认是9999
        self.ai_port = ai_port
        # 运行状态标志，控制服务器主循环
        self.running = True
        
        # 内部状态：AI 节点的连接状态
        # ai_socket: 连接到AI服务器的socket对象
        self.ai_socket = None
        # ai_online: AI节点是否在线
        self.ai_online = False

    def connect_to_ai_node(self):
        """尝试连接 AI 语义节点 (上行链路)"""
        try:
            # 创建一个TCP socket对象
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置2秒连接超时，防止长时间等待
            s.settimeout(2.0)
            # 尝试连接到AI服务器
            s.connect((self.ai_ip, self.ai_port))
            # 连接成功，恢复阻塞模式
            s.settimeout(None)
            # 保存socket对象
            self.ai_socket = s
            # 设置AI在线状态为True
            self.ai_online = True
            # 打印成功连接信息
            print("[INFO] 🔗 成功上连到 AI 节点，语义分析引擎就绪。")
        except socket.error:  # 连接失败
            # 重置socket为None
            self.ai_socket = None
            # 设置AI离线状态
            self.ai_online = False
            # 打印警告信息
            print("[WARN] ❌ 无法连接到 AI 节点。系统将进入[纯CV空间生存模式]，AI 语义将采用本地降级保活。")

    def forward_to_ai_with_timeout(self, header, img_bytes):
        """核心能力 3：简单超时机制与数据路由"""
        # 如果AI不在线或socket不存在，尝试重新连接
        if not self.ai_online or not self.ai_socket:
            # 尝试动态重连 AI
            self.connect_to_ai_node()
            # 如果重连后AI仍然不在线，生成降级响应
            if not self.ai_online:
                return self.generate_fallback_response("AI Node Offline")

        try:
            # 1. 转发图片给 AI 节点
            # 发送包头和图片数据到AI服务器
            self.ai_socket.sendall(header + img_bytes)
            
            # 2. 设置接收超时机制（死守 3 秒界线，超过就认为云端网络炸了）
            # 设置3秒接收超时
            self.ai_socket.settimeout(3.0)
            # 尝试接收AI服务器的响应，最多1024字节
            ai_data = self.ai_socket.recv(1024)
            # 恢复为阻塞模式
            self.ai_socket.settimeout(None)
            
            # 如果接收到的数据为空，说明连接已断开
            if not ai_data:
                raise socket.error("AI 节点主动断开")
                
            # 正常收到 AI 节点的真实神经网络结果
            return ai_data
            
        except (socket.error, socket.timeout):  # socket错误或超时
            print("[WARN] 🚨 AI 节点响应超时或断连！启动本地优雅降级算法...")
            # 设置AI离线状态
            self.ai_online = False
            # 如果socket存在，关闭它
            if self.ai_socket:
                self.ai_socket.close()
            # 生成降级响应
            return self.generate_fallback_response("AI Jitter/Timeout")

    def generate_fallback_response(self, reason):
        """模拟计算盒本地建图（Mapping）与自救，伪造极简保活数据"""
        # 创建一个降级响应的字典
        fallback = {
            "scene": f"Desk (Local CV Backup - {reason})",  # 场景信息，包含原因
            "objects": ["Local Tracking Active"],           # 物体列表
            "latency_ms": 0                                 # 延迟为0（本地处理）
        }
        # 将字典转换为JSON字符串，再编码为字节
        return json.dumps(fallback).encode('utf-8')

    def _handle_glasses_client(self, glass_conn):
        """专门伺候连进来的眼镜端的业务线程"""
        print("[INFO] 👤 眼镜端已进入计算盒工作区，启动数据流调度机制。")
        
        # 持续为这个眼镜客户端服务
        while self.running:
            try:
                # 1. 读取 4 字节的自定义包头（得知后面图片数据的长度）
                # 接收4字节的包头，包含图片长度信息
                header = glass_conn.recv(4)
                # 如果接收到的数据为空或长度不足4字节，说明连接断开
                if not header or len(header) < 4:
                    raise socket.error("眼镜端断开")
                    
                # 将4字节转换为整数，得到图片长度
                # byteorder='big' 表示大端序（高位在前）
                img_len = int.from_bytes(header, byteorder='big')
                
                # 2. 循环读取，直到完整收完这张 JPEG 图片的二进制内容
                # 初始化图片字节数据
                img_bytes = b""
                # 循环接收，直到接收完指定长度的图片数据
                while len(img_bytes) < img_len:
                    # 每次最多接收4096字节，或者剩余需要的字节数
                    packet = glass_conn.recv(min(img_len - len(img_bytes), 4096))
                    # 如果接收到的数据为空，说明连接断开
                    if not packet:
                        raise socket.error("图片流读取中断")
                    # 将接收到的数据添加到img_bytes
                    img_bytes += packet

                # 核心能力 5：打印极简、合规的通信日志
                # 打印接收到的关键帧信息
                print(f"[RECV] 从眼镜端收到 Keyframe. 大小: {img_len/1024:.1f} KB. 优先级: 低(低频)")
                
                # ---------------------------------------------------------
                # 【模拟计算盒本地高频任务：Mapping 与 空间重定位】
                # 实际 XR 系统在这里将图片塞入局部地图进行特征点匹配、剔除回环
                # ---------------------------------------------------------
                # 模拟本地 Mapping 优化耗时
                time.sleep(0.005)  # 休眠5毫秒，模拟本地处理时间
                print("[MAPPING] 本地局部建图与优化完成 (耗时: 5ms)")

                # 3. 路由转发：将数据扔给 AI 节点，并等待其结果（包含超时自救）
                # 将图片数据转发给AI节点，获取AI响应
                ai_response_bytes = self.forward_to_ai_with_timeout(header, img_bytes)
                
                # 4. 把最终的语义 JSON 数据，顺着网络线原路甩回给眼镜
                # 将AI响应发送回眼镜端
                glass_conn.sendall(ai_response_bytes)
                print(f"[FORWARD] ↩️ 已成功向眼镜端反馈最新的语义空间状态。\n")

            except socket.error:  # socket错误
                print("[WARN] 🚨 眼镜端与计算盒连接断开。释放当前处理线程。")
                break  # 退出循环，结束这个客户端处理线程
                
        # 关闭与眼镜端的连接
        glass_conn.close()

    def start_server(self):
        """核心能力 1：基础 TCP Socket 多线程服务端监听"""
        # 创建服务器socket
        server_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许端口快速重用，防止调测时产生 Address already in use 报错
        # SO_REUSEADDR 选项允许重用本地地址
        server_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # 绑定到所有网络接口（0.0.0.0）和指定端口
            server_s.bind(("0.0.0.0", self.listen_port))
            # 开始监听，最多允许5个连接排队
            server_s.listen(5)
            # 打印服务器启动信息
            print("\n=======================================================")
            print(f"  FranceHonor 计算盒中间件启动。正在监听端口: {self.listen_port}")
            print("=======================================================\n")
        except socket.error as e:  # 绑定失败
            print(f"[ERROR] 端口绑定失败: {e}")
            return  # 退出方法

        # 启动时先尝试连接 AI 节点
        self.connect_to_ai_node()

        # 服务器主循环
        while self.running:
            try:
                # 等待客户端连接
                # 这是阻塞调用，直到有客户端连接
                glass_conn, addr = server_s.accept()
                # 打印客户端连接信息
                print(f"\n[INFO] 📡 侦测到物理眼镜节点接入！来自: {addr}")
                
                # 每来一副眼镜，开辟一个独立线程去服务，主循环立刻回到 accept 挂起
                # 创建线程处理这个客户端连接
                t = threading.Thread(target=self._handle_glasses_client, args=(glass_conn,), daemon=True)
                t.start()  # 启动线程
            except KeyboardInterrupt:  # 捕获Ctrl+C中断
                self.running = False
                break  # 退出循环
                
        # 关闭服务器socket
        server_s.close()
        # 打印服务器关闭信息
        print("[INFO] 计算盒系统已安全关闭。")

# 程序入口点
if __name__ == "__main__":
    # 计算盒启动：本地监听 8888 等眼镜；同时去连 9999 端口的 AI 节点
    # 创建ComputeBox实例
    box = ComputeBox(listen_port=8888, ai_ip="127.0.0.1", ai_port=9999)
    # 启动服务器
    box.start_server()