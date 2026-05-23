# 导入必要的Python库
import socket      # 提供网络通信功能，用于与计算盒建立TCP连接
import cv2         # OpenCV库，用于图像采集、处理和显示
import json        # JSON数据处理库，用于解析和生成JSON格式的网络消息
import threading   # 多线程支持库，允许程序同时执行多个任务
import time        # 时间相关功能，用于延时、时间戳等

# 定义AR眼镜显示类
class DisplayGlass:
    # 初始化方法，设置眼镜端的基础配置
    def __init__(self, box_ip="127.0.0.1", box_port=8888):
        # 计算盒的IP地址，默认是本机（127.0.0.1）
        self.box_ip = box_ip
        # 计算盒的端口号，默认是8888
        self.box_port = box_port
        # 运行状态标志，控制程序主循环
        self.running = True
        
        # 共享系统状态字典，存储从计算盒接收的AI分析结果
        # 这个字典被网络接收线程更新，被主渲染线程读取
        self.system_status = {
            "scene": "Initializing...",  # 场景识别结果
            "objects": [],               # 检测到的物体列表
            "ai_latency": "N/A"          # AI处理延迟
        }
        
        # 尝试建立与计算盒的初始连接
        # 调用connect_to_compute_box方法创建socket连接
        self.box_socket = self.connect_to_compute_box()

    def connect_to_compute_box(self):
        """
        核心能力2：简单断线恢复机制
        连接到计算盒服务器，如果连接失败会每隔3秒自动重试
        返回: 已连接的socket对象，如果程序停止运行则返回None
        """
        while self.running:  # 只要程序还在运行就持续尝试连接
            try:
                # 创建一个TCP socket对象
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 设置连接超时为3秒，防止长时间等待
                s.settimeout(3.0)
                # 打印连接尝试信息
                print(f"[INFO] 正在尝试连接计算盒 [{self.box_ip}:{self.box_port}]...")
                # 尝试连接到指定的IP和端口
                s.connect((self.box_ip, self.box_port))
                # 连接成功后，将socket恢复为阻塞模式，这样recv会一直等待数据
                s.settimeout(None)
                # 打印连接成功信息
                print("[SUCCESS] 已成功连接到计算盒！系统启动。")
                # 返回连接成功的socket对象
                return s
            except (socket.error, socket.timeout):  # 连接失败或超时
                # 打印警告信息
                print("[WARN] 无法连接到计算盒。3秒后将自动发起重连...")
                # 等待3秒后重试
                time.sleep(3.0)
        # 如果程序停止运行(running=False)，返回None
        return None

    def start_network_receiver(self):
        """
        后台网络接收线程：专门接收计算盒返回的低频AI语义结果
        这个方法启动一个独立的线程，持续监听来自计算盒的消息
        """
        def receive_loop():
            # 接收循环，持续运行直到程序停止
            while self.running:
                # 如果socket不存在，等待1秒后继续检查
                if not self.box_socket:
                    time.sleep(1.0)
                    continue
                try:
                    # 从socket接收最多1024字节的数据
                    # 这是阻塞调用，会一直等待直到有数据到达
                    data = self.box_socket.recv(1024)
                    # 如果接收到的数据为空，说明连接已断开
                    if not data:
                        raise socket.error("计算盒主动断开连接")
                        
                    # 解析接收到的JSON格式数据
                    # 计算盒发送的消息应该是JSON字符串
                    msg = json.loads(data.decode('utf-8'))
                    
                    # 核心能力5：打印规范的通信日志
                    # 📥 是一个emoji符号，表示接收
                    print(f"[RECV] 📥 收到 AI 增强语义更新: {msg}")
                    
                    # 更新全局状态，供主线程渲染使用
                    # 使用get方法安全获取字典值，如果键不存在则使用默认值
                    self.system_status["scene"] = msg.get("scene", "Unknown")
                    self.system_status["objects"] = msg.get("objects", [])
                    self.system_status["ai_latency"] = f"{msg.get('latency_ms', 'N/A')}ms"
                    
                except socket.error:  # socket错误，通常是连接断开
                    print("[WARN] 🚨 与计算盒的通信意外中断！引发自救，挂起网络模块...")
                    # 将socket设为None，标记为已断开
                    self.box_socket = None
                    # 触发自动重连机制，尝试重新连接
                    self.box_socket = self.connect_to_compute_box()

        # 启动后台线程，绝不占用前端摄像头抓取的主线程
        # daemon=True表示这是守护线程，主程序退出时自动结束
        t = threading.Thread(target=receive_loop, daemon=True)
        t.start()  # 启动线程

    def send_keyframe_async(self, frame):
        """
        异步发送关键帧：将高延迟的图片发送任务外包出去，确保主线程秒回
        参数:
            frame: 要发送的图像帧（numpy数组）
        """
        def send_worker(f):
            # 如果socket不存在，直接返回
            if not self.box_socket:
                return
            try:
                # 核心功耗调优：绝不传原始大图，本地用JPEG压缩（质量设为70%），数据量瞬间暴跌90%
                # 将图像编码为JPEG格式，质量70%
                success, encoded_image = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not success:  # 如果编码失败，直接返回
                    return
                    
                # 将编码后的图像转换为字节数据
                img_bytes = encoded_image.tobytes()
                # 制作一个极简的自定义包头：4字节的图片长度 + 原始二进制数据
                # 抛弃复杂的互联网框架，用最纯粹的系统底标
                # 将图片长度转换为4字节的大端序（big-endian）字节表示
                header = len(img_bytes).to_bytes(4, byteorder='big')
                
                # 打印发送信息，显示图片大小
                print(f"[SEND] ⏱️ 2s周期触发: 正在异步发射关键帧 Keyframe (大小: {len(img_bytes)/1024:.1f} KB)...")
                # 发送包头和图片数据
                self.box_socket.sendall(header + img_bytes)
                
            except socket.error:  # 发送失败
                print("[WARN] 关键帧发送失败，网络管道阻塞。")

        # 每次发送都新开一个"临时快递员线程"，发完自动销毁
        # 创建并启动一个线程来执行发送任务
        threading.Thread(target=send_worker, args=(frame,), daemon=True).start()

    def run_runtime_loop(self):
        """
        眼镜端主运行循环：绝对的'速度狂魔'，死守低延迟空间定位与渲染
        这个方法包含主循环，处理摄像头输入、本地渲染和网络通信
        """
        # 打开笔记本本地摄像头，0表示默认摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():  # 检查摄像头是否成功打开
            print("[ERROR] 找不到摄像头，请检查硬件权限。")
            return  # 如果打不开摄像头，直接返回

        # 打印启动信息
        print("\n=======================================================")
        print("  FranceHonor 眼镜端 Runtime 启动 (主控方向: Spatial Tracking)")
        print("=======================================================\n")
        
        # 帧计数器，用于控制关键帧发送频率
        frame_count = 0
        # 启动网络接收线程，开始监听计算盒的消息
        self.start_network_receiver()

        # 主循环，持续运行直到用户退出
        while self.running:
            # 从摄像头读取一帧图像
            ret, frame = cap.read()
            if not ret:  # 如果读取失败，退出循环
                break

            # 帧计数器加1
            frame_count += 1

            # -------------------------------------------------------------
            # 【模拟 30fps 实时 Tracking 核心部门】
            # 真实 XR 系统在这里跑 ORB 特征提取、PnP 姿态解算。我们在这里画点模拟
            # -------------------------------------------------------------
            # 模拟：本地高频渲染一个定位方块（代表空间追踪定位永远在线，死守 5ms）
            # 获取图像的高度、宽度和通道数
            h, w, _ = frame.shape
            # 在图像中心画一个绿色矩形，表示跟踪目标
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            
            # -------------------------------------------------------------
            # 核心能力 4：数据优先级策略（实时 Tracking 优先，AI 语义超低频卸载）
            # 每隔 60 帧（大约 2 秒），低频向外发送一个关键帧图片用于 AI 语义分析
            # -------------------------------------------------------------
            if frame_count % 60 == 0:  # 每60帧发送一次关键帧
                # 异步发送关键帧，发送的是当前帧的副本
                self.send_keyframe_async(frame.copy())

            # -------------------------------------------------------------
            # 【本地近眼显示渲染部门 (Render)】
            # 将后台接收到的、哪怕延迟很高的 AI 语义，平滑地叠加在画面上
            # -------------------------------------------------------------
            # 从系统状态中获取信息，构建要显示的文本
            status_text = f"Scene: {self.system_status['scene']}"
            objects_text = f"Objects: {', '.join(self.system_status['objects'])}"
            ai_time_text = f"AI Latency: {self.system_status['ai_latency']}"

            # 在左上角渲染 AR 状态面板
            # 绘制场景信息文本
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # 绘制物体列表文本
            cv2.putText(frame, objects_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # 绘制AI延迟文本
            cv2.putText(frame, ai_time_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
            # 在左下角绘制跟踪状态
            cv2.putText(frame, "Tracking Status: LOCK (Local)", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 刷新本地屏幕（模拟近眼光机输出）
            # 在窗口中显示处理后的图像
            cv2.imshow("FranceHonor - Glasses Node View", frame)

            # 检查是否按下了ESC键（ASCII码27）
            if cv2.waitKey(1) & 0xFF == 27:
                # 如果按下ESC，设置运行标志为False，退出循环
                self.running = False
                break

        # 释放摄像头资源
        cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        # 如果socket存在，关闭socket连接
        if self.box_socket:
            self.box_socket.close()
        # 打印退出信息
        print("[INFO] 眼镜端 Runtime 已安全关闭。")

# 程序入口点
if __name__ == "__main__":
    # 本地局域网联调，默认连接本机的 8888 端口（计算盒的入口）
    # 创建DisplayGlass实例，连接到本地计算盒
    node = DisplayGlass(box_ip="127.0.0.1", box_port=8888)
    # 启动主运行循环
    node.run_runtime_loop()