# 导入必要的Python库
import socket      # 提供网络通信功能，用于建立TCP连接
import cv2         # OpenCV库，用于图像处理、神经网络推理
import numpy as np # NumPy库，用于高效的数组操作
import json        # JSON数据处理库，用于处理AI响应
import time        # 时间相关功能，用于性能计时

# 定义云AI节点类
class CloudComputerAINode:
    # 初始化方法，设置AI节点的基本配置
    def __init__(self, port=9999):
        # 监听端口，计算盒会连接这个端口发送图片
        self.port = port
        # 运行状态标志，控制服务器主循环
        self.running = True
        
        # MobileNet-SSD 能够识别的 21 种标准现实世界物体标签
        # 这是预训练模型的类别标签，对应COCO数据集的20个类别+1个背景类
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
                        
        # 核心能力 1：在开机时本地硬加载真实神经网络，不联网，不训练
        print("[INFO] 🧠 正在加载本地 MobileNet-SSD 神经网络内核...")
        try:
            # 加载MobileNet-SSD模型
            # deploy.prototxt: 模型结构描述文件
            # mobilenet_iter_73000.caffemodel: 预训练模型权重文件
            self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
            print("[SUCCESS] 🟢 神经网络内核加载成功！100% 本地算力准备就绪。")
        except cv2.error:  # 如果加载模型失败
            print("[ERROR] ❌ 找不到模型文件！请确保 deploy.prototxt 和 mobilenet_iter_73000.caffemodel 在当前目录下。")
            exit(1)  # 退出程序

    def run_inference(self, cv_img):
        """核心部门：将图片塞入神经网络，压榨算力提取空间'语义'"""
        # 记录推理开始时间
        start_time = time.time()
        
        # 1. 图像预处理：将 OpenCV 的 BGR 图片缩放到 300x300，并减去均值，转化为神经网络能懂的 Blob 矩阵
        # cv2.dnn.blobFromImage参数说明：
        #   cv_img: 输入图像
        #   0.007843: 缩放因子，将像素值归一化到[0,1]范围
        #   (300, 300): 模型要求的输入尺寸
        #   127.5: 均值，从每个像素值中减去127.5
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_img, (300, 300)), 0.007843, (300, 300), 127.5)
        # 将预处理后的blob数据设置为网络的输入
        self.net.setInput(blob)
        
        # 2. 前向传播（Forward Pass）：真正的数学矩阵疯狂解算
        # 执行神经网络推理，返回检测结果
        detections = self.net.forward()
        
        # 初始化检测到的物体列表
        detected_objects = []
        # detections.shape[2] 是模型预测出来的可能物体的数量（默认最多100个）
        for i in range(0, detections.shape[2]):
            # 提取第i个检测结果的置信度（概率）
            # detections数组的形状是[1, 1, N, 7]，其中N是检测到的物体数量
            # 每个检测结果的格式是：[batch_id, class_id, confidence, x_min, y_min, x_max, y_max]
            confidence = detections[0, 0, i, 2]  # 提取该物体的置信度（概率）
            
            # 过滤掉概率低于 60% 的杂音目标
            if confidence > 0.6:  # 置信度阈值设为0.6
                # 获取类别ID
                class_id = int(detections[0, 0, i, 1])
                # 根据类别ID获取类别名称
                label_name = self.CLASSES[class_id]
                # 如果类别不在检测到的物体列表中，且不是背景类，则添加到列表中
                if label_name not in detected_objects and label_name != "background":
                    detected_objects.append(label_name)

        # 计算推理耗时（毫秒）
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 3. 场景宏观语义推断（根据检测到的物体，断定用户当前处于什么空间环境）
        # 基于检测到的物体类型判断场景类型
        scene_type = "Standard Room"  # 默认场景类型
        
        # 如果检测到电视或椅子，可能是电脑桌/工作站
        if "tvmonitor" in detected_objects or "chair" in detected_objects:
            scene_type = "Computer Desk / Workstation"
        # 如果检测到猫或狗，可能是客厅
        elif "cat" in detected_objects or "dog" in detected_objects:
            scene_type = "Living Room (Pet Area)"
        # 如果只检测到人，可能是用户工作空间
        elif "person" in detected_objects and len(detected_objects) == 1:
            scene_type = "User Workspace"

        # 返回场景类型、检测到的物体列表和推理延迟
        return scene_type, detected_objects, latency_ms

    def start_server(self):
        """网络服务端：开辟 9999 端口，专心当计算盒的'语义外包工'"""
        # 创建服务器socket
        server_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许端口快速重用
        server_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # 绑定到所有网络接口（0.0.0.0）和指定端口
            server_s.bind(("0.0.0.0", self.port))
            # 开始监听，最多允许5个连接排队
            server_s.listen(5)
            # 打印服务器启动信息
            print("\n=======================================================")
            print(f"  FranceHonor AI Node ({self.__class__.__name__}) 启动")
            print(f"  监听端口: {self.port} | 职责: 场景理解与物体识别 (低频高延迟)")
            print("=======================================================\n")
        except socket.error as e:  # 绑定失败
            print(f"[ERROR] AI端口绑定失败: {e}")
            return  # 退出方法

        # 服务器主循环
        while self.running:
            print("[INFO] ⏳ 正在大门口挂起，等待计算盒发送关键帧...")
            # 等待客户端连接
            conn, addr = server_s.accept()
            print(f"[INFO] 📡 计算盒已接入 AI 算力中心。接入源: {addr}")
            
            # 为每个客户端连接创建处理循环
            while self.running:
                try:
                    # 1. 读取计算盒转发过来的 4 字节图片长度包头
                    # 接收4字节的包头，包含图片长度信息
                    header = conn.recv(4)
                    # 如果接收到的数据为空或长度不足4字节，说明连接断开
                    if not header or len(header) < 4:
                        raise socket.error("计算盒断开链路")
                        
                    # 将4字节转换为整数，得到图片长度
                    img_len = int.from_bytes(header, byteorder='big')
                    
                    # 2. 循环接收完整的 JPEG 二进制字节流
                    # 初始化图片字节数据
                    img_bytes = b""
                    # 循环接收，直到接收完指定长度的图片数据
                    while len(img_bytes) < img_len:
                        # 每次最多接收4096字节，或者剩余需要的字节数
                        packet = conn.recv(min(img_len - len(img_bytes), 4096))
                        # 如果接收到的数据为空，说明连接断开
                        if not packet:
                            raise socket.error("图片流接收中断")
                        # 将接收到的数据添加到img_bytes
                        img_bytes += packet
                        
                    # 核心能力 5：打印合规的通信与运行日志
                    print(f"[RECV] 📥 接收到计算盒转发的关键帧。大小: {img_len/1024:.1f} KB")

                    # 3. 将二进制 JPEG 字节还原为物理内存里的 OpenCV 图片矩阵
                    # 将字节数据转换为NumPy数组
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    # 将JPEG字节解码为OpenCV图像
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # 如果解码失败，跳过本次推理
                    if cv_img is None:
                        print("[WARN] 图片解码失败，跳过本次推理。")
                        continue  # 继续下一次循环

                    # 4. 驱动 AI 核心跑真模型推理
                    # 运行神经网络推理
                    scene, objects, latency = self.run_inference(cv_img)
                    print(f"[INFERENCE] 🧠 神经网络计算完成。耗时: {latency}ms | 识别出: {objects}")

                    # 5. 按照 GPT 指示，打包成极简的 JSON 语义包返回
                    # 创建响应消息字典
                    response_msg = {
                        "scene": scene,          # 场景类型
                        "objects": objects,      # 检测到的物体列表
                        "latency_ms": latency    # AI推理延迟
                    }
                    
                    # 将字典转换为JSON字符串，再编码为字节
                    json_bytes = json.dumps(response_msg).encode('utf-8')
                    # 发送响应回计算盒
                    conn.sendall(json_bytes)
                    print(f"[SEND] ↩️ 语义反馈包已甩回给计算盒。\n")

                except socket.error:  # socket错误
                    print("[WARN] 🚨 计算盒与 AI 节点的连接断开。重新回到大门口等待恢复...")
                    break  # 退出客户端处理循环
            # 关闭与计算盒的连接
            conn.close()

# 程序入口点
if __name__ == "__main__":
    # 创建CloudComputerAINode实例，监听9999端口
    ai_node = CloudComputerAINode(port=9999)
    # 启动服务器
    ai_node.start_server()