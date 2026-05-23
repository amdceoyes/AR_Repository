"""
FranceHonor 实战级云端AI服务器
- 剔除所有 time.sleep() 和 random 随机数幻觉
- 注入真正的 OpenCV 视觉追踪内核，实现强算力端的“异构计算有效输出”
- 与交互端（Y键定格、手动画圈ROI）完美对接
"""

import socket
import json
import time
import base64
import threading
import struct
import traceback
import cv2
import numpy as np

# ==================== 配置类 ====================
class CloudConfig:
    SERVER_IP = "127.0.0.1"  # 单机联调使用本地环回地址
    SERVER_PORT = 9999
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    DEBUG = True

# ==================== 核心视觉算力内核 ====================
class RealAICore:
    """真正的 AI/CV 算力核心，拒绝幻觉"""
    def __init__(self):
        # 初始化 OpenCV 的内置跟踪器 (这里采用经典的 KCF 算法)
        # 如果你的 OpenCV 编译了 CUDA，这里可以无缝替换为 GPU 加速版本，实现真正的异构计算
        self.tracker = cv2.TrackerKCF_create()
        self.is_initialized = False

    def process_frame(self, cv_img: np.ndarray, roi: list) -> dict:
        """
        真正的图像处理逻辑
        :param cv_img: 解码后的真实 OpenCV 图像矩阵
        :param roi: 交互端传过来的 [x, y, w, h] 矩形框
        """
        # 如果前端传来了有效的 ROI，且追踪器还没初始化，则进行“硬初始化”
        if roi and not self.is_initialized:
            # roi 格式: [x, y, w, h]
            bbox = (roi[0], roi[1], roi[2], roi[3])
            self.tracker.init(cv_img, bbox)
            self.is_initialized = True
            print(f"[AI Core] 🎯 视觉追踪器初始化成功！目标区域: {bbox}")
            return {"status": "INITIALIZED", "bbox": roi, "confidence": 1.0}

        # 如果已经初始化，则在当前新帧上进行高强度追踪计算
        if self.is_initialized:
            success, bbox = self.tracker.update(cv_img)
            if success:
                # 追踪成功，提取真实的物理坐标
                x, y, w, h = [int(v) for v in bbox]
                print(f"[AI Core] 🚀 目标追踪中... 当前真实坐标: [{x}, {y}, {w}, {h}]")
                return {
                    "status": "SUCCESS",
                    "bbox": [x, y, w, h],
                    "confidence": 0.95
                }
            else:
                print("[AI Core] ⚠️ 目标丢失或超出视野！")
                return {"status": "TRACKING_LOST", "bbox": [0,0,0,0], "confidence": 0.0}

        return {"status": "WAITING_FOR_ROI", "bbox": [0,0,0,0], "confidence": 0.0}

# ==================== 网络服务端骨架 ====================
class CloudAIServer:
    def __init__(self):
        self.config = CloudConfig()
        self.ai_core = RealAICore()
        self.server_socket = None
        self.running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.config.SERVER_IP, self.config.SERVER_PORT))
        self.server_socket.listen(1)
        self.running = True
        print(f"==================================================")
        print(f"🚀 FranceHonor 云端AI服务器已启动！")
        print(f"📡 监听地址: {self.config.SERVER_IP}:{self.config.SERVER_PORT}")
        print(f"==================================================")

        try:
            while self.running:
                conn, addr = self.server_socket.accept()
                print(f"\n[网络层] 💻 收到来自计算盒/交互端的连接: {addr}")
                # 开启线程处理这个客户端
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        except KeyboardInterrupt:
            print("\n[系统] 收到中断信号，服务器正在关闭...")
        finally:
            self.cleanup()

    def _handle_client(self, conn: socket.socket):
        """处理计算盒的数据流交换"""
        while self.running:
            try:
                # 1. 读取元宝保留的标准 4 字节大端序包头
                header = self._recv_all(conn, 4)
                if not header: break
                
                msg_len = struct.unpack('!I', header)[0]
                if msg_len > self.config.MAX_REQUEST_SIZE:
                    print(f"[警告] 数据包超限 ({msg_len} bytes)，强制拦截！")
                    break

                # 2. 读取完整的 JSON 字符串消息体
                raw_data = self._recv_all(conn, msg_len)
                if not raw_data: break

                request_json = json.loads(raw_data.decode('utf-8'))
                
                # 3. 解析 Base64 图像并还原为 OpenCV 矩阵（真正的异构数据还原）
                img_b64 = request_json.get("image_data")
                roi = request_json.get("roi") # 获取画圈的坐标 [x, y, w, h]
                
                if img_b64:
                    img_bytes = base64.b64decode(img_b64)
                    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    # 4. 送入视觉内核进行真正的核心算力追踪
                    start_time = time.time()
                    ai_results = self.ai_core.process_frame(cv_img, roi)
                    cost_time = time.time() - start_time

                    # 5. 组装响应字典，把纯粹的真实追踪坐标返回去
                    response_data = {
                        "status": ai_results["status"],
                        "processing_time": f"{cost_time*1000:.2f}ms",
                        "tracked_bbox": ai_results["bbox"],
                        "confidence": ai_results["confidence"]
                    }
                else:
                    response_data = {"status": "ERROR", "message": "未收到有效的图像数据"}

                # 6. 将有效输出封包并回传给客户端
                response_bytes = json.dumps(response_data).encode('utf-8')
                header_bytes = struct.pack('!I', len(response_bytes))
                conn.sendall(header_bytes + response_bytes)

            except Exception as e:
                print(f"[错误] 处理数据时发生异常: {e}")
                traceback.print_exc()
                break

        print("[网络层] 🔌 客户端断开连接。")
        conn.close()
        self.ai_core.is_initialized = False # 断开后重置追踪器

    def _recv_all(self, conn: socket.socket, n: int) -> bytes:
        """保证读取到指定长度的字节流，防止断包"""
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk: return b""
            data += chunk
        return data

    def cleanup(self):
        if self.server_socket:
            self.server_socket.close()
        print("[系统] 资源清理完毕，服务器已安全离线。")

if __name__ == "__main__":
    server = CloudAIServer()
    server.start()