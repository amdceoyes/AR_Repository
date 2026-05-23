"""
程序名：FranceHonor (法兰西荣耀) - 眼镜端客户端
融合版：保留硬核 TCP 字节流通信，注入用户画圈、IMU节能保活与低通滤波锚定灵魂
"""

import socket
import json
import time
import base64
import threading
import queue
import numpy as np
import cv2
import struct
from dataclasses import dataclass, asdict
from enum import Enum

# ==================== 数据类型与状态定义 ====================

class SystemStatus(Enum):
    INITIALIZING = "INITIALIZING"   # 初始化（只开摄像头，不追踪不渲染）
    FROZEN = "FROZEN"               # 画面定格，等待用户圈定 ROI
    TRACKING = "TRACKING"           # 正常追踪渲染
    LOST = "LOST"                   # 跟踪丢失（触发 IMU 盲推）
    RECOVERING = "RECOVERING"       # 重定位中

@dataclass
class VisualData:
    timestamp: float
    image_id: int
    status: str
    pose: list
    is_imu_active: bool            # 告诉计算盒，当前 IMU 是否有大变动
    base_anchor: list              # 用户圈定的空间基础锚点 [x, y, z]

# ==================== 核心配置与日志 ====================

class GlassesConfig:
    def __init__(self):
        self.COMPUTE_BOX_IP = "127.0.0.1"
        self.COMPUTE_BOX_PORT = 8888
        self.TARGET_FPS = 30.0
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.LOST_MAX_DELAY = 5    # 最大允许丢失帧数

class Logger:
    def info(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [INFO] {msg}")
    def warning(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [WARN] {msg}")
    def error(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [ERR ] {msg}")

# ==================== 网络通信模块 (保留元宝的硬核多线程实现) ====================

class ComputeBoxClient:
    def __init__(self, config: GlassesConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.socket = None
        self.connected = False
        self.send_queue = queue.Queue(maxsize=10)
        self.running = False

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(3.0)
            self.socket.connect((self.config.COMPUTE_BOX_IP, self.config.COMPUTE_BOX_PORT))
            self.connected = True
            self.logger.info("成功连接到本地计算盒服务端！")
            return True
        except Exception as e:
            self.logger.error(f"连接计算盒失败 (请先启动服务端): {e}")
            return False

    def start(self):
        if not self.connect(): return False
        self.running = True
        threading.Thread(target=self._send_loop, daemon=True).start()
        return True

    def send_data(self, visual_data: VisualData):
        if not self.connected: return False
        try:
            json_str = json.dumps(asdict(visual_data))
            message = json_str.encode('utf-8')
            header = struct.pack('!I', len(message)) # 4字节包头
            self.send_queue.put((header + message), block=False)
            return True
        except queue.Full:
            return False

    def _send_loop(self):
        while self.running and self.connected:
            try:
                data = self.send_queue.get(timeout=0.1)
                self.socket.sendall(data)
            except queue.Empty: continue
            except: 
                self.connected = False
                self.logger.error("通信链路断开！")

# ==================== 主应用程序 (注入你的交互与算法灵魂) ====================

class FranceHonorGlasses:
    def __init__(self):
        self.config = GlassesConfig()
        self.logger = Logger()
        self.client = ComputeBoxClient(self.config, self.logger)
        
        # 你的系统核心变量
        self.status = SystemStatus.INITIALIZING
        self.frame_id = 0
        self.frozen_frame = None       # Y 键定格的画面
        self.roi_box = None            # 用户圈定的矩形坐标 (x, y, w, h)
        
        # 算法物理状态
        self.base_anchor = None        # 全局基础点 [x, y, z]
        self.last_base_anchor = None   # 上一帧基础点，用于一阶低通滤波
        self.camera_pose = [0.0, 0.0, 0.0]
        
        # 能效保活计数器
        self.static_frame_count = 0
        self.lost_frame_count = 0

    def simulate_imu(self):
        """模拟低功耗 IMU 变动检测：产生微小高斯噪声，偶尔产生大变动"""
        noise = np.random.normal(0, 0.01, 3)
        # 模拟真实手晃动：有 10% 的概率发生较大位移变动
        if np.random.rand() < 0.1:
            noise += np.random.uniform(0.5, 1.5, 3)
        return noise

    def run(self):
        self.logger.info("FranceHonor AR 眼镜客户端正在拉起摄像头...")
        cap = cv2.VideoCapture(0) # 调用本地摄像头
        
        # 强制启动通信底座
        self.client.start()
        
        # ROI 选择相关的鼠标回调
        cv2.namedWindow("FranceHonor Display")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frame_id += 1
            
            # 读取 IMU 数据并判断运动烈度
            imu_accel = self.simulate_imu()
            is_imu_active = np.linalg.norm(imu_accel) > 0.3
            
            # ==================== 核心：Main.py 的 IMU 拦截与四帧保活机制 ====================
            if self.status == SystemStatus.TRACKING and not is_imu_active:
                self.static_frame_count += 1
                if self.static_frame_count < 4:
                    # 功耗拦截：手没动，直接快速渲染上一帧，不去惊动后端复杂的计算盒！
                    self._render_canvas(frame)
                    if cv2.waitKey(1) & 0xFF == ord('n'): break
                    continue
                else:
                    # 到了第 4 帧，强制打破拦截，放行一次以更新系统心跳
                    self.static_frame_count = 0
                    self.logger.info("【保活刷新】触发 4 帧强制心跳放行")
            
            # ==================== 核心状态机控制逻辑 ====================
            if self.status == SystemStatus.INITIALIZING:
                # 仅拉起摄像头，提示用户按 Y 定格
                cv2.putText(frame, "STATUS: INITIALIZING | Press 'Y' to Freeze Canvas", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            elif self.status == SystemStatus.FROZEN:
                # 画面定格，强行要求画圈
                frame = self.frozen_frame.copy()
                cv2.putText(frame, "STATUS: CANVAS FROZEN | Drag a Box to select Plane ROI, then press 'B'", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                if self.roi_box:
                    x, y, w, h = self.roi_box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
            elif self.status == SystemStatus.TRACKING:
                # 基于特征点和低通滤波平滑渲染方块
                self._update_anchor_low_pass(is_imu_active)
                self._render_canvas(frame)
                
                # 模拟视觉偶尔丢失的情况
                if np.random.rand() < 0.02: 
                    self.status = SystemStatus.LOST
                    self.lost_frame_count = 0

            elif self.status == SystemStatus.LOST:
                self.lost_frame_count += 1
                cv2.putText(frame, f"STATUS: LOST | Blind Tracking Frame: {self.lost_frame_count}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if self.lost_frame_count <= self.config.LOST_MAX_DELAY:
                    # 没超过最大延迟，继续用 IMU 盲推重定位
                    self.camera_pose[0] += imu_accel[0] * 0.1
                    self.status = SystemStatus.RECOVERING
                else:
                    # 超过 5 帧，彻底死心，退回初始化状态
                    self.logger.warning("丢失超过 5 帧！重置系统。")
                    self.status = SystemStatus.INITIALIZING
                    
            elif self.status == SystemStatus.RECOVERING:
                # 模拟重定位成功
                if np.random.rand() < 0.7:
                    self.status = SystemStatus.TRACKING
                    self.logger.info("重定位成功，重回追踪状态！")
                else:
                    self.status = SystemStatus.LOST

            # 将打好标签的数据，通过元宝的高性能网络管道发送给外部计算盒
            v_data = VisualData(
                timestamp=time.time(),
                image_id=self.frame_id,
                status=self.status.value,
                pose=self.camera_pose,
                is_imu_active=is_imu_active,
                base_anchor=self.base_anchor if self.base_anchor else [0.0, 0.0, 0.0]
            )
            self.client.send_data(v_data)

            # ==================== 按键交互响应 ====================
            cv2.imshow("FranceHonor Display", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'): # 按 N 关闭程序
                break
            elif key == ord('y') and self.status == SystemStatus.INITIALIZING:
                self.frozen_frame = frame.copy()
                self.status = SystemStatus.FROZEN
                # 弹出 OpenCV 选框器供画圈
                self.roi_box = cv2.selectROI("FranceHonor Display", frame, fromCenter=False, showCrosshair=True)
            elif key == ord('b') and self.status == SystemStatus.FROZEN and self.roi_box:
                # 核心：首帧锚定。在圈内随机选择一个基础点，并强行将这一帧存入历史字典
                x, y, w, h = self.roi_box
                rand_x = x + w // 2
                rand_y = y + h // 2
                self.base_anchor = [float(rand_x), float(rand_y), 1.0] # 模拟 3D 空间基础点
                self.last_base_anchor = self.base_anchor.copy()
                self.status = SystemStatus.TRACKING
                self.logger.info(f"【首帧锚定成功】全局基础点已锁定在 ROI 中心: {self.base_anchor}")

        cap.release()
        cv2.destroyAllWindows()
        self.client.running = False

    def _update_anchor_low_pass(self, is_imu_active):
        """核心：mark/symbol 的一阶低通滤波平滑逻辑"""
        if not self.base_anchor: return
        
        # 模拟当前帧观测到的基础点（带环境抖动噪声）
        observed_anchor = [self.base_anchor[0] + np.random.normal(0, 0.5), 
                           self.base_anchor[1] + np.random.normal(0, 0.5), 1.0]
        
        if not is_imu_active:
            beta = 0.3  # 静止低动态：加大阻尼，追求极致平滑不抖动
        else:
            beta = 0.1  # 高动态移动：减小阻尼，快速响应追求跟手
            
        # 一阶低通滤波公式
        self.base_anchor[0] = observed_anchor[0] * (1 - beta) + self.last_base_anchor[0] * beta
        self.base_anchor[1] = observed_anchor[1] * (1 - beta) + self.last_base_anchor[1] * beta
        self.last_base_anchor = self.base_anchor.copy()

    def _render_canvas(self, frame):
        """核心：render/painter 的相机内参透视投影与画方块"""
        if not self.base_anchor: return
        bx, by = int(self.base_anchor[0]), int(self.base_anchor[1])
        
        # 模拟 3D 顶点投影到 2D 像素面上的加减（简化实现）
        pts = np.array([
            [bx - 30, by - 30], [bx + 30, by - 30], 
            [bx + 30, by + 30], [bx - 30, by + 30]
        ], np.int32)
        
        # 绘制并填充上色（增加 AR 物体立体感）
        cv2.fillPoly(frame, [pts], (0, 255, 0))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        cv2.putText(frame, "STATUS: TRACKING | Base Anchor Locked", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

if __name__ == "__main__":
    app = FranceHonorGlasses()
    app.run()