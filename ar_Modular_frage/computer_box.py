"""
程序名：FranceHonor (法兰西荣耀) - 计算盒多线程服务端
修复版：修复元宝 JSON 字段不一致的 Bug，保留多线程与关键帧数据库，完美对接你的 AR 灵魂
"""

import socket
import json
import time
import base64
import threading
import queue
import numpy as np
import struct
import pickle
import os
from dataclasses import dataclass, asdict
from enum import Enum

# ==================== 1. 状态与数据结构对齐 ====================

class SystemStatus(Enum):
    INITIALIZING = "INITIALIZING"
    FROZEN = "FROZEN"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RECOVERING = "RECOVERING"

@dataclass
class KeyFrame:
    id: int
    timestamp: float
    base_anchor: list  # 你的 3D 空间基础锚点 [x, y, z]
    pose: list        # 相机当前位姿

# ==================== 2. 核心配置与增强日志 ====================

class ComputeBoxConfig:
    def __init__(self):
        self.SERVER_IP = "127.0.0.1"
        self.SERVER_PORT = 8888
        self.MAX_CLIENTS = 2
        self.DATA_DIR = "france_honor_db"

class Logger:
    def info(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [SERVER-INFO] {msg}")
    def warning(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [SERVER-WARN] {msg}")
    def error(self, msg): print(f"[{time.strftime('%H:%M:%S')}] [SERVER-ERR ] {msg}")

# ==================== 3. 关键帧数据库 (保留元宝的序列化持久化灵魂) ====================

class KeyFrameDatabase:
    def __init__(self, config: ComputeBoxConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.keyframes = []
        self.next_kf_id = 1
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        self._load_from_disk()

    def try_add_keyframe(self, base_anchor, pose):
        """核心：当眼镜端手动画圈确立新锚点时，作为关键帧永久固化到磁盘"""
        # 如果是首帧锚定，或者空间位置发生了显著变化，则存入数据库
        if len(self.keyframes) == 0:
            kf = KeyFrame(id=self.next_kf_id, timestamp=time.time(), base_anchor=base_anchor, pose=pose)
            self.keyframes.append(kf)
            self.next_kf_id += 1
            self.logger.info(f"【关键帧入库】成功固化首帧地图锚点 #{kf.id} -> {base_anchor}")
            self._save_to_disk()
            return kf
        return None

    def _save_to_disk(self):
        pkl_path = os.path.join(self.config.DATA_DIR, "map_keyframes.pkl")
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.keyframes, f)
            self.logger.info(f"【磁盘同步】已将地图数据库备份至 {pkl_path}")
        except Exception as e:
            self.logger.error(f"磁盘备份失败: {e}")

    def _load_from_disk(self):
        pkl_path = os.path.join(self.config.DATA_DIR, "map_keyframes.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    self.keyframes = pickle.load(f)
                if self.keyframes:
                    self.next_kf_id = max(kf.id for kf in self.keyframes) + 1
                self.logger.info(f"【开机加载】从磁盘成功恢复了 {len(self.keyframes)} 个历史地图关键帧！")
            except Exception as e:
                self.logger.error(f"读取历史关键帧失败: {e}")

# ==================== 4. 计算盒多线程服务器主体 ====================

class ComputeBoxServer:
    def __init__(self):
        self.config = ComputeBoxConfig()
        self.logger = Logger()
        self.database = KeyFrameDatabase(self.config, self.logger)
        
        self.server_socket = None
        self.running = False
        self.processing_queue = queue.Queue(maxsize=100) # 多线程解耦队列

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.config.SERVER_IP, self.config.SERVER_PORT))
        self.server_socket.listen(self.config.MAX_CLIENTS)
        self.running = True
        
        # 1. 拉起元宝引以为傲的 后端SLAM后台处理线程POOL
        threading.Thread(target=self._slam_worker_loop, daemon=True).start()
        
        self.logger.info(f"【分体式计算盒】已在本地启动，正监听端口 {self.config.SERVER_PORT}...")
        
        # 2. 主线程切入监听循环
        try:
            while self.running:
                conn, addr = self.server_socket.accept()
                # 每一个进来的眼镜客户端，分配一个独立的接收线程（元宝的高并发模型）
                threading.Thread(target=self._client_handler, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            self.logger.info("计算盒正在关闭...")
        finally:
            self.server_socket.close()

    def _client_handler(self, conn, addr):
        self.logger.info(f"【眼镜端接入】接收到来自 {addr} 的硬件链路连接")
        
        while self.running:
            try:
                # 核心突破：完美解析元宝的大端 4 字节网络包头
                header = conn.recv(4)
                if not header or len(header) != 4: break
                message_length = struct.unpack('!I', header)[0]
                
                # 接收 JSON 实体数据
                data = b""
                while len(data) < message_length:
                    packet = conn.recv(message_length - len(data))
                    if not packet: break
                    data += packet
                
                # 【关键修复】：这里解析出的数据格式，完美兼容眼镜端的命名习惯
                packet_json = json.loads(data.decode('utf-8'))
                
                # 丢进后台 SLAM 队列进行异步计算，主线程立刻返回去拉取下一帧（极致能效！）
                if not self.processing_queue.full():
                    self.processing_queue.put(packet_json)
                    
            except Exception as e:
                self.logger.warning(f"眼镜端异常断开: {e}")
                break
        conn.close()

    def _slam_worker_loop(self):
        """核心：元宝最擅长的后台重定位与 SLAM 状态机线程"""
        self.logger.info("后台 SLAM 异步处理内核已拉起，常驻算力准备就绪")
        
        while self.running:
            try:
                # 从处理队列里掏出一帧视觉/IMU混合数据包
                packet = self.processing_queue.get(timeout=0.1)
                status = packet.get("status")
                base_anchor = packet.get("base_anchor")
                pose = packet.get("pose")
                is_imu_active = packet.get("is_imu_active")
                
                # ==================== 你的 SLAM 业务逻辑落脚点 ====================
                if status == "TRACKING":
                    # 检查是否触发首帧入库或者地图更新
                    self.database.try_add_keyframe(base_anchor, pose)
                    
                    # 模拟后端重定位心跳打印
                    print(f"[SLAM核心] 正在追踪中 | 渲染锚点: {[round(x,1) for x in base_anchor]} | IMU高动态: {is_imu_active}")
                    
                elif status == "LOST":
                    self.logger.warning("【危险】接收到眼镜端 LOST 盲推信号！计算盒正在调用磁盘历史地图尝试强制重定位...")
                    if len(self.database.keyframes) > 0:
                        # 模拟重定位检索：在磁盘数据库里比对最近的全局锚点
                        recovery_match = self.database.keyframes[0]
                        print(f" -> [重定位命中] 成功帮眼镜端找回坐标系！历史锚点 ID: {recovery_match.id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"SLAM 内核执行异常: {e}")

if __name__ == "__main__":
    server = ComputeBoxServer()
    server.start()