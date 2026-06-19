# 导入必要的Python库
import cv2         # OpenCV库，用于图像采集和处理
import threading   # 多线程库，用于异步处理云端任务
# 注意：这里缺少了numpy的导入，但在主程序入口使用了numpy

# 从自定义模块导入各个核心组件
from glasses.vision_core import VisionCore  # 视觉核心，负责特征提取
from glasses.pose_core import PoseCore      # 位姿核心，负责位姿解算
from map_engine import MapEngine    # 地图引擎，负责SLAM和地图管理
from Transport_box import BoxTransport  # 传输层，负责与云端通信
from FSM_box import BoxFSM          # 状态机，管理计算盒状态

class BoxRuntime:
    """
    计算盒运行时主类，负责协调计算盒的各个子系统
    
    这个类是计算盒的核心，它：
    1. 初始化视觉、位姿、地图、传输和状态机模块
    2. 从摄像头获取图像帧
    3. 进行视觉特征提取和位姿解算
    4. 更新地图
    5. 与云端协同进行AI推理
    6. 协调所有子系统协同工作
    """
    
    def __init__(self, camera_matrix):
        """
        初始化计算盒运行时系统
        
        参数:
        camera_matrix: 相机内参矩阵，用于位姿解算
                       3x3矩阵，格式为：
                       [[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]]
        """
        # 1. 核心模块初始化
        # 创建视觉核心实例，负责特征提取
        self.vision = VisionCore()
        
        # 创建位姿核心实例，传入相机内参
        self.pose = PoseCore(camera_matrix)
        
        # 创建地图引擎实例，负责SLAM和地图管理
        self.map = MapEngine()
        
        # 创建传输层实例，监听8888端口（等待云端连接）
        self.transport = BoxTransport(port=8888)
        
        # 创建状态机实例，管理计算盒工作状态
        self.fsm = BoxFSM()
        
        # 打开摄像头（默认摄像头）
        self.camera = cv2.VideoCapture(0)
        
        # 注意：这里没有检查摄像头是否成功打开
        # 也没有检查各模块是否初始化成功

    def run(self):
        """
        运行计算盒主循环
        
        这个方法实现了计算盒的主要工作流程：
        1. 从摄像头获取图像帧
        2. 视觉特征提取
        3. 位姿解算
        4. 地图更新
        5. 云端AI协同
        6. 状态机管理
        
        这个循环会一直运行，直到摄像头无法读取或程序被中断
        """
        print("[Box] 计算盒系统启动，开始循环处理...")
        
        # 主循环，持续处理摄像头图像
        while True:
            # 读取一帧图像
            ret, frame = self.camera.read()
            
            # 如果读取失败，退出循环
            if not ret: 
                break

            # 2. 视觉前端：特征提取
            # 处理当前帧，提取视觉特征
            vision_data = self.vision.process_frame(frame)
            
            # 3. 位姿解算（纯视觉位姿）
            # 使用视觉数据计算当前位姿
            # 注意：这里参数不完整，'...'表示需要传入具体的特征点
            # 通常需要传入3D点和对应的2D点
            pose_matrix = self.pose.solve_pose(vision_data['keypoints'], ...)
            
            # 4. 地图更新与融合
            # 如果位姿解算成功，更新地图
            if pose_matrix is not None:
                # 将VIO数据（位姿+特征）传递给地图引擎
                self.map.receive_vio_data({'pose': pose_matrix, 'features': vision_data})

            # 5. 云端 AI 协同（如果需要）
            # 如果状态机处于空闲状态，触发云端任务
            if self.fsm.state == 'IDLE':
                # 创建新线程执行云端任务，避免阻塞主循环
                # 这样视觉前端和位姿解算可以继续运行
                threading.Thread(target=self._cloud_task, args=(frame,)).start()
                
            # 注意：这里没有控制帧率
            # 摄像头可能会以最高速度运行，占用大量CPU
            # 可以考虑添加延迟来控制帧率

    def _cloud_task(self, frame):
        """
        异步云端任务，避免阻塞视觉流
        
        参数:
        frame: 要发送到云端的图像帧
        
        这个函数在一个独立的线程中运行，负责：
        1. 压缩图像
        2. 发送到云端
        3. 接收并处理云端响应
        4. 更新状态机（如果需要）
        """
        # 压缩图像为JPEG格式，质量为70%
        # 这样可以减少网络传输的数据量
        _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        # 发送压缩后的图像到云端，并接收结果
        result = self.transport.send_and_receive(img.tobytes())
        
        # 处理云端返回的结果
        # 这里只是占位符，实际需要根据云端返回的JSON解析和处理
        # 例如：更新地图、触发渲染、记录日志等
        # 处理结果逻辑...

# 主程序入口
# 当这个文件被直接运行时，执行以下代码
if __name__ == "__main__":
    # 导入numpy库（原代码中缺少这行）
    import numpy as np
    
    # 定义相机内参矩阵
    # 这是一个示例矩阵，实际需要使用标定得到的真实内参
    # 格式：[[fx, 0, cx],
    #        [0, fy, cy],
    #        [0,  0,  1]]
    cam_matrix = np.array([[800, 0, 320], 
                           [0, 800, 240], 
                           [0, 0, 1]])
    
    # 创建BoxRuntime实例
    runtime = BoxRuntime(cam_matrix)
    
    # 运行计算盒系统
    runtime.run()