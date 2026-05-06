import time
import enum

# 定义系统状态枚举
# 枚举（Enumeration）是一种特殊的数据类型，用于表示一组命名的常量
# 在状态机设计中，使用枚举明确表示系统的所有可能状态
class SystemState(enum.Enum):
    """
    系统状态枚举
    定义了SLAM系统可能处于的四种状态
    
    状态机（State Machine）是一种设计模式：
    - 系统在任何时刻只能处于一个状态
    - 状态之间的转换由特定事件触发
    - 每个状态有自己的行为逻辑
    
    这四个状态构成了完整的SLAM工作流程：
    1. INITIALIZING → TRACKING: 找到足够的特征，开始追踪
    2. TRACKING → LOST: 视觉跟踪失败
    3. LOST → RECOVERING: 尝试重定位
    4. RECOVERING → TRACKING: 重定位成功
    5. RECOVERING → LOST: 重定位失败
    6. LOST → INITIALIZING: 丢失时间过长，彻底重置
    """
    INITIALIZING = 0  # 初始化状态：寻找足够的特征点，建立初始地图
    TRACKING = 1      # 追踪状态：正常定位和建图
    LOST = 2          # 丢失状态：视觉信号中断，使用IMU预测
    RECOVERING = 3    # 恢复状态：尝试重定位，找回跟踪

class SimpleSLAMSystem:
    """
    基于状态机的简化版SLAM系统
    
    核心设计思想：状态模式（State Pattern）
    将系统在不同状态下的行为封装到不同的方法中
    通过状态转换控制系统的行为逻辑
      
    状态机优势：
    1. 代码清晰：每个状态的行为逻辑独立
    2. 易于维护：状态转换明确，易于调试
    3. 鲁棒性强：能处理各种异常情况
    4. 可扩展性好：容易添加新的状态
    """ 
    
    def __init__(self):
        """
        初始化SLAM系统
        设置初始状态、时间记录和系统参数
        """
        # 当前系统状态
        # 初始状态为INITIALIZING，表示系统刚开始运行
        self.state = SystemState.INITIALIZING
        
        # 丢失状态的开始时间
        # 用于记录系统进入LOST状态的时间点
        # 当视觉丢失超过一定时间后，系统会完全重置
        self.lost_start_time = 0
        
        # 最大丢失持续时间（秒）
        # 如果视觉丢失超过这个时间，系统会完全重置
        # 这是一个安全机制，避免系统长时间处于不可靠状态
        self.max_lost_duration = 5.0
        
        # 模拟已有的算法组件（实际应用中需要实例化这些组件）
        # 这里用注释表示，实际代码中需要创建这些对象
        
        # 视觉追踪器：负责从图像中提取特征并跟踪
        # self.tracker = VisualTracker()
        
        # IMU管理器：处理惯性测量单元数据
        # self.imu = IMUManager()
        
        # 重定位器：在丢失后尝试重新定位
        # self.relocalizer = Relocalizer()

    def main_loop(self, frame):
        """
        主循环：每一帧图像进来，只根据当前状态执行对应的逻辑区块
        
        这是状态机的核心调度函数：
        1. 检查当前系统状态
        2. 根据状态调用对应的处理函数
        3. 每个处理函数负责该状态下的具体行为
        
        参数:
        frame: 当前图像帧
        """
        # 检查当前状态，并调用对应的处理函数
        if self.state == SystemState.INITIALIZING:
            # 系统正在初始化，寻找足够的特征点
            self._handle_initializing(frame)
            
        elif self.state == SystemState.TRACKING:
            # 系统正常跟踪，执行核心定位和建图
            self._handle_tracking(frame)
            
        elif self.state == SystemState.LOST:
            # 视觉丢失，使用IMU进行预测
            self._handle_lost(frame)
            
        elif self.state == SystemState.RECOVERING:
            # 尝试重定位，找回跟踪
            self._handle_recovering(frame)

    # ==========================================================
    # 逻辑区块 A：初始化
    # 系统启动时的状态，寻找足够的特征点建立初始地图
    # ==========================================================
    def _handle_initializing(self, frame):
        """
        处理初始化状态
        
        在这个状态下，系统需要：
        1. 检测图像中的特征点
        2. 判断特征点数量是否足够
        3. 建立初始地图和坐标系
        4. 切换到追踪状态
        
        参数:
        frame: 当前图像帧
        
        实现要点：
        - 需要足够的特征点（通常50-100个）
        - 特征点需要分布均匀
        - 可能需要多帧初始化以提高稳定性
        - 初始化失败时保持INITIALIZING状态
        """
        print("[State: INITIALIZING] 正在寻找地面或特征点集中区域...")
        
        # 尝试初始化：检测特征点数量是否达标
        # 返回True表示初始化成功，False表示失败
        success = self._try_init(frame)
        
        if success:
            # 初始化成功，切换到追踪状态
            print(">>> 初始化成功，切入追踪模式")
            self.state = SystemState.TRACKING
        else:
            # 初始化失败，保持在初始化状态
            # 实际应用中可能需要记录失败次数，多次失败后给出提示
            pass

    # ==========================================================
    # 逻辑区块 B：正常追踪 (核心业务)
    # 系统的主要工作状态，执行定位和建图
    # ==========================================================
    def _handle_tracking(self, frame):
        """
        处理追踪状态（核心业务逻辑）
        
        在这个状态下，系统需要：
        1. 执行视觉PnP追踪，估计相机位姿
        2. 融合IMU数据，提高定位精度和稳定性
        3. 检测跟踪质量，判断是否需要切换状态
        4. 保存关键帧，更新地图
        
        参数:
        frame: 当前图像帧
        """
        print("[State: TRACKING] 正常工作周期")
        
        # 1. 尝试视觉PnP追踪
        # 通过特征匹配和PnP算法估计相机位姿
        # 返回True表示跟踪成功，False表示跟踪失败
        tracking_ok = self._visual_pnp_track(frame)
        
        # 2. 结合IMU进行卡尔曼滤波融合
        # 视觉定位可能会有抖动，IMU数据可以提供平滑的运动估计
        # 卡尔曼滤波融合视觉和IMU数据，得到更稳定的位姿估计
        self._imu_fusion_update()

        # 3. 检查跟踪质量
        if not tracking_ok:
            # 视觉跟踪失败，切换到丢失状态
            print("!!! 预警：视觉特征丢失，切入LOST模式")
            
            # 记录丢失开始时间
            self.lost_start_time = time.time()
            
            # 切换到丢失状态
            self.state = SystemState.LOST
        else:
            # 跟踪成功，保持在追踪状态
            # 可以在这里执行建图、关键帧保存等操作
            pass

    # ==========================================================
    # 逻辑区块 C：视觉丢失 (临时应急)
    # 视觉跟踪失败时的应急状态，使用IMU进行预测
    # ==========================================================
    def _handle_lost(self, frame):
        """
        处理视觉丢失状态（临时应急）
        
        在这个状态下，系统需要：
        1. 完全不信任视觉输入
        2. 仅依靠IMU进行航位推算
        3. 检查是否超过最大丢失时间
        4. 决定是继续丢失状态还是尝试恢复
        
        参数:
        frame: 当前图像帧
        """
        print("[State: LOST] 警告：视觉信号中断")
        
        # 行为：完全不信任视觉输入，仅依靠IMU航位推算
        # IMU（惯性测量单元）包含加速度计和陀螺仪
        # 即使在视觉丢失的情况下，也能提供短时的运动估计
        self._predict_pose_by_imu_only()
        
        # 检查超时逻辑
        # 计算从进入LOST状态到现在的时间
        lost_duration = time.time() - self.lost_start_time
        
        if lost_duration > self.max_lost_duration:
            # 丢失时间过长，系统强制重置
            # 这是一种安全机制，避免系统长时间处于不可靠状态
            print("!!! 错误：丢失时间过长，系统强制重置")
            self.state = SystemState.INITIALIZING
        else:
            # 只要没超时，就尝试去"恢复"
            # 切换到恢复状态，尝试重定位
            self.state = SystemState.RECOVERING

    # ==========================================================
    # 逻辑区块 D：重定位 (找回模式)
    # 尝试找回跟踪，与历史关键帧匹配
    # ==========================================================
    def _handle_recovering(self, frame):
        """
        处理重定位状态（找回模式）
        
        在这个状态下，系统需要：
        1. 启动全局搜索逻辑
        2. 与历史关键帧进行特征匹配
        3. 判断是否重定位成功
        4. 根据结果决定下一个状态
        
        参数:
        frame: 当前图像帧
        """
        print("[State: RECOVERING] 正在尝试与历史关键帧匹配...")
        
        # 行为：启动全局搜索逻辑
        # 使用词袋模型（BoW）或全局特征匹配
        # 与之前保存的所有关键帧进行匹配
        found = self._try_relocalization(frame)
        
        if found:
            # 重定位成功，切换回追踪状态
            print(">>> 重定位成功！恢复追踪")
            self.state = SystemState.TRACKING
        else:
            # 没找回就继续回LOST状态使用IMU预测
            # 这是一个循环：LOST → RECOVERING → LOST ...
            # 直到成功恢复或超时重置
            self.state = SystemState.LOST

    # ----------------------------------------------------------
    # 下面是具体的底层算法占位符（之前写的代码填到这里）
    # 这些是状态机的"行为"，每个状态对应不同的行为
    # ----------------------------------------------------------
    
    def _try_init(self, frame):
        """
        尝试初始化系统
        
        实现初始化逻辑：
        1. 检测图像中的特征点
        2. 检查特征点数量和质量
        3. 建立初始地图
        4. 返回是否成功
        
        参数:
        frame: 当前图像帧
        
        返回:
        bool: 初始化是否成功
        """
        # 在这里实现初始化逻辑
        # 示例：检测特征点，如果数量大于50则返回True
        return True
    
    def _visual_pnp_track(self, frame):
        """
        视觉PnP追踪
        
        使用特征匹配和PnP算法估计相机位姿：
        1. 提取当前帧特征
        2. 与上一帧或参考帧匹配
        3. 使用PnP求解相机位姿
        4. 检查重投影误差
        
        参数:
        frame: 当前图像帧
        
        返回:
        bool: 追踪是否成功
        """
        # 在这里实现视觉追踪逻辑
        return True  # 返回True/False决定是否丢失
    
    def _imu_fusion_update(self):
        """
        IMU融合更新
        
        使用卡尔曼滤波融合视觉和IMU数据：
        1. 获取IMU数据（加速度、角速度）
        2. 预测步骤：使用IMU进行状态预测
        3. 更新步骤：使用视觉测量进行校正
        4. 输出融合后的位姿
        
        这是VIO（视觉惯性里程计）的核心
        """
        # 在这里实现IMU融合逻辑
        pass
    
    def _predict_pose_by_imu_only(self):
        """
        仅使用IMU预测位姿
        
        在视觉丢失时，仅使用IMU进行航位推算：
        1. 获取IMU角速度和加速度
        2. 积分得到旋转和平移
        3. 注意：IMU有漂移，只能短时使用
        
        这是VIO中的"预测模式"
        """
        # 在这里实现IMU预测逻辑
        pass
    
    def _try_relocalization(self, frame):
        """
        尝试重定位
        
        与历史关键帧匹配，尝试找回跟踪：
        1. 提取当前帧特征
        2. 与所有历史关键帧匹配
        3. 使用词袋模型加速搜索
        4. 通过PnP计算相对位姿
        
        参数:
        frame: 当前图像帧
        
        返回:
        bool: 重定位是否成功
        """
        # 在这里实现重定位逻辑
        return False
    

"""
系统首先初始化参考帧并提取特征点建立初始状态。
在正常 tracking 状态下，通过特征匹配与 Perspective-n-Point (PnP) 进行位姿估计，并结合 IMU 进行短时稳定。
当视觉 tracking 质量下降时，系统进入失效状态，短时间依赖 IMU 进行姿态预测。
如果在失效窗口内，当前帧与历史 Keyframe 匹配成功，则触发重定位（relocalization），恢复 tracking。
若匹配失败，则系统进入完全丢失状态，等待重新初始化。
在恢复过程中，通过插值或平滑方式避免虚拟物体的跳变。
"""

# 额，基于我们三个小时的研究，下面的注释是没有必要的，过于详细以至于拖累了进度。


"""
完整代码：

import time
import enum
import cv2
import numpy as np

# 定义状态枚举
class SystemState(enum.Enum):
    INITIALIZING = 0
    TRACKING = 1
    LOST = 2
    RECOVERING = 3

class CompleteSLAMSystem:
    def __init__(self, camera_matrix, dist_coeffs):
        self.state = SystemState.INITIALIZING
        self.lost_start_time = 0
        self.max_lost_duration = 5.0
        
        # 初始化ORB特征检测器
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 相机参数
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 跟踪状态
        self.last_kp = None
        self.last_des = None
        self.current_rvec = None
        self.current_tvec = None
        
        # 模拟的3D点（假设平面在z=0）
        self.object_points = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0]
        ], dtype=np.float32)
        
        # 关键帧列表
        self.keyframes = []
        
        # 状态统计
        self.frame_count = 0
        self.tracking_success_count = 0
        
        print("SLAM系统初始化完成")

    def main_loop(self, frame):
        ""
        主循环：处理每一帧
        ""
        self.frame_count += 1
        
        if self.state == SystemState.INITIALIZING:
            self._handle_initializing(frame)
        elif self.state == SystemState.TRACKING:
            self._handle_tracking(frame)
        elif self.state == SystemState.LOST:
            self._handle_lost(frame)
        elif self.state == SystemState.RECOVERING:
            self._handle_recovering(frame)
        
        # 可视化当前状态
        return self._visualize_state(frame)

    # ==========================================================
    # 初始化状态
    # ==========================================================
    def _handle_initializing(self, frame):
        print(f"[帧 {self.frame_count}] 初始化: 寻找特征点...")
        
        # 提取特征
        kp, des = self.orb.detectAndCompute(frame, None)
        
        if kp is not None and len(kp) > 50:
            # 初始化成功
            self.last_kp = kp
            self.last_des = des
            
            # 创建初始关键帧
            self.keyframes.append({
                'kp': kp,
                'des': des,
                'rvec': np.zeros((3, 1)),
                'tvec': np.zeros((3, 1))
            })
            
            print(">>> 初始化成功！找到", len(kp), "个特征点")
            self.state = SystemState.TRACKING
        else:
            # 初始化失败
            if kp is not None:
                print(f"  特征点不足: {len(kp)} (需要 > 50)")
            else:
                print("  未检测到特征点")

    # ==========================================================
    # 追踪状态
    # ==========================================================
    def _handle_tracking(self, frame):
        # 提取特征
        kp, des = self.orb.detectAndCompute(frame, None)
        
        if kp is None or len(kp) < 20:
            print(f"[帧 {self.frame_count}] 追踪失败: 特征点不足")
            self.state = SystemState.LOST
            self.lost_start_time = time.time()
            return
        
        # 与上一帧匹配
        if self.last_des is not None:
            matches = self.bf.match(self.last_des, des)
            
            # 筛选优质匹配
            good_matches = [m for m in matches if m.distance < 30]
            
            if len(good_matches) > 15:
                # 追踪成功
                self.tracking_success_count += 1
                
                # 计算位姿（简化版本，实际应该用PnP）
                src_pts = np.float32([self.last_kp[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                # 计算单应性矩阵
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # 保存当前特征
                    self.last_kp, self.last_des = kp, des
                    
                    # 每50帧保存一个关键帧
                    if self.frame_count % 50 == 0:
                        self.keyframes.append({
                            'kp': kp.copy(),
                            'des': des.copy()
                        })
                        print(f"  保存关键帧 #{len(self.keyframes)}")
                    
                    return
        
        # 追踪失败
        print(f"[帧 {self.frame_count}] 追踪失败: 匹配点不足")
        self.state = SystemState.LOST
        self.lost_start_time = time.time()

    # ==========================================================
    # 丢失状态
    # ==========================================================
    def _handle_lost(self, frame):
        lost_duration = time.time() - self.lost_start_time
        
        if lost_duration > self.max_lost_duration:
            print(f"!!! 丢失超过 {self.max_lost_duration} 秒，系统重置")
            self.state = SystemState.INITIALIZING
            self.last_kp = None
            self.last_des = None
        else:
            print(f"[帧 {self.frame_count}] 丢失中 ({lost_duration:.1f}s)")
            # 尝试恢复
            self.state = SystemState.RECOVERING

    # ==========================================================
    # 恢复状态
    # ==========================================================
    def _handle_recovering(self, frame):
        print(f"[帧 {self.frame_count}] 尝试重定位...")
        
        # 提取特征
        kp, des = self.orb.detectAndCompute(frame, None)
        
        if kp is None or des is None:
            self.state = SystemState.LOST
            return
        
        # 与历史关键帧匹配
        best_score = 0
        best_match = None
        
        for i, kf in enumerate(self.keyframes):
            matches = self.bf.match(des, kf['des'])
            good_matches = [m for m in matches if m.distance < 40]
            
            if len(good_matches) > best_score:
                best_score = len(good_matches)
                best_match = kf
        
        # 判断是否重定位成功
        if best_score > 20:
            print(f">>> 重定位成功！匹配到关键帧，{best_score} 个匹配点")
            self.last_kp = best_match['kp']
            self.last_des = best_match['des']
            self.state = SystemState.TRACKING
        else:
            print(f"  重定位失败，最佳匹配: {best_score} 个点")
            self.state = SystemState.LOST

    # ==========================================================
    # 可视化
    # ==========================================================
    def _visualize_state(self, frame):
        # 创建显示副本
        display = frame.copy()
        
        # 状态文本
        state_text = f"状态: {self.state.name}"
        state_color = (0, 255, 0)  # 绿色
        
        if self.state == SystemState.LOST:
            state_color = (0, 165, 255)  # 橙色
        elif self.state == SystemState.RECOVERING:
            state_color = (255, 255, 0)  # 青色
        elif self.state == SystemState.INITIALIZING:
            state_color = (255, 0, 0)  # 红色
        
        # 绘制状态
        cv2.putText(display, state_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
        
        # 帧计数
        cv2.putText(display, f"帧: {self.frame_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 关键帧数量
        cv2.putText(display, f"关键帧: {len(self.keyframes)}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 在追踪状态时显示特征点
        if self.state == SystemState.TRACKING and self.last_kp is not None:
            # 绘制特征点
            for kp in self.last_kp[:50]:  # 只显示前50个
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(display, (x, y), 3, (0, 255, 255), -1)
        
        return display

# 主程序
if __name__ == "__main__":
    # 相机参数（示例值，实际需要标定）
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    # 创建SLAM系统
    slam = CompleteSLAMSystem(camera_matrix, dist_coeffs)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    
    print("\n=== SLAM系统启动 ===")
    print("按 'q' 键退出")
    print("按 'r' 键手动重置系统")
    print("==================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 水平翻转，使操作更直观
        frame = cv2.flip(frame, 1)
        
        # 处理当前帧
        processed_frame = slam.main_loop(frame)
        
        # 显示结果
        cv2.imshow("SLAM System", processed_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 手动重置
            slam.state = SystemState.INITIALIZING
            slam.last_kp = None
            slam.last_des = None
            print("系统已手动重置")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 输出统计
    print(f"\n=== 运行统计 ===")
    print(f"总帧数: {slam.frame_count}")
    print(f"关键帧数量: {len(slam.keyframes)}")
    print(f"最终状态: {slam.state.name}")
    """
















"""
1.完整可运行的VIO示例


import numpy as np
import cv2
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

导入库详细解释

这些是Python的库导入语句，每个库都有特定的用途。我来详细解释每个部分的作用：
1. NumPy（数值计算库）
import numpy as np 
作用：这是科学计算的基础库，可以看作是Python中的"数学引擎"
在VIO中的用途：
# 1. 矩阵运算（VIO的核心）
#  R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # 旋转矩阵 
# t = np.array([1.0, 2.0, 3.0]) # 平移向量 
# pose = np.eye(4) # 4x4单位矩阵，表示位姿 

# 2. 线性代数运算 
# A = np.array([[1, 2], [3, 4]]) 
# B = np.array([[5, 6], [7, 8]])
# C = np.dot(A, B) # 矩阵乘法 
# D = np.linalg.inv(A) # 矩阵求逆 
# E = np.linalg.norm(t) # 计算向量的模（长度） 

# # 3. 数组操作 
# points_3d = np.zeros((100, 3)) # 创建100个3D点 
# trajectory = np.array([[x1, y1, z1], [x2, y2, z2], ...]) # 轨迹点 

为什么用NumPy：比Python列表快100倍以上，支持向量化运算

2. OpenCV（计算机视觉库）
import cv2 
作用：这是计算机视觉的核心库，专门处理图像和视频
在VIO中的用途：
# 1. 图像读取和处理 
# img = cv2.imread("image.jpg") # 读取图像 
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转为灰度图 
# resized = cv2.resize(img, (640, 480)) # 调整大小 
# undistorted = cv2.undistort(img, K, D) # 去除镜头畸变 
 
# 2. 特征检测和匹配（VIO视觉部分的核心） 
# kp, des = cv2.ORB_create().detectAndCompute(img, None) # ORB特征 
# sift = cv2.SIFT_create() # SIFT特征 
# surf = cv2.xfeatures2d.SURF_create() # SURF特征 

# 3. 特征匹配 
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
# matches = bf.match(des1, des2) # 暴力匹配 
# flann = cv2.FlannBasedMatcher(index_params, search_params) # FLANN匹配 

# 4. 几何计算 
# E, mask = cv2.findEssentialMat(pts1, pts2, K) # 本质矩阵 
# F, mask = cv2.findFundamentalMat(pts1, pts2) # 基础矩阵 
# H, mask = cv2.findHomography(pts1, pts2) # 单应性矩阵 
# R, t = cv2.recoverPose(E, pts1, pts2, K) # 从本质矩阵恢复位姿 
# _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D) # PnP求解 

# 5. 可视化 
# cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # 画圆 
# cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 画线 
# cv2.putText(img, "text", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # 文字 
# cv2.imshow("Window", img) # 显示图像 
# cv2.waitKey(1) # 等待按键 

为什么用OpenCV：专门为计算机视觉优化，有数百个图像处理函数

3. time（时间库）
import time 
作用：处理时间相关的操作，VIO中时间同步至关重要
在VIO中的用途：
# 1. 获取时间戳 
# timestamp = time.time() # 当前时间（秒，浮点数） 
# 示例：1630000000.123456 表示2021年8月26日的时间 

# 2. 计算时间间隔 
# start_time = time.time() 
# 执行一些操作... 
# end_time = time.time() 
# elapsed_time = end_time - start_time # 耗时 
# print(f"处理时间: {elapsed_time:.3f}秒") 

# 3. 控制频率 
# desired_fps = 30 
# frame_time = 1.0 / desired_fps 
# while True: 
         start = time.time() 
         process_frame() # 处理一帧 
         elapsed = time.time() - start 
         if elapsed < frame_time: 
            time.sleep(frame_time - elapsed) # 等待剩余时间 

 4. 超时检测 
 timeout = 5.0 # 5秒超时 
 start = time.time() 
 while condition: 
    if time.time() - start > timeout: 
        print("超时!") 
        break 
    
5. 性能分析 
import time 
t1 = time.perf_counter() # 更高精度的计时 
执行操作 
t2 = time.perf_counter() 
print(f"耗时: {t2 - t1:.6f}秒") 

为什么需要时间：IMU数据是100-1000Hz，相机是15-60Hz，需要精确的时间戳来同步



4. threading（多线程库）
import threading 
作用：实现多线程并行处理，提高VIO系统的实时性
在VIO中的用途：
# 1. 创建线程处理不同任务 
def imu_processing_thread(): 
    ""独立线程处理IMU数据"" 
    while running: 
        imu_data = read_imu()
        process_imu(imu_data) 
        
def camera_processing_thread(): 
    ""独立线程处理相机数据"" 
    while running: 
        image_data = read_camera() 
        process_image(image_data) 

创建线程 
imu_thread = threading.Thread(target=imu_processing_thread) 
camera_thread = threading.Thread(target=camera_processing_thread) 
启动线程 
imu_thread.start() 
camera_thread.start() 
2. 线程同步 
lock = threading.Lock() # 创建锁 
def shared_resource_access(): 
    with lock: # 获取锁 
        安全地访问共享资源 
        shared_data.append(new_data) 
3. 线程间通信 
import queue 
imu_queue = queue.Queue(maxsize=100) # 线程安全队列
image_queue = queue.Queue(maxsize=10) 
def producer(): data = get_data() imu_queue.put(data) # 生产者放入数据 def consumer(): data = imu_queue.get() # 消费者获取数据 process_data(data) # 4. 事件通知 event = threading.Event() def waiting_thread(): event.wait() # 等待事件 print("事件触发!") def signaling_thread(): time.sleep(2) event.set() # 触发事件 
为什么需要多线程：
IMU：1000Hz → 每1ms一个数据
相机：30Hz → 每33ms一帧
需要并行处理，否则会丢失数据
5. collections.deque（双端队列）
from collections import deque 
作用：高效的数据缓冲区，用于存储时间序列数据
在VIO中的用途：
# 1. 创建固定大小的缓冲区 imu_buffer = deque(maxlen=1000) # 最多存储1000个IMU数据 image_buffer = deque(maxlen=30) # 最多存储30帧图像 # 2. 添加数据 imu_data = {'timestamp': time.time(), 'gyro': gyro, 'accel': accel} imu_buffer.append(imu_data) # 添加到末尾 # 3. 查找特定时间的数据 def get_imu_between(t1, t2): ""获取时间t1到t2之间的IMU数据"" result = [] for data in imu_buffer: if t1 <= data['timestamp'] <= t2: result.append(data) return result # 4. 时间同步：找到最接近图像时间戳的IMU数据 image_time = image_data['timestamp'] closest_imu = min(imu_buffer, key=lambda x: abs(x['timestamp'] - image_time)) # 5. IMU预积分：获取两帧之间的所有IMU数据 prev_time = 0.0 current_time = 0.1 imu_between = [data for data in imu_buffer if prev_time <= data['timestamp'] <= current_time] # 6. 滑动窗口优化 window_size = 10 pose_window = deque(maxlen=window_size) # 只保留最近10个位姿 for pose in poses: pose_window.append(pose) optimize_window(list(pose_window)) # 优化窗口内的位姿 
为什么用deque：
比列表(list)更快地从两端添加/删除
自动限制大小，避免内存泄漏
线程不安全，但配合锁可以安全使用
6. matplotlib.pyplot（绘图库）
import matplotlib.pyplot as plt 
作用：数据可视化，用于调试和分析VIO系统
在VIO中的用途：
# 1. 绘制2D轨迹 def plot_2d_trajectory(trajectory): ""绘制2D轨迹图"" x = trajectory[:, 0] # X坐标 y = trajectory[:, 1] # Y坐标 plt.figure(figsize=(10, 8)) plt.plot(x, y, 'b-', linewidth=2, label='轨迹') plt.scatter(x[0], y[0], c='g', s=100, label='起点') plt.scatter(x[-1], y[-1], c='r', s=100, label='终点') plt.xlabel('X (m)') plt.ylabel('Y (m)') plt.title('相机轨迹') plt.legend() plt.grid(True) plt.axis('equal') plt.show() # 2. 绘制误差曲线 def plot_errors(errors): ""绘制误差随时间的变化"" plt.figure(figsize=(12, 6)) # 位置误差 plt.subplot(2, 1, 1) plt.plot(errors['position'], 'r-', linewidth=2) plt.xlabel('帧数') plt.ylabel('位置误差 (m)') plt.title('位置误差') plt.grid(True) # 角度误差 plt.subplot(2, 1, 2) plt.plot(errors['rotation'], 'b-', linewidth=2) plt.xlabel('帧数') plt.ylabel('角度误差 (度)') plt.title('旋转误差') plt.grid(True) plt.tight_layout() plt.show() # 3. 绘制特征点 def plot_features(image, kp): ""可视化特征点"" plt.figure(figsize=(10, 8)) plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # 绘制特征点 for kp in kp: x, y = kp.pt plt.scatter(x, y, s=20, c='r', marker='o', alpha=0.5) plt.title(f'特征点检测 ({len(kp)} 个点)') plt.axis('off') plt.show() # 4. 绘制IMU数据 def plot_imu_data(imu_list): ""可视化IMU数据"" times = [data['timestamp'] for data in imu_list] gyro_x = [data['gyro'][0] for data in imu_list] accel_z = [data['accel'][2] for data in imu_list] plt.figure(figsize=(12, 6)) plt.subplot(2, 1, 1) plt.plot(times, gyro_x, 'b-') plt.xlabel('时间 (s)') plt.ylabel('角速度X (rad/s)') plt.title('陀螺仪数据') plt.grid(True) plt.subplot(2, 1, 2) plt.plot(times, accel_z, 'r-') plt.xlabel('时间 (s)') plt.ylabel('加速度Z (m/s²)') plt.title('加速度计数据') plt.grid(True) plt.tight_layout() plt.show() 
为什么用matplotlib：
标准的数据可视化库
支持2D/3D绘图
适合科学计算可视化
7. mpl_toolkits.mplot3d.Axes3D（3D绘图）
from mpl_toolkits.mplot3d import Axes3D 
作用：专门用于3D可视化，VIO的核心是3D运动
在VIO中的用途：
# 1. 绘制3D轨迹 def plot_3d_trajectory(trajectory): ""绘制3D轨迹图"" fig = plt.figure(figsize=(12, 10)) ax = fig.add_subplot(111, projection='3d') # 提取坐标 x = trajectory[:, 0] y = trajectory[:, 1] z = trajectory[:, 2] # 绘制轨迹线 ax.plot(x, y, z, 'b-', linewidth=2, label='轨迹') # 标记起点和终点 ax.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='起点') ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='^', label='终点') # 设置坐标轴 ax.set_xlabel('X (m)') ax.set_ylabel('Y (m)') ax.set_zlabel('Z (m)') ax.set_title('VIO估计的3D轨迹') ax.legend() ax.grid(True) # 设置视角 ax.view_init(elev=20, azim=45) plt.show() # 2. 绘制相机位姿 def plot_camera_poses(poses): ""绘制多个相机位姿"" fig = plt.figure(figsize=(12, 10)) ax = fig.add_subplot(111, projection='3d') for i, pose in enumerate(poses): # 提取位置 position = pose[:3, 3] # 提取旋转矩阵 R = pose[:3, :3] # 绘制相机位置 ax.scatter(position[0], position[1], position[2], c='b', s=20) # 绘制坐标系 axis_length = 0.1 x_axis = R[:, 0] * axis_length y_axis = R[:, 1] * axis_length z_axis = R[:, 2] * axis_length ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1, linewidth=1) ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1, linewidth=1) ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1, linewidth=1) ax.set_xlabel('X') ax.set_ylabel('Y') ax.set_zlabel('Z') ax.set_title('相机位姿') ax.grid(True) plt.show() # 3. 绘制点云 def plot_point_cloud(points_3d): ""绘制3D点云地图"" fig = plt.figure(figsize=(12, 10)) ax = fig.add_subplot(111, projection='3d') # 提取坐标 x = points_3d[:, 0] y = points_3d[:, 1] z = points_3d[:, 2] # 绘制点云 ax.scatter(x, y, z, c='b', s=1, alpha=0.5, marker='.') ax.set_xlabel('X (m)') ax.set_ylabel('Y (m)') ax.set_zlabel('Z (m)') ax.set_title('3D点云地图') ax.grid(True) plt.show() 
实际VIO系统中这些库如何协同工作
# 一个完整的VIO数据处理流程 class CompleteVIO: def __init__(self): # 1. 使用NumPy存储位姿和点云 self.trajectory = np.zeros((1000, 3)) # 存储轨迹 self.poses = [] # 存储位姿矩阵 self.point_cloud = np.zeros((10000, 3)) # 3D点云 # 2. 使用deque作为数据缓冲区 self.imu_buffer = deque(maxlen=1000) # IMU数据 self.image_buffer = deque(maxlen=30) # 图像数据 # 3. 使用OpenCV处理图像 self.orb = cv2.ORB_create(nfeatures=1000) self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # 4. 使用threading并行处理 self.running = True self.imu_thread = threading.Thread(target=self.imu_loop) self.vision_thread = threading.Thread(target=self.vision_loop) # 5. 使用time进行同步 self.last_time = time.time() self.start_time = time.time() # 6. 使用matplotlib进行可视化（可选） self.fig = None self.ax = None def imu_loop(self): ""IMU处理线程"" while self.running: # 获取当前时间 current_time = time.time() # 读取IMU数据 imu_data = self.read_imu_hardware() # 添加时间戳 imu_data['timestamp'] = current_time # 存储到缓冲区 self.imu_buffer.append(imu_data) # 控制频率 elapsed = time.time() - current_time if elapsed < 0.001: # 1kHz time.sleep(0.001 - elapsed) def vision_loop(self): ""视觉处理线程"" while self.running: # 获取图像 ret, frame = self.cap.read() if not ret: continue # 记录时间 image_time = time.time() # 使用OpenCV处理图像 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) kp, des = self.orb.detectAndCompute(gray, None) # 时间同步：找到最近的IMU数据 if self.imu_buffer: closest_imu = min(self.imu_buffer, key=lambda x: abs(x['timestamp'] - image_time)) # 融合处理 pose = self.fuse_vision_imu(kp, des, closest_imu) # 使用NumPy存储结果 self.trajectory = np.vstack([self.trajectory, pose[:3, 3]]) self.poses.append(pose) # 控制帧率 time.sleep(1/30) # 30Hz def visualize(self): ""可视化结果"" if self.fig is None: # 创建3D图形 self.fig = plt.figure(figsize=(10, 8)) self.ax = self.fig.add_subplot(111, projection='3d') plt.ion() # 开启交互模式 # 清空当前图形 self.ax.clear() # 绘制轨迹 x = self.trajectory[:, 0] y = self.trajectory[:, 1] z = self.trajectory[:, 2] self.ax.plot(x, y, z, 'b-', linewidth=2) # 设置图形属性 self.ax.set_xlabel('X (m)') self.ax.set_ylabel('Y (m)') self.ax.set_zlabel('Z (m)') self.ax.set_title(f'VIO轨迹 - 帧数: {len(self.trajectory)}') self.ax.grid(True) # 更新图形 plt.pause(0.01) def run(self): ""主运行函数"" # 启动线程 self.imu_thread.start() self.vision_thread.start() try: while self.running: # 定期可视化 if len(self.trajectory) % 10 == 0: self.visualize() time.sleep(0.1) except KeyboardInterrupt: self.running = False self.imu_thread.join() self.vision_thread.join() plt.ioff() plt.show() 
为什么需要这些库？
库作用如果没有会怎样NumPy数学计算引擎无法进行矩阵运算，VIO根本无法实现OpenCV视觉处理核心无法读取图像、检测特征、计算位姿time时间管理无法同步传感器，VIO会完全失效threading并行处理无法实时处理，会丢失IMU数据deque数据缓冲无法存储时间序列数据，同步困难matplotlib可视化无法调试和查看结果，开发困难 
总结
这6个导入是VIO系统的基础设施：
NumPy - 数学引擎（处理数字）
OpenCV - 视觉引擎（处理图像）
time - 时钟引擎（处理时间）
threading - 并发引擎（处理并行）
deque - 存储引擎（处理数据流）
matplotlib - 显示引擎（处理可视化）
它们共同构成了VIO系统的技术栈基础，缺少任何一个都会让VIO系统难以实现或效率低下。


class VIOSystem:
    ""
    完整的视觉惯性里程计系统
    演示如何获取和融合视觉-IMU数据
    ""
    
    def __init__(self, camera_matrix, dist_coeffs):
        ""
        初始化VIO系统
        ""
        # 1. 相机参数
        self.K = camera_matrix
        self.D = dist_coeffs
        
        # 2. 视觉模块初始化
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 3. 状态变量
        self.current_pose = np.eye(4)  # 当前位姿（4x4变换矩阵）
        self.velocity = np.zeros(3)    # 当前速度
        self.bias_gyro = np.zeros(3)   # 陀螺仪零偏
        self.bias_acc = np.zeros(3)    # 加速度计零偏
        
        # 4. 数据缓冲区
        self.imu_buffer = deque(maxlen=100)  # IMU数据缓冲区
        self.image_buffer = deque(maxlen=10)  # 图像数据缓冲区
        
        # 5. 时间管理
        self.last_imu_time = None
        self.last_image_time = None
        
        # 6. 关键帧管理
        self.keyframes = []
        self.last_kp = None
        self.last_des = None
        
        # 7. 线程同步
        self.lock = threading.Lock()
        
        print("VIO系统初始化完成")

    # ==========================================================
    # 1. 数据获取模块
    # ==========================================================
    
    def get_imu_data_simulation(self):
        ""
        模拟获取IMU数据
        实际应用中，这里应该从硬件IMU读取
        ""
        # 模拟IMU数据：包含时间戳、角速度、加速度
        current_time = time.time()
        
        # 模拟运动：缓慢绕Y轴旋转，轻微振动
        imu_data = {
            'timestamp': current_time,
            'gyro': np.array([0.0, 0.1, 0.0]) + np.random.randn(3) * 0.01,  # 角速度 (rad/s)
            'accel': np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1,  # 加速度 (m/s²)
            'temperature': 25.0
        }
        
        return imu_data
    
    def get_camera_data_simulation(self, width=640, height=480):
        ""
        模拟获取相机数据
        实际应用中，这里应该从摄像头读取
        ""
        # 创建一个模拟的棋盘格图像
        img = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # 添加棋盘格纹理
        for i in range(0, height, 40):
            for j in range(0, width, 40):
                if (i//40 + j//40) % 2 == 0:
                    img[i:i+40, j:j+40] = 200
                else:
                    img[i:i+40, j:j+40] = 60
        
        # 添加一些特征点
        cv2.circle(img, (width//2, height//2), 5, (0, 255, 0), -1)
        cv2.circle(img, (width//4, height//4), 5, (0, 0, 255), -1)
        cv2.circle(img, (width*3//4, height*3//4), 5, (255, 0, 0), -1)
        
        camera_data = {
            'timestamp': time.time(),
            'image': img,
            'width': width,
            'height': height
        }
        
        return camera_data
    
    def read_imu_from_file(self, filepath):
        ""
        从文件读取IMU数据
        实际应用中，IMU数据可能以CSV或二进制格式存储
        ""
        imu_data_list = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        imu_data = {
                            'timestamp': float(parts[0]),
                            'gyro': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                            'accel': np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                        }
                        imu_data_list.append(imu_data)
        except FileNotFoundError:
            print(f"警告：IMU文件 {filepath} 不存在，使用模拟数据")
        
        return imu_data_list
    
    def read_images_from_folder(self, folder_path):
        ""
        从文件夹读取图像序列
        实际应用中，相机数据可能是视频流或图像序列
        ""
        import os
        image_files = []
        
        if os.path.exists(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(folder_path, filename))
        
        return image_files
    
    # ==========================================================
    # 2. 数据处理模块
    # ==========================================================
    
    def preprocess_imu(self, imu_data):
        ""
        IMU数据预处理
        包括：去除零偏、尺度校正、温度补偿等
        ""
        # 1. 去除零偏
        gyro_corrected = imu_data['gyro'] - self.bias_gyro
        accel_corrected = imu_data['accel'] - self.bias_acc
        
        # 2. 尺度校正（假设已知尺度矩阵）
        # 实际应用中需要通过标定得到
        gyro_scale = np.eye(3)
        accel_scale = np.eye(3)
        
        gyro_corrected = np.dot(gyro_scale, gyro_corrected)
        accel_corrected = np.dot(accel_scale, accel_corrected)
        
        # 3. 去除重力（需要知道当前姿态）
        # 这里简化处理，实际需要根据姿态旋转重力向量
        
        processed_data = {
            'timestamp': imu_data['timestamp'],
            'gyro': gyro_corrected,
            'accel': accel_corrected
        }
        
        return processed_data
    
    def preprocess_image(self, image_data):
        ""
        图像数据预处理
        包括：去畸变、灰度化、直方图均衡等
        ""
        image = image_data['image']
        
        # 1. 去畸变
        if self.D is not None and np.any(self.D != 0):
            image = cv2.undistort(image, self.K, self.D)
        
        # 2. 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 3. 直方图均衡（增强对比度）
        gray_eq = cv2.equalizeHist(gray)
        
        processed_data = {
            'timestamp': image_data['timestamp'],
            'image': gray_eq,
            'original': image
        }
        
        return processed_data
    
    def extract_visual_features(self, image):
        ""
        提取视觉特征
        返回特征点和描述子
        ""
        # 使用ORB提取特征
        kp, des = self.orb.detectAndCompute(image, None)
        
        if kp is None or des is None:
            return None, None
        
        return kp, des
    
    def match_features(self, des1, des2):
        ""
        特征匹配
        返回匹配点对
        ""
        if des1 is None or des2 is None:
            return []
        
        matches = self.bf.match(des1, des2)
        
        # 按距离排序，取前N个最佳匹配
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    # ==========================================================
    # 3. 数据融合模块（核心）
    # ==========================================================
    
    def integrate_imu(self, imu_data, dt):
        ""
        IMU积分：从角速度和加速度推算位姿变化
        这是VIO的核心算法之一
        
        参数:
        imu_data: 包含角速度和加速度的字典
        dt: 时间间隔
        
        返回:
        delta_pose: 4x4变换矩阵，表示位姿变化
        ""
        gyro = imu_data['gyro']
        accel = imu_data['accel']
        
        # 1. 旋转积分
        # 使用指数映射将角速度积分转换为旋转矩阵
        # 简化：假设旋转很小，使用一阶近似
        theta = np.linalg.norm(gyro) * dt
        if theta < 1e-6:
            # 旋转角度太小，近似为单位矩阵
            delta_R = np.eye(3)
        else:
            # 旋转轴
            axis = gyro / np.linalg.norm(gyro)
            
            # 使用罗德里格斯公式计算旋转矩阵
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            delta_R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
        # 2. 位置积分
        # 需要双重积分加速度
        # 注意：加速度测量包含重力，需要去除
        
        # 获取当前旋转矩阵
        R = self.current_pose[:3, :3]
        
        # 将加速度从机体坐标系转换到世界坐标系
        accel_world = np.dot(R, accel)
        
        # 去除重力（假设Z轴向上）
        gravity = np.array([0, 0, 9.81])
        accel_world = accel_world - gravity
        
        # 积分得到速度变化
        delta_v = accel_world * dt
        
        # 积分得到位置变化
        delta_p = self.velocity * dt + 0.5 * accel_world * dt**2
        
        # 3. 构建变换矩阵
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = delta_R
        delta_pose[:3, 3] = delta_p
        
        return delta_pose, delta_v
    
    def estimate_pose_from_vision(self, kp1, kp2, matches):
        ""
        从视觉特征估计位姿
        使用对极几何或PnP算法
        ""
        if len(matches) < 8:
            return None
        
        # 提取匹配点
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # 使用对极几何估计本质矩阵
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            return None
        
        # 从本质矩阵恢复旋转和平移
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # 构建变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        return pose
    
    def fuse_vision_imu_loose(self, visual_pose, imu_pose, visual_confidence=0.7):
        ""
        松耦合融合：分别计算视觉和IMU位姿，然后加权平均
        
        参数:
        visual_pose: 视觉估计的位姿
        imu_pose: IMU估计的位姿
        visual_confidence: 视觉置信度（0-1）
        
        返回:
        fused_pose: 融合后的位姿
        ""
        if visual_pose is None:
            # 视觉失效，完全信任IMU
            return imu_pose
        
        # 将位姿转换为旋转向量和平移向量
        def pose_to_rt(pose):
            R = pose[:3, :3]
            t = pose[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            return rvec.flatten(), t
        
        def rt_to_pose(rvec, t):
            pose = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            pose[:3, :3] = R
            pose[:3, 3] = t
            return pose
        
        # 转换
        rvec_v, t_v = pose_to_rt(visual_pose)
        rvec_i, t_i = pose_to_rt(imu_pose)
        
        # 加权融合
        alpha = visual_confidence
        rvec_fused = alpha * rvec_v + (1 - alpha) * rvec_i
        t_fused = alpha * t_v + (1 - alpha) * t_i
        
        # 转换回变换矩阵
        fused_pose = rt_to_pose(rvec_fused, t_fused)
        
        return fused_pose
    
    def fuse_vision_imu_tight(self, kp, des, imu_data, prev_pose):
        ""
        紧耦合融合：在优化框架中同时优化视觉和IMU约束
        
        这是一个简化的紧耦合示例
        实际应用中会使用非线性优化（如g2o、Ceres）
        ""
        # 这里简化处理，实际紧耦合更复杂
        # 需要构建包含视觉重投影误差和IMU预积分误差的代价函数
        
        # 1. 视觉约束
        if self.last_kp is not None and self.last_des is not None:
            matches = self.match_features(self.last_des, des)
            if len(matches) > 8:
                visual_pose = self.estimate_pose_from_vision(self.last_kp, kp, matches)
            else:
                visual_pose = None
        else:
            visual_pose = None
        
        # 2. IMU预测
        dt = 0.01  # 假设时间间隔
        imu_pose, delta_v = self.integrate_imu(imu_data, dt)
        
        # 3. 简单融合
        if visual_pose is not None:
            # 如果有视觉测量，进行融合
            fused_pose = self.fuse_vision_imu_loose(visual_pose, np.dot(prev_pose, imu_pose))
        else:
            # 否则使用IMU预测
            fused_pose = np.dot(prev_pose, imu_pose)
        
        return fused_pose
    
    # ==========================================================
    # 4. 主处理循环
    # ==========================================================
    
    def process_single_frame(self, image_data, imu_data_list):
        ""
        处理单帧图像和对应的IMU数据
        ""
        # 1. 图像预处理
        processed_image = self.preprocess_image(image_data)
        image = processed_image['image']
        
        # 2. 提取视觉特征
        kp, des = self.extract_visual_features(image)
        
        if kp is None or des is None:
            print("警告：无法提取特征点")
            return self.current_pose
        
        # 3. IMU预积分
        # 处理从上一帧到当前帧的所有IMU数据
        imu_pose = np.eye(4)
        for imu_data in imu_data_list:
            processed_imu = self.preprocess_imu(imu_data)
            dt = 0.01  # 假设固定时间间隔
            delta_pose, delta_v = self.integrate_imu(processed_imu, dt)
            imu_pose = np.dot(imu_pose, delta_pose)
            self.velocity += delta_v
        
        # 4. 视觉位姿估计
        visual_pose = None
        if self.last_kp is not None and self.last_des is not None:
            matches = self.match_features(self.last_des, des)
            if len(matches) > 8:
                visual_pose = self.estimate_pose_from_vision(self.last_kp, kp, matches)
        
        # 5. 传感器融合
        if visual_pose is not None:
            # 有视觉测量，进行松耦合融合
            fused_pose = self.fuse_vision_imu_loose(visual_pose, np.dot(self.current_pose, imu_pose))
        else:
            # 无视觉测量，只使用IMU
            fused_pose = np.dot(self.current_pose, imu_pose)
        
        # 6. 更新状态
        self.current_pose = fused_pose
        
        # 7. 保存当前特征
        self.last_kp = kp
        self.last_des = des
        
        return fused_pose
    
    def run_simulation(self, num_frames=100):
        ""
        运行模拟：生成模拟数据并处理
        ""
        trajectory = []
        
        print("开始VIO模拟运行...")
        
        for i in range(num_frames):
            # 生成模拟数据
            image_data = self.get_camera_data_simulation()
            imu_data_list = [self.get_imu_data_simulation() for _ in range(10)]  # 模拟10个IMU数据
            
            # 处理当前帧
            pose = self.process_single_frame(image_data, imu_data_list)
            
            # 记录轨迹
            position = pose[:3, 3]
            trajectory.append(position)
            
            # 显示进度
            if i % 10 == 0:
                print(f"处理第 {i}/{num_frames} 帧，位置: {position}")
        
        return np.array(trajectory)
    
    def visualize_trajectory(self, trajectory):
        ""
        可视化轨迹
        ""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取坐标
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        
        # 绘制轨迹
        ax.plot(x, y, z, 'b-', linewidth=2, label='VIO轨迹')
        ax.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='起点')
        ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='^', label='终点')
        
        # 添加坐标轴
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.5, arrow_length_ratio=0.1, label='X轴')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.5, arrow_length_ratio=0.1, label='Y轴')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.5, arrow_length_ratio=0.1, label='Z轴')
        
        # 设置图形属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('VIO估计的相机轨迹')
        ax.legend()
        ax.grid(True)
        
        plt.show()

# 主程序
if __name__ == "__main__":
    # 相机参数（示例值）
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    # 创建VIO系统
    vio = VIOSystem(camera_matrix, dist_coeffs)
    
    # 运行模拟
    trajectory = vio.run_simulation(num_frames=50)
    
    # 可视化结果
    vio.visualize_trajectory(trajectory)
    
    print("VIO模拟完成")






2. 数据获取的详细解释

2.1 IMU数据获取

硬件接口：
# 实际硬件读取示例
def read_imu_from_serial(self, serial_port='/dev/ttyACM0', baudrate=115200):
    ""
    从串口读取IMU数据
    实际硬件：MPU6050, BMI160, ICM-20948等
    ""
    import serial
    
    ser = serial.Serial(serial_port, baudrate, timeout=1)
    
    while True:
        data = ser.readline().decode('utf-8').strip()
        if data:
            # 解析数据
            parts = data.split(',')
            if len(parts) >= 7:
                imu_data = {
                    'timestamp': time.time(),
                    'accel_x': float(parts[0]),
                    'accel_y': float(parts[1]),
                    'accel_z': float(parts[2]),
                    'gyro_x': float(parts[3]),
                    'gyro_y': float(parts[4]),
                    'gyro_z': float(parts[5]),
                    'temp': float(parts[6])
                }
                return imu_data

                



ROS中的IMU获取：

# ROS节点示例
def imu_callback(self, msg):
    ""
    ROS IMU消息回调
    ""
    imu_data = {
        'timestamp': msg.header.stamp.to_sec(),
        'gyro': np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z]),
        'accel': np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z]),
        'orientation': np.array([msg.orientation.x,
                                msg.orientation.y,
                                msg.orientation.z,
                                msg.orientation.w])
    }
    self.imu_buffer.append(imu_data)


    



2.2 相机数据获取

OpenCV摄像头读取：


def read_camera_from_usb(self, camera_id=0):
    ""
    从USB摄像头读取图像
    ""
    cap = cv2.VideoCapture(camera_id)
    
    # 设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        ret, frame = cap.read()
        if ret:
            camera_data = {
                'timestamp': time.time(),
                'image': frame
            }
            return camera_data


ROS图像获取：


def image_callback(self, msg):
    ""
    ROS图像消息回调
    ""
    # 将ROS图像转换为OpenCV格式
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    
    camera_data = {
        'timestamp': msg.header.stamp.to_sec(),
        'image': cv_image
    }
    self.image_buffer.append(camera_data)


    

3. 数据融合的详细数学原理

3.1 IMU预积分

核心公式：
位置：p_bk+1^w = p_bk^w + v_bk^w Δt + ∫∫[R_t^w(a_t - b_a^t) - g^w]dt²
速度：v_bk+1^w = v_bk^w + ∫[R_t^w(a_t - b_a^t) - g^w]dt
旋转：q_bk+1^w = q_bk^w ⊗ ∫ exp((ω_t - b_g^t)dt)




代码实现：
class IMUPreintegration:
    def __init__(self):
        self.delta_p = np.zeros(3)  # 位置变化
        self.delta_v = np.zeros(3)  # 速度变化
        self.delta_q = np.array([1, 0, 0, 0])  # 旋转变化（四元数）
        self.delta_t = 0.0
        
    def integrate(self, acc, gyro, dt):
        ""
        预积分一个IMU测量
        ""
        # 去除零偏
        acc_no_bias = acc - self.bias_acc
        gyro_no_bias = gyro - self.bias_gyro
        
        # 旋转变化
        delta_angle = gyro_no_bias * dt
        delta_q = self.angle_axis_to_quaternion(delta_angle)
        self.delta_q = self.quaternion_multiply(self.delta_q, delta_q)
        
        # 速度变化
        R = self.quaternion_to_matrix(self.delta_q)
        acc_world = np.dot(R, acc_no_bias)
        self.delta_v += (acc_world - self.gravity) * dt
        
        # 位置变化
        self.delta_p += self.delta_v * dt + 0.5 * (acc_world - self.gravity) * dt**2
        self.delta_t += dt

        



3.2 紧耦合优化的代价函数

视觉重投影误差：

e_visual = Σ ||π(T_cw * X_i) - u_i||²
其中：
- π: 投影函数
- T_cw: 相机到世界的变换
- X_i: 3D地图点
- u_i: 2D观测



IMU预积分误差：
e_imu = [e_R, e_v, e_p]^T
e_R = Log((ΔR_bk_bk+1)^T * (R_w_bk)^T * R_w_bk+1)
e_v = R_w_bk^T * (v_bk+1^w - v_bk^w - g^w Δt) - Δv_bk_bk+1
e_p = R_w_bk^T * (p_bk+1^w - p_bk^w - v_bk^w Δt - 0.5 g^w Δt²) - Δp_bk_bk+1



总代价函数：
E_total = Σ ρ(e_visual^T Σ_visual^{-1} e_visual) + Σ e_imu^T Σ_imu^{-1} e_imu





3.3 卡尔曼滤波融合

预测步骤：

def kalman_predict(self, imu_data, dt):
    ""
    卡尔曼滤波预测步骤
    ""
    # 状态向量: [位置, 速度, 旋转, 陀螺零偏, 加计零偏]
    
    # 状态转移矩阵
    F = self.compute_F(imu_data, dt)
    
    # 过程噪声
    Q = self.compute_Q(dt)
    
    # 状态预测
    self.x = F @ self.x
    
    # 协方差预测
    self.P = F @ self.P @ F.T + Q





更新步骤：

def kalman_update(self, visual_measurement):
    ""
    卡尔曼滤波更新步骤
    ""
    # 观测矩阵
    H = self.compute_H()
    
    # 观测噪声
    R = self.compute_R()
    
    # 卡尔曼增益
    S = H @ self.P @ H.T + R
    K = self.P @ H.T @ np.linalg.inv(S)
    
    # 状态更新
    y = visual_measurement - H @ self.x
    self.x = self.x + K @ y
    
    # 协方差更新
    I = np.eye(len(self.x))
    self.P = (I - K @ H) @ self.P






4. 实际数据集处理示例

4.1 EuRoC MAV数据集

def load_euroc_dataset(self, dataset_path):
    ""
    加载EuRoC MAV数据集
    ""
    import pandas as pd
    
    # 加载IMU数据
    imu_file = os.path.join(dataset_path, "mav0", "imu0", "data.csv")
    imu_df = pd.read_csv(imu_file)
    
    imu_data_list = []
    for _, row in imu_df.iterrows():
        imu_data = {
            'timestamp': row['timestamp'] * 1e-9,  # 纳秒转秒
            'gyro': np.array([row['w.x'], row['w.y'], row['w.z']]),
            'accel': np.array([row['a.x'], row['a.y'], row['a.z']])
        }
        imu_data_list.append(imu_data)
    
    # 加载图像数据
    image_folder = os.path.join(dataset_path, "mav0", "cam0", "data")
    image_files = sorted(os.listdir(image_folder))
    
    image_data_list = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        
        # 从文件名提取时间戳
        timestamp = float(img_file.replace('.png', '')) * 1e-9
        
        image_data = {
            'timestamp': timestamp,
            'image': img
        }
        image_data_list.append(image_data)
    
    return imu_data_list, image_data_list




    

5. 完整的实时VIO系统
class RealTimeVIO:
    ""
    实时VIO系统
    使用多线程处理视觉和IMU数据
    ""
    
    def __init__(self):
        # 初始化
        self.imu_thread = threading.Thread(target=self.imu_loop)
        self.vision_thread = threading.Thread(target=self.vision_loop)
        self.fusion_thread = threading.Thread(target=self.fusion_loop)
        
        # 数据队列
        self.imu_queue = queue.Queue(maxsize=1000)
        self.vision_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # 状态
        self.running = True
        
    def imu_loop(self):
        ""IMU数据采集线程""
        while self.running:
            imu_data = self.read_imu_hardware()
            timestamp = time.time()
            
            # 时间同步
            synchronized_data = self.synchronize_imu(timestamp, imu_data)
            
            # 放入队列
            self.imu_queue.put(synchronized_data)
            
            # 控制频率
            time.sleep(0.001)  # 1kHz
    
    def vision_loop(self):
        ""视觉处理线程""
        while self.running:
            # 获取图像
            image_data = self.read_camera_hardware()
            
            # 特征提取
            kp, des = self.extract_features(image_data['image'])
            
            # 时间戳对齐
            aligned_imu = self.get_aligned_imu(image_data['timestamp'])
            
            # 放入队列
            self.vision_queue.put({
                'timestamp': image_data['timestamp'],
                'kp': kp,
                'des': des,
                'imu': aligned_imu
            })
            
            # 控制频率
            time.sleep(0.033)  # 30Hz
    
    def fusion_loop(self):
        ""融合线程""
        while self.running:
            try:
                # 从队列获取数据
                vision_data = self.vision_queue.get(timeout=0.1)
                
                # 紧耦合优化
                pose = self.tightly_coupled_optimization(
                    vision_data['kp'],
                    vision_data['des'],
                    vision_data['imu']
                )
                
                # 发布结果
                self.result_queue.put({
                    'timestamp': vision_data['timestamp'],
                    'pose': pose
                })
                
            except queue.Empty:
                continue
    
    def start(self):
        ""启动VIO系统""
        self.imu_thread.start()
        self.vision_thread.start()
        self.fusion_thread.start()
        
        print("VIO系统已启动")
    
    def stop(self):
        ""停止VIO系统""
        self.running = False
        self.imu_thread.join()
        self.vision_thread.join()
        self.fusion_thread.join()
        print("VIO系统已停止")

        



总结

VIO获取和融合视觉-IMU数据的过程：
数据获取：
IMU：从硬件接口读取角速度和加速度
相机：从摄像头读取图像帧
关键：时间同步和标定
数据处理：
IMU：去除零偏、尺度校正、预积分
视觉：特征提取、匹配、位姿估计
数据融合：
松耦合：分别计算，后融合（卡尔曼滤波）
紧耦合：联合优化（非线性最小二乘）
实时性保证：
多线程处理
数据缓冲区
异步优化
实际应用中，VIO系统需要精心设计的时间同步、传感器标定和异常处理机制，才能在各种环境下稳定工作。
"""



















"""
状态机驱动的SLAM（同步定位与建图）系统代码详细注释
状态机设计详细解析
1. 状态机模式的基本概念
状态机（State Machine） 是一种行为设计模式，它允许对象在内部状态改变时改变其行为。状态机模式将状态和行为封装在不同的类中，使得状态转换更加清晰。
在SLAM中的应用：
SLAM系统有明确的状态： 1. 初始化：刚开始运行，寻找特征 2. 追踪：正常工作，定位和建图 3. 丢失：视觉中断，使用IMU 4. 恢复：尝试找回跟踪 每个状态有不同的行为： - 初始化状态：检测特征，建立地图 - 追踪状态：定位，建图，保存关键帧 - 丢失状态：IMU预测，计时 - 恢复状态：全局匹配，重定位 
2. 状态转换逻辑
2.1 完整的状态转换图
INITIALIZING → TRACKING → LOST → RECOVERING → TRACKING ↓ ↓ 失败保持 超时→INITIALIZING ↓ ↓ INITIALIZING RECOVERING→LOST 
2.2 状态转换触发条件
# 1. INITIALIZING → TRACKING # 条件：初始化成功，找到足够特征 if _try_init(frame)返回True: self.state = SystemState.TRACKING # 2. TRACKING → LOST # 条件：视觉跟踪失败 if not _visual_pnp_track(frame)返回True: self.state = SystemState.LOST self.lost_start_time = time.time() # 3. LOST → RECOVERING # 条件：未超时 if 未超时: self.state = SystemState.RECOVERING # 4. LOST → INITIALIZING # 条件：超时 if 超时: self.state = SystemState.INITIALIZING # 5. RECOVERING → TRACKING # 条件：重定位成功 if _try_relocalization(frame)返回True: self.state = SystemState.TRACKING # 6. RECOVERING → LOST # 条件：重定位失败 if _try_relocalization(frame)返回False: self.state = SystemState.LOST 
3. 每个状态的详细行为
3.1 初始化状态（INITIALIZING）
目标：建立系统初始状态
关键步骤：
def _handle_initializing(self, frame): # 1. 特征检测 kp, des = detect_features(frame) # 2. 质量检查 if len(kp) < MIN_FEATURES: return False # 特征不足 # 3. 初始化地图 # 选择第一帧作为关键帧 # 建立初始坐标系 # 初始化地图点 # 4. 切换到追踪状态 return True 
注意事项：
需要足够数量和分布的特征点
可能需要多帧初始化（如ORB-SLAM使用多帧）
初始化失败时，可以调整参数重新尝试
3.2 追踪状态（TRACKING）
目标：持续定位和建图
关键步骤：
def _handle_tracking(self, frame): # 1. 局部追踪 # 与上一帧或参考关键帧匹配 matches = match_features(prev_frame, frame) # 2. 位姿估计 # 使用PnP估计相机位姿 rvec, tvec = solvePnP(object_points, image_points) # 3. 融合IMU # 使用卡尔曼滤波融合视觉和IMU fused_pose = kalman_filter(rvec, tvec, imu_data) # 4. 关键帧决策 if need_keyframe(fused_pose, matches): save_keyframe(frame, fused_pose) # 5. 状态检查 if tracking_quality < THRESHOLD: return False # 跟踪失败 return True 
质量检查指标：
def check_tracking_quality(self, matches, reprojection_error): ""检查跟踪质量"" quality_score = 1.0 # 1. 匹配点数量 if len(matches) < 20: quality_score *= 0.3 elif len(matches) < 50: quality_score *= 0.7 else: quality_score *= 1.0 # 2. 重投影误差 if reprojection_error > 5.0: quality_score *= 0.5 elif reprojection_error > 2.0: quality_score *= 0.8 else: quality_score *= 1.0 # 3. 特征点分布 distribution = check_feature_distribution() quality_score *= distribution return quality_score 
3.3 丢失状态（LOST）
目标：应急处理，保持系统运行
关键步骤：
def _handle_lost(self, frame): # 1. 完全忽略视觉输入 # 视觉数据可能完全错误，直接忽略 # 2. 纯IMU预测 # 获取IMU数据 gyro = get_gyro_data() # 角速度 accel = get_accel_data() # 加速度 # 积分得到位姿变化 delta_pose = integrate_imu(gyro, accel, dt) # 3. 更新位姿 current_pose = last_pose * delta_pose # 4. 时间检查 lost_time = time.time() - self.lost_start_time if lost_time > self.max_lost_duration: return "RESET" # 需要重置 return "CONTINUE" 
IMU预测的局限性：
IMU只能提供相对运动估计： 1. 旋转：陀螺仪积分，有累积误差 2. 平移：加速度计双重积分，误差更大 3. 没有绝对位置信息 4. 只能短时使用（几秒钟） 因此，丢失状态只能是临时应急 
3.4 恢复状态（RECOVERING）
目标：找回跟踪，重新定位
关键步骤：
def _handle_recovering(self, frame): # 1. 提取当前帧特征 kp, des = extract_features(frame) # 2. 与历史关键帧匹配 # 使用词袋模型加速 best_match = None best_score = 0 for keyframe in self.keyframes: # 计算相似度 score = compute_similarity(des, keyframe.des) if score > best_score and score > THRESHOLD: best_score = score best_match = keyframe # 3. 如果找到匹配，计算相对位姿 if best_match: # 特征匹配 matches = match_features(des, best_match.des) # 计算位姿 rvec, tvec = compute_relative_pose(matches) # 更新系统位姿 self.current_pose = combine_pose(best_match.pose, (rvec, tvec)) return True return False 
重定位技术：
1. 直接匹配：与所有关键帧匹配，计算量大 2. 词袋模型：将图像转换为视觉单词，快速检索 3. 位置识别：使用深度学习的全局特征 4. 几何验证：通过RANSAC验证匹配的正确性 
4. 状态机的扩展与优化
4.1 添加更多状态
class ExtendedSystemState(enum.Enum): ""扩展的系统状态"" BOOTING = 0 # 系统启动，加载配置 INITIALIZING = 1 # 初始化 TRACKING = 2 # 正常追踪 MAPPING = 3 # 建图模式（更关注建图） LOST = 4 # 丢失 RECOVERING = 5 # 恢复 CLOSING_LOOP = 6 # 闭环检测中 OPTIMIZING = 7 # 位姿图优化中 PAUSED = 8 # 暂停状态 SHUTTING_DOWN = 9 # 关闭中 
4.2 状态持久化和恢复
class StatePersistentSLAM(SimpleSLAMSystem): ""支持状态持久化的SLAM系统"" def save_state(self, filepath): ""保存系统状态到文件"" state_data = { 'system_state': self.state.name, 'lost_start_time': self.lost_start_time, 'current_pose': self.current_pose.tolist(), 'keyframes': [kf.serialize() for kf in self.keyframes] } import json with open(filepath, 'w') as f: json.dump(state_data, f) def load_state(self, filepath): ""从文件加载系统状态"" import json with open(filepath, 'r') as f: state_data = json.load(f) self.state = SystemState[state_data['system_state']] self.lost_start_time = state_data['lost_start_time'] self.current_pose = np.array(state_data['current_pose']) # 重新创建关键帧对象 self.keyframes = [] for kf_data in state_data['keyframes']: kf = Keyframe.deserialize(kf_data) self.keyframes.append(kf) 
4.3 状态转换监控
class MonitoredSLAM(SimpleSLAMSystem): ""带有监控的状态机"" def __init__(self): super().__init__() # 状态转换历史 self.state_history = [] # 状态统计 self.state_stats = {state.name: 0 for state in SystemState} # 转换监听器 self.transition_listeners = [] def set_state(self, new_state): ""设置新状态，记录转换历史"" old_state = self.state # 调用父类方法 super().set_state(new_state) # 记录状态转换 transition = { 'timestamp': time.time(), 'from': old_state.name, 'to': new_state.name } self.state_history.append(transition) # 更新统计 self.state_stats[new_state.name] += 1 # 通知监听器 for listener in self.transition_listeners: listener(transition) def get_state_duration(self, state): ""计算某个状态的持续时间"" durations = [] start_time = None for transition in self.state_history: if start_time is None and transition['to'] == state.name: start_time = transition['timestamp'] elif start_time is not None and transition['from'] == state.name: duration = transition['timestamp'] - start_time durations.append(duration) start_time = None if start_time is not None: # 当前还在这个状态 durations.append(time.time() - start_time) return durations 
5. 实际应用示例
5.1 完整的主循环
def main(): ""主程序：运行SLAM系统"" # 创建SLAM系统 slam = SimpleSLAMSystem() # 打开摄像头 cap = cv2.VideoCapture(0) # 帧计数器 frame_count = 0 print("=== SLAM系统启动 ===") print("按 'q' 键退出") print("==================") while True: # 读取一帧 ret, frame = cap.read() if not ret: print("摄像头读取失败") break # 处理帧 slam.main_loop(frame) # 显示当前状态 display_frame = frame.copy() cv2.putText(display_frame, f"状态: {slam.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 显示帧 cv2.imshow("SLAM System", display_frame) # 检查退出键 if cv2.waitKey(1) & 0xFF == ord('q'): break frame_count += 1 # 释放资源 cap.release() cv2.destroyAllWindows() # 输出统计 print(f"\n=== 运行统计 ===") print(f"总帧数: {frame_count}") print(f"最终状态: {slam.state.name}") if __name__ == "__main__": main() 
5.2 与AR系统集成
class ARWithStateMachine: ""结合状态机的AR系统"" def __init__(self): self.slam = SimpleSLAMSystem() self.ar_objects = [] self.renderer = ARRenderer() def process_frame(self, frame): ""处理每一帧"" # 运行SLAM self.slam.main_loop(frame) # 根据状态决定AR渲染策略 if self.slam.state == SystemState.TRACKING: # 正常状态：渲染所有AR内容 frame = self.render_normal_ar(frame) elif self.slam.state == SystemState.LOST: # 丢失状态：简化渲染，显示警告 frame = self.render_lost_warning(frame) elif self.slam.state == SystemState.RECOVERING: # 恢复状态：简化渲染，显示恢复提示 frame = self.render_recovering_hint(frame) elif self.slam.state == SystemState.INITIALIZING: # 初始化状态：显示初始化提示 frame = self.render_initializing_hint(frame) return frame def render_normal_ar(self, frame): ""正常状态下的AR渲染"" # 获取当前位姿 rvec, tvec = self.slam.get_current_pose() # 渲染所有AR物体 for obj in self.ar_objects: frame = self.renderer.render_object(frame, obj, rvec, tvec) return frame def render_lost_warning(self, frame): ""丢失状态下的警告渲染"" cv2.putText(frame, "视觉丢失，使用IMU预测", (frame.shape[1]//2 - 150, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) return frame 
这个基于状态机的SLAM系统设计清晰地划分了系统的不同工作状态，每个状态有明确的行为和转换条件。这种设计使得系统更加健壮，能够优雅地处理各种异常情况，如视觉丢失、初始化失败等。在实际应用中，你可以根据需要填充各个状态下的具体算法实现。
"""