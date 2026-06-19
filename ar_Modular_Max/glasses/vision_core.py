# 导入必要的Python库
import cv2    # OpenCV库，用于计算机视觉处理
import numpy as np  # NumPy库，用于数值计算

class VisionCore:
    """
    视觉核心类，负责视觉特征提取和运动估计
    
    这个类实现了纯视觉里程计（Visual Odometry）的核心功能：
    1. 特征点检测（ORB算法）
    2. 特征点匹配
    3. 运动估计
    
    主要用于在IMU不可用或需要验证时提供视觉运动信息
    """
    
    def __init__(self):
        """
        初始化视觉核心
        
        创建ORB特征提取器和特征匹配器
        在真实VIO系统中，这里可能还需要初始化相机内参、畸变参数等
        """
        # 初始化OpenCV的ORB特征提取器
        # ORB（Oriented FAST and Rotated BRIEF）是一种快速的特征检测和描述子算法
        # 参数说明：
        #   nfeatures=500: 最多检测500个特征点
        # 其他可选参数（这里使用默认值）：
        #   scaleFactor=1.2: 金字塔缩放因子
        #   nlevels=8: 金字塔层数
        #   edgeThreshold=31: 边缘阈值
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # 初始化暴力匹配器（Brute-Force Matcher）
        # cv2.BFMatcher: 暴力匹配器，比较所有特征描述子
        # 参数说明：
        #   cv2.NORM_HAMMING: 使用汉明距离作为相似性度量（适用于二进制描述子如ORB）
        #   crossCheck=True: 交叉检查，确保匹配是双向的
        #                     即特征点A匹配到B，B也匹配到A
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 存储上一帧的描述子和关键点
        # 用于与当前帧进行特征匹配，计算帧间运动
        self.last_descriptor = None
        self.last_keypoints = None
        
        # 打印初始化信息
        print("[Vision] VisionCore 已启动，开启纯视觉模式...")

    def process_frame(self, frame):
        """
        核心任务：提取特征、匹配追踪，返回视觉观测数据
        
        参数:
        frame: 输入的图像帧（BGR格式）
        
        返回:
        字典，包含：
        - keypoints: 当前帧检测到的关键点
        - motion_trend: 运动趋势（如果可计算）
        - frame_timestamp: 帧时间戳（用于时序分析）
        
        这个函数实现了视觉里程计的基本流程：
        1. 图像预处理
        2. 特征提取
        3. 特征匹配
        4. 运动估计
        5. 状态更新
        """
        # 1. 图像预处理（灰度化以加速计算）
        # 将BGR图像转换为灰度图像
        # 大多数特征提取算法在灰度图像上工作
        # 这可以减少计算量，因为只有一个通道
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 特征提取
        # 使用ORB检测关键点和计算描述子
        # 关键点（keypoints）: 包含位置、方向、尺度等信息
        # 描述子（descriptors）: 特征点的二进制描述向量，用于匹配
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # 3. 特征匹配（对比当前帧与上一帧，实现视觉运动追踪）
        # 初始化运动数据
        motion_data = None
        
        # 检查是否有上一帧的描述子和当前帧的描述子
        if self.last_descriptor is not None and descriptors is not None:
            # 使用暴力匹配器匹配两帧之间的特征点
            # matches: 匹配结果列表，每个元素是一个DMatch对象
            # DMatch对象包含：
            #   queryIdx: 查询图像（上一帧）中特征点的索引
            #   trainIdx: 训练图像（当前帧）中特征点的索引
            #   distance: 两个描述子之间的距离（越小越相似）
            matches = self.matcher.match(self.last_descriptor, descriptors)
            
            # 简化逻辑：根据匹配点的位移量估算运动趋势
            # 如果有足够的匹配点，计算运动趋势
            motion_data = self._estimate_motion(self.last_keypoints, keypoints, matches)

        # 4. 更新状态
        # 将当前帧的描述子和关键点保存，用于下一帧匹配
        self.last_descriptor = descriptors
        self.last_keypoints = keypoints
        
        # 返回处理结果
        return {
            "keypoints": keypoints,  # 当前帧的关键点
            "motion_trend": motion_data,  # 运动趋势
            "frame_timestamp": cv2.getTickCount()  # 当前时间戳（用于性能分析）
        }

    def _estimate_motion(self, kp1, kp2, matches):
        """
        计算两帧之间的视觉位移趋势
        
        参数:
        kp1: 上一帧的关键点列表
        kp2: 当前帧的关键点列表
        matches: 匹配结果列表
        
        返回:
        运动趋势描述（字符串）或详细的运动参数
        
        这是一个简化版本的运动估计算法
        真实系统中会计算：
        1. 本质矩阵（Essential Matrix）
        2. 基础矩阵（Fundamental Matrix）
        3. 单应性矩阵（Homography）
        4. 通过RANSAC去除异常值
        5. 恢复旋转和平移
        """
        # 如果匹配点数量太少，返回None
        # 10是一个经验值，实际应用中可能需要更多匹配点
        if len(matches) < 10: 
            return None
            
        # 计算匹配点间的平均位移
        # 这是一个简化的运动估计
        # 在实际应用中，会计算更复杂的运动模型
        
        # 示例：可以计算匹配点的平均位移向量
        # displacements = []
        # for match in matches:
        #     pt1 = kp1[match.queryIdx].pt
        #     pt2 = kp2[match.trainIdx].pt
        #     displacement = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        #     displacements.append(displacement)
        # 
        # avg_dx = np.mean([d[0] for d in displacements])
        # avg_dy = np.mean([d[1] for d in displacements])
        # 
        # return {"dx": avg_dx, "dy": avg_dy}
        
        # 原代码中返回一个固定的字符串
        return "SHIFT_DETECTED"