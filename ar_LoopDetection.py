import cv2
import numpy as np
import time

class Keyframe:
    """
    关键帧：系统的'笔记'载体
    在SLAM系统中，关键帧是系统在特定位置、特定时间点保存的'快照'
    包含该位置的视觉特征和相机位姿信息
    """
    
    def __init__(self, kf_id, des, rvec, tvec):
        """
        初始化关键帧对象
        
        参数:
        kf_id: 关键帧的唯一标识符
        des: 描述子矩阵，存储视觉特征
        rvec: 旋转向量，表示相机在世界坐标系中的方向
        tvec: 平移向量，表示相机在世界坐标系中的位置
        """
        self.id = kf_id
        self.descriptors = des     # 该位置的视觉特征（用于匹配）
        self.rvec = rvec           # 对应的旋转（方向）
        self.tvec = tvec           # 对应的平移（位置）

class SimpleSLAMSystem:
    """
    简化版SLAM系统
    实现基本的视觉SLAM功能：定位、建图和回环检测
    
    主要功能：
    1. 特征提取与匹配
    2. 关键帧管理
    3. 回环检测
    """
    
    def __init__(self):
        """
        初始化SLAM系统
        包括硬件连接、空间记忆和参数设定
        """
        # 1. 硬件连接：初始化ORB检测器
        # ORB是快速的特征检测和描述子算法，适合实时应用
        # nfeatures=1000: 每帧最多提取1000个特征点
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # 暴力匹配器，用于特征匹配
        # cv2.NORM_HAMMING: ORB是二进制描述子，使用汉明距离
        # crossCheck=True: 开启交叉验证，确保匹配质量
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 2. 空间记忆：存放关键帧的'本子'
        # 这个列表存储所有关键帧对象
        self.keyframes = []
        
        # 关键帧计数器，用于生成唯一ID
        self.kf_counter = 0
        
        # 3. 阈值设定（这是你的'工业化'参数）
        # 这些参数控制系统的行为，需要根据实际场景调整
        
        # 距离阈值：当相机移动超过这个距离时，创建新的关键帧
        # 单位：与世界坐标系一致，通常是米
        self.dist_threshold = 15.0
        
        # 角度阈值：当相机旋转超过这个角度时，创建新的关键帧
        # 单位：度
        self.angle_threshold = 20.0
        
        # 回环匹配阈值：当匹配点超过这个数量时，认为检测到回环
        self.loop_match_threshold = 85

    def extract_features(self, frame):
        """
        本地感知层：从图像中提取特征
        
        参数:
        frame: 输入图像帧
        
        返回:
        kps: 关键点列表
        des: 描述子矩阵
        """
        # 使用ORB算法检测关键点和计算描述子
        kps, des = self.orb.detectAndCompute(frame, None)
        return kps, des

    def should_add_keyframe(self, rvec, tvec, des):
        """
        逻辑判定：是否需要添加新的关键帧
        
        判断依据：
        1. 如果没有关键帧，总是添加第一个
        2. 如果移动距离超过阈值
        3. 如果旋转角度超过阈值
        
        参数:
        rvec: 当前帧的旋转向量
        tvec: 当前帧的平移向量
        des: 当前帧的描述子（这里没有使用，但可以用于质量评估）
        
        返回:
        bool: 是否需要添加关键帧
        """
        # 如果还没有任何关键帧，总是添加第一个
        if not self.keyframes:
            return True
        
        # 获取最近的关键帧
        last_kf = self.keyframes[-1]
        
        # 计算平移距离（L2范数/欧几里得距离）
        # 公式：√((x2-x1)² + (y2-y1)² + (z2-z1)²)
        dist = np.linalg.norm(tvec - last_kf.tvec)
        
        # 计算角度变化（近似值）
        # 严格来说应该将旋转向量转换为旋转矩阵，然后计算旋转角度
        # 这里使用旋转向量的欧几里得范数作为近似
        # 旋转向量的长度等于旋转角度（弧度），所以乘以180/π转换为度
        angle_diff = np.linalg.norm(rvec - last_kf.rvec) * (180 / np.pi)
        
        # 如果移动距离或旋转角度超过阈值，则添加新的关键帧
        if dist > self.dist_threshold or angle_diff > self.angle_threshold:
            return True
        
        return False

    def detect_loop(self, current_des):
        """
        回环检测：跨越时空的查重
        检测当前帧是否回到了之前访问过的位置
        
        参数:
        current_des: 当前帧的描述子
        
        返回:
        如果检测到回环：返回(关键帧ID, 匹配点数量)
        如果没有检测到回环：返回None
        """
        # 性能策略：只对比20帧以前的关键帧，避免原地查重
        # 如果关键帧数量太少，不进行回环检测
        if len(self.keyframes) < 10:
            return None
        
        # 遍历历史关键帧（跳过最近的20帧，避免检测到刚刚经过的位置）
        for kf in self.keyframes[:-20]:
            # 将当前帧与历史关键帧进行特征匹配
            matches = self.bf.match(current_des, kf.descriptors)
            
            # 筛选优质匹配：距离小于35的匹配点
            # 距离越小，匹配质量越高
            good_matches = [m for m in matches if m.distance < 35]
            
            # 如果优质匹配点数量超过阈值，认为检测到回环
            if len(good_matches) > self.loop_match_threshold:
                return kf.id, len(good_matches)
        
        return None

    def run_frame(self, frame, current_rvec, current_tvec):
        """
        主循环：每一帧的处理逻辑
        
        参数:
        frame: 当前图像帧
        current_rvec: 当前帧的旋转向量（从PnP或其他位姿估计算法得到）
        current_tvec: 当前帧的平移向量（从PnP或其他位姿估计算法得到）
        """
        # 从当前帧提取特征
        kps, des = self.extract_features(frame)
        
        # 如果没有提取到特征，直接返回
        if des is None:
            return
        
        # --- 动作1: 关键帧保存 ---
        # 判断是否需要添加新的关键帧
        if self.should_add_keyframe(current_rvec, current_tvec, des):
            # 创建新的关键帧对象
            new_kf = Keyframe(self.kf_counter, des, current_rvec, current_tvec)
            
            # 添加到关键帧列表
            self.keyframes.append(new_kf)
            
            # 输出日志
            print(f"✨ [New Keyframe] ID: {self.kf_counter} saved.")
            
            # 增加关键帧计数器
            self.kf_counter += 1
            
        # --- 动作2: 回环检测（翻本子） ---
        # 检测是否回到了之前访问过的位置
        loop_result = self.detect_loop(des)
        
        if loop_result:
            # 如果检测到回环
            kf_id, match_count = loop_result
            print(f"🔥 [LOOP DETECTED] Re-visited Area! Matched with KF #{kf_id} ({match_count} pts)")
            
            # 此时系统可以'知道'漂移发生了，并准备修正
            # 在实际的SLAM系统中，这里会触发位姿图优化
            # 将回环检测结果用于校正累积误差

# --- 模拟使用流程 ---
"""
# 创建SLAM系统实例
slam = SimpleSLAMSystem()

# 主循环
while True:
    # 1. 获取图像和位姿（来自你的PnP或其他位姿估计算法）
    # ret, frame = cap.read()
    # rvec, tvec = estimate_pose(frame)  # 这需要你自己实现
    
    # 2. 运行SLAM系统
    # slam.run_frame(frame, rvec, tvec)
    
    # 3. 根据结果渲染AR
    # 可以根据关键帧和回环检测结果增强AR体验
    
    # 4. 显示结果
    # cv2.imshow("SLAM System", frame)
    
    # 5. 检查退出条件
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
"""

"""
系统架构与技术原理详解
1. SLAM系统概述
SLAM（Simultaneous Localization and Mapping，同步定位与地图构建）：
目标：在未知环境中，同时估计自身位置并构建环境地图 输入：传感器数据（摄像头、IMU、激光雷达等） 输出：1. 自身位姿（定位） 2. 环境地图（建图） 应用：机器人导航、自动驾驶、增强现实 
视觉SLAM（使用摄像头）的挑战：
尺度不确定性：单目摄像头无法直接获取深度
累积误差：小误差随时间累积导致漂移
计算复杂度：需要实时处理图像数据
环境变化：光照、动态物体等干扰
2. 关键帧管理策略
2.1 为什么需要关键帧？
存储效率：
原始方法：保存所有帧 问题：1分钟视频（30FPS）= 1800帧 × 1KB/帧 ≈ 1.8MB 结果：内存迅速耗尽，无法长期运行 关键帧方法：只保存重要的帧 假设：每2秒保存1个关键帧 结果：1分钟 = 30个关键帧 × 1KB/帧 ≈ 30KB 优势：存储需求降低60倍 
计算效率：
匹配比较：当前帧需要与历史帧匹配 - 与所有帧匹配：计算量巨大，无法实时 - 与关键帧匹配：计算量可控，可以实时 
信息冗余：
连续帧之间：80%以上的信息是重复的 关键帧之间：信息变化显著，冗余度低 
2.2 关键帧添加条件
距离条件：
dist = np.linalg.norm(tvec - last_kf.tvec) if dist > self.dist_threshold: add_keyframe() 
角度条件：
angle_diff = np.linalg.norm(rvec - last_kf.rvec) * (180 / np.pi) if angle_diff > self.angle_threshold: add_keyframe() 
阈值选择经验：
室内场景： - 距离阈值：0.1-0.5米 - 角度阈值：10-30度 室外场景： - 距离阈值：0.5-2.0米 - 角度阈值：5-20度 
3. 回环检测原理
3.1 什么是回环检测？
定义：检测相机是否回到了之前访问过的位置
重要性：
校正累积误差：将当前位置与历史位置对齐
构建一致地图：避免同一位置在地图中出现多次
提高定位精度：利用历史信息修正当前位姿
3.2 回环检测算法
特征匹配法：
# 1. 提取当前帧特征 current_des = extract_features(current_frame) # 2. 与历史关键帧匹配 for historical_kf in keyframes: matches = match_features(current_des, historical_kf.des) # 3. 统计匹配点数量 good_matches = filter_matches(matches) # 4. 判断是否回环 if len(good_matches) > threshold: return historical_kf 
性能优化：
# 跳过最近的关键帧（避免检测到刚刚经过的位置） for kf in self.keyframes[:-20]: # 只检查20帧以前的关键帧 
匹配质量评估：
# 使用距离阈值筛选优质匹配 good_matches = [m for m in matches if m.distance < 35] # 阈值选择：距离越小，匹配质量越高 # ORB描述子的距离范围：0-256 # 经验值：20-50之间 
4. 完整的SLAM工作流程
4.1 初始化阶段
1. 创建ORB检测器和匹配器 2. 初始化空的关键帧列表 3. 设置系统参数 
4.2 跟踪阶段（每帧）
1. 提取当前帧特征 2. 估计相机位姿（通过PnP或其他方法） 3. 判断是否添加关键帧 4. 执行回环检测 5. 更新系统状态 
4.3 优化阶段（检测到回环时）
1. 识别回环关键帧 2. 计算相对位姿变换 3. 执行位姿图优化 4. 校正地图和轨迹 
5. 参数调优指南
5.1 特征提取参数
# ORB参数调优 self.orb = cv2.ORB_create( nfeatures=1000, # 特征点数量 scaleFactor=1.2, # 金字塔尺度因子 nlevels=8, # 金字塔层数 edgeThreshold=31, # 边缘阈值 firstLevel=0, # 第一层 WTA_K=2, # 产生描述子的点数 scoreType=cv2.ORB_HARRIS_SCORE, # 评分类型 patchSize=31, # 描述子块大小 fastThreshold=20 # FAST阈值 ) 
调整建议：
特征丰富场景：减少nfeatures（500-800）
特征稀疏场景：增加nfeatures（1000-1500）
实时性要求高：减少nlevels（4-6）
精度要求高：增加nlevels（8-10）
5.2 匹配参数
# 匹配器参数 self.bf = cv2.BFMatcher( cv2.NORM_HAMMING, # 距离度量 crossCheck=True # 交叉验证 ) 
交叉验证的优势：
传统匹配：A→B单向匹配 交叉验证：A→B和B→A双向验证 结果：匹配质量更高，错误匹配更少 代价：计算量增加一倍 
5.3 系统参数
# 关键帧添加阈值 self.dist_threshold = 15.0 # 根据场景尺度调整 self.angle_threshold = 20.0 # 根据旋转速度调整 # 回环检测阈值 self.loop_match_threshold = 85 # 根据特征点数量调整 
调整方法：
小尺度场景：减小距离阈值
大尺度场景：增大距离阈值
快速旋转：减小角度阈值
慢速旋转：增大角度阈值
特征丰富：增大回环阈值
特征稀疏：减小回环阈值
6. 扩展功能建议
6.1 添加位姿图优化
def optimize_pose_graph(self, loop_kf_id, current_rvec, current_tvec): "" 位姿图优化 当检测到回环时，优化所有关键帧的位姿 "" # 获取回环关键帧的位姿 loop_kf = self.keyframes[loop_kf_id] # 计算相对位姿变换 # 这里简化处理，实际应该使用更复杂的优化算法 relative_rvec = current_rvec - loop_kf.rvec relative_tvec = current_tvec - loop_kf.tvec # 对轨迹进行校正 # 实际应用中会使用g2o、Ceres等优化库 print(f"Pose graph optimization triggered by loop closure with KF #{loop_kf_id}") 
6.2 添加关键帧质量评估
def evaluate_keyframe_quality(self, des, kps): "" 评估关键帧质量 返回质量分数（0-1） "" quality = 1.0 # 1. 特征点数量 num_features = len(kps) if num_features < 50: quality *= 0.3 elif num_features < 100: quality *= 0.6 elif num_features < 200: quality *= 0.8 else: quality *= 1.0 # 2. 特征点分布 points = np.array([kp.pt for kp in kps]) std_x, std_y = np.std(points[:, 0]), np.std(points[:, 1]) # 如果特征点分布均匀，质量更高 if std_x < 50 or std_y < 50: quality *= 0.7 return quality 
6.3 添加重定位功能
def relocalize(self, frame): "" 重定位：当跟踪丢失时，重新确定位置 "" # 提取当前帧特征 kps, des = self.extract_features(frame) if des is None: return None # 与所有关键帧匹配 best_match = None best_score = 0 for kf in self.keyframes: matches = self.bf.match(des, kf.descriptors) good_matches = [m for m in matches if m.distance < 40] score = len(good_matches) if score > best_score and score > 30: best_score = score best_match = kf if best_match: print(f"Relocalized! Matched with KF #{best_match.id} ({best_score} matches)") return best_match.rvec, best_match.tvec return None 
6.4 添加可视化功能
def visualize_trajectory(self, frame, current_rvec, current_tvec): "" 可视化相机轨迹 "" # 绘制所有关键帧位置 for kf in self.keyframes: # 将关键帧位置投影到当前帧 # 这里简化处理，实际需要正确的投影 x = int(kf.tvec[0] * 10 + frame.shape[1] // 2) y = int(kf.tvec[1] * 10 + frame.shape[0] // 2) cv2.circle(frame, (x, y), 3, (0, 255, 0), -1) # 绘制当前帧位置 x = int(current_tvec[0] * 10 + frame.shape[1] // 2) y = int(current_tvec[1] * 10 + frame.shape[0] // 2) cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) return frame 
7. 实际应用示例
7.1 完整的主程序
def main(): # 创建SLAM系统 slam = SimpleSLAMSystem() # 打开摄像头 cap = cv2.VideoCapture(0) # 帧计数器 frame_id = 0 while True: # 读取一帧 ret, frame = cap.read() if not ret: break # 水平翻转，使操作更直观 frame = cv2.flip(frame, 1) # 简化：这里假设我们已经有了位姿估计 # 在实际应用中，位姿应该从PnP或其他算法得到 # 模拟相机运动：绕圈运动 angle = frame_id * 0.05 current_rvec = np.array([0, 0, angle]) current_tvec = np.array([ np.cos(angle) * 10, np.sin(angle) * 10, frame_id * 0.1 ]) # 运行SLAM系统 slam.run_frame(frame, current_rvec, current_tvec) # 可视化 # 可以添加轨迹可视化、关键帧显示等 # 显示帧 cv2.imshow("SLAM System", frame) # 检查退出键 if cv2.waitKey(1) & 0xFF == ord('q'): break frame_id += 1 # 释放资源 cap.release() cv2.destroyAllWindows() # 输出统计信息 print(f"\n=== SLAM System Statistics ===") print(f"Total frames processed: {frame_id}") print(f"Total keyframes saved: {len(slam.keyframes)}") print(f"Keyframe density: {len(slam.keyframes)/frame_id*100:.1f}%") if __name__ == "__main__": main() 
7.2 与AR系统集成
class ARWithSLAM: "" 将SLAM系统与AR系统集成 "" def __init__(self): self.slam = SimpleSLAMSystem() self.camera_matrix = None self.dist_coeffs = None def setup_camera(self, camera_matrix, dist_coeffs): ""设置相机参数"" self.camera_matrix = camera_matrix self.dist_coeffs = dist_coeffs def process_frame(self, frame): ""处理单帧：SLAM + AR"" # 1. 提取特征 kps, des = self.slam.extract_features(frame) if des is None: return frame # 2. 估计位姿（通过PnP） # 这里简化，实际需要与上一关键帧匹配 rvec, tvec = self.estimate_pose(kps, des) if rvec is not None: # 3. 运行SLAM self.slam.run_frame(frame, rvec, tvec) # 4. 渲染AR内容 frame = self.render_ar_content(frame, rvec, tvec) return frame def estimate_pose(self, kps, des): ""通过PnP估计位姿"" # 简化实现 # 实际应该与上一关键帧匹配，然后使用solvePnP return np.zeros(3), np.zeros(3) def render_ar_content(self, frame, rvec, tvec): ""渲染AR内容"" # 在图像上绘制坐标轴 cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1) return frame 
这个简化版SLAM系统提供了一个完整的视觉SLAM框架，包括特征提取、关键帧管理、回环检测等核心功能。通过调整参数和添加扩展功能，可以适应各种实际应用场景。
"""