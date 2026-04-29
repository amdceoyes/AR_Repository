import numpy as np
import cv2

class Keyframe:
    """
    关键帧类：保存空间记忆的载体
    
    在视觉SLAM（Simultaneous Localization and Mapping，同步定位与地图构建）中，
    关键帧是系统在特定位置、特定时间点保存的"快照"，包含：
    1. 视觉特征（特征点、描述子）
    2. 相机位姿（位置和方向）
    3. 时间戳或帧ID
    
    关键帧是构建地图的基础，后续的定位和建图都依赖于这些关键帧。
    """
    
    def __init__(self, frame_id, kpts, des, rvec, tvec):
        """
        初始化关键帧
        
        参数:
        frame_id: 帧的唯一标识符（通常是帧编号）
        kpts: 关键点列表，每个关键点是一个包含(x,y)坐标的对象
        des: 描述子矩阵，形状为(n, 32)或(n, 64)的numpy数组，n是关键点数量
        rvec: 旋转向量（3×1），表示相机在世界坐标系中的方向
        tvec: 平移向量（3×1），表示相机在世界坐标系中的位置
        """
        self.id = frame_id
        self.keypoints = kpts      # 特征点坐标
        self.descriptors = des     # 描述子
        self.rvec = rvec.copy()    # 旋转向量（深拷贝，避免后续修改影响）
        self.tvec = tvec.copy()    # 平移向量（深拷贝，避免后续修改影响）

class SimpleMap:
    """
    简化版地图管理：负责关键帧的筛选与保存
    
    核心功能：
    1. 管理关键帧列表
    2. 决定何时添加新的关键帧
    3. 避免冗余的关键帧，节约存储空间
    
    在完整的SLAM系统中，地图管理还包括：
    - 回环检测
    - 位姿图优化
    - 地图点管理
    - 重定位
    """
    
    def __init__(self, dist_thresh=10.0, angle_thresh=15.0, min_matches=50):
        """
        初始化地图管理器
        
        参数:
        dist_thresh: 平移阈值（单位：与世界坐标系一致，通常是米）
                    当相机移动超过这个距离时，创建新关键帧
        angle_thresh: 旋转阈值（单位：度）
                    当相机旋转超过这个角度时，创建新关键帧
        min_matches: 最小匹配数阈值
                    只有当前帧与上一关键帧的匹配点数量超过这个值，才考虑创建新关键帧
        """
        self.keyframes = []  # 存储关键帧对象的列表
        
        # 关键帧添加的判定阈值
        self.dist_thresh = dist_thresh    # 平移阈值
        self.angle_thresh = angle_thresh  # 旋转阈值（度）
        self.min_matches = min_matches    # 稳定性阈值
        
        self.frame_count = 0  # 帧计数器

    def should_add_keyframe(self, current_rvec, current_tvec, match_count):
        """
        判断是否应该将当前帧添加为新的关键帧
        
        关键帧选择策略：
        1. 当前帧必须稳定（足够多的特征匹配）
        2. 当前帧与上一个关键帧在空间上有足够的变化
        
        参数:
        current_rvec: 当前帧的旋转向量
        current_tvec: 当前帧的平移向量
        match_count: 当前帧与上一关键帧的匹配特征点数量
        
        返回:
        bool: True表示应该添加为关键帧，False表示不应该添加
        """
        # 1. 基础检查：追踪是否稳定
        # 如果匹配点数量太少，说明特征追踪不稳定，不适合作为关键帧
        if match_count < self.min_matches:
            return False
        
        # 2. 如果还没有关键帧，直接添加第一个
        # 第一个关键帧是地图的起点
        if not self.keyframes:
            return True

        # 3. 计算与上一个关键帧的差异
        # 获取最近添加的关键帧
        last_kf = self.keyframes[-1]
        
        # 平移距离：计算欧几里得距离
        # ||t1 - t2|| = √((x1-x2)² + (y1-y2)² + (z1-z2)²)
        dist = np.linalg.norm(current_tvec - last_kf.tvec)
        
        # 旋转角度差异
        # 严格来说应该将旋转向量转换为旋转矩阵，然后计算旋转矩阵之间的角度
        # 简化方法：使用旋转向量的欧几里得范数作为旋转量的度量
        # 注意：旋转向量是轴角表示，其长度等于旋转角度（弧度）
        # 这里我们比较两个旋转向量的差异
        angle_diff = np.linalg.norm(current_rvec - last_kf.rvec) * (180 / np.pi)
        
        # 4. 判定：移动距离够远 OR 旋转角度够大
        # 满足任一条件就添加新关键帧
        if dist > self.dist_thresh or angle_diff > self.angle_thresh:
            return True
        
        # 5. 不满足添加条件
        return False

    def add_keyframe(self, kpts, des, rvec, tvec):
        """
        添加新的关键帧到地图中
        
        参数:
        kpts: 关键点列表
        des: 描述子矩阵
        rvec: 旋转向量
        tvec: 平移向量
        """
        # 创建新的关键帧对象
        new_kf = Keyframe(self.frame_count, kpts, des, rvec, tvec)
        
        # 添加到关键帧列表
        self.keyframes.append(new_kf)
        
        # 增加帧计数器
        self.frame_count += 1
        
        # 输出日志信息
        print(f"✨ [New Keyframe Added] ID: {new_kf.id} | Total: {len(self.keyframes)}")
        print(f"   Pos: {tvec.flatten()} | Count: {len(kpts)} points")



"""
关键帧管理策略详解
1. 为什么需要关键帧？
在视觉SLAM中，我们不能（也不应该）保存每一帧，原因包括：
1.1 存储效率
原始方法：保存所有帧 问题：1分钟视频（30FPS） = 1800帧 × 每个描述子1KB = 1.8GB 结果：存储爆炸，无法长期运行 关键帧方法：只保存重要的帧 假设：每10秒保存1个关键帧 结果：1分钟 = 6个关键帧 × 1KB = 6KB 优势：存储需求降低300倍 
1.2 计算效率
匹配比较：当前帧需要与历史帧进行特征匹配 - 与所有帧匹配：计算量巨大，无法实时 - 与关键帧匹配：计算量可控，可以实时 
1.3 信息冗余
连续帧之间：80%以上的特征点是相同的 关键帧之间：特征点变化显著，信息不冗余 
2. 关键帧选择策略
2.1 基于平移距离
# 计算相机移动距离 dist = np.linalg.norm(current_tvec - last_kf.tvec) # 阈值选择经验值： # - 室内场景：0.1-0.5米 # - 室外场景：0.5-2.0米 # 本代码中使用10.0，可能是为了演示，实际应用中应该调整 
物理意义：
距离太小：关键帧太密集，冗余
距离太大：关键帧太稀疏，可能导致跟踪丢失
适当距离：既能覆盖环境，又不浪费资源
2.2 基于旋转角度
# 计算旋转角度差异 angle_diff = np.linalg.norm(current_rvec - last_kf.rvec) * (180 / np.pi) # 阈值选择经验值： # - 平稳场景：10-30度 # - 快速转动：5-15度 
物理意义：
角度太小：视角变化不大，特征重复
角度太大：可能视角变化太大，特征匹配困难
适当角度：既能覆盖不同视角，又能保持特征匹配
2.3 基于特征匹配数量
# 检查匹配点数量 if match_count < self.min_matches: return False # 不添加关键帧 # 阈值选择经验值： # - 特征丰富的场景：20-50个匹配点 # - 特征稀疏的场景：10-20个匹配点 
物理意义：
匹配太少：跟踪不稳定，不适合作为参考
匹配适中：跟踪稳定，适合作为关键帧
匹配很多：可能太相似，冗余
3. 数学原理深入
3.1 旋转向量与旋转角度
旋转向量（轴角表示）：
方向：旋转轴
长度：旋转角度（弧度）
旋转角度计算（精确方法）：
def rotation_angle(rvec1, rvec2): ""计算两个旋转向量之间的角度差"" # 将旋转向量转换为旋转矩阵 R1, _ = cv2.Rodrigues(rvec1) R2, _ = cv2.Rodrigues(rvec2) # 计算相对旋转 R_rel = R2 * R1.T R_rel = np.dot(R2, R1.T) # 从旋转矩阵提取旋转角度 # 公式：θ = arccos((trace(R) - 1) / 2) trace_R = np.trace(R_rel) angle_rad = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0)) return np.degrees(angle_rad) 
代码中的简化方法：
# 简化：使用欧几里得距离近似 angle_diff = np.linalg.norm(rvec1 - rvec2) * (180 / np.pi) # 这个简化在旋转角度较小（<15度）时近似较好 # 但在大角度旋转时误差较大 
3.2 平移距离计算
欧几里得距离公式：
dist = √((x2-x1)² + (y2-y1)² + (z2-z1)²) 
实际应用中的考虑：
# 在实际SLAM中，我们可能要考虑： # 1. 尺度不确定性（单目SLAM） # 2. 不同坐标轴的重要性（比如室内场景中Z轴变化较小） # 3. 场景尺度（室内vs室外） # 调整权重的方法 def weighted_distance(tvec1, tvec2, weights=[1.0, 1.0, 0.5]): ""加权欧几里得距离，Z轴权重较小（假设地面基本水平）"" diff = tvec1 - tvec2 weighted_diff = diff * weights return np.linalg.norm(weighted_diff) 
4. 扩展功能建议
4.1 自适应阈值调整
class AdaptiveMap(SimpleMap): def __init__(self): super().__init__() self.last_dist = 0 self.last_angle = 0 def should_add_keyframe(self, current_rvec, current_tvec, match_count): # 基础检查 if match_count < self.min_matches: return False if not self.keyframes: return True last_kf = self.keyframes[-1] # 计算当前变化 dist = np.linalg.norm(current_tvec - last_kf.tvec) angle_diff = np.linalg.norm(current_rvec - last_kf.rvec) * (180 / np.pi) # 自适应阈值 # 如果之前变化很大，这次可以宽松一些 # 如果之前变化很小，这次可以严格一些 adaptive_dist_thresh = self.dist_thresh * (1.0 + 0.5 * np.tanh(self.last_dist - dist)) adaptive_angle_thresh = self.angle_thresh * (1.0 + 0.5 * np.tanh(self.last_angle - angle_diff)) # 保存当前值供下次使用 self.last_dist = dist self.last_angle = angle_diff # 判断 if dist > adaptive_dist_thresh or angle_diff > adaptive_angle_thresh: return True return False 
4.2 关键帧质量评估
def evaluate_keyframe_quality(self, kpts, des, rvec, tvec): "" 评估关键帧质量 返回0-1之间的分数，1表示质量最好 "" quality_score = 1.0 # 1. 特征点数量 num_features = len(kpts) if num_features < 50: quality_score *= 0.5 elif num_features > 300: quality_score *= 1.0 else: quality_score *= num_features / 300 # 2. 特征点分布 points = np.array([kp.pt for kp in kpts]) std_x, std_y = np.std(points[:, 0]), np.std(points[:, 1]) # 如果特征点分布均匀，质量高 if std_x > 50 and std_y > 50: quality_score *= 1.0 else: quality_score *= 0.7 # 3. 特征点响应值（如果可用） if hasattr(kpts[0], 'response'): responses = [kp.response for kp in kpts] avg_response = np.mean(responses) quality_score *= min(1.0, avg_response / 100.0) return quality_score 
4.3 关键帧剔除
def prune_keyframes(self, max_keyframes=50): "" 当关键帧数量超过限制时，剔除质量较低的关键帧 "" if len(self.keyframes) <= max_keyframes: return # 评估所有关键帧的质量 qualities = [] for kf in self.keyframes: quality = self.evaluate_keyframe_quality( kf.keypoints, kf.descriptors, kf.rvec, kf.tvec ) qualities.append(quality) # 按质量排序 sorted_indices = np.argsort(qualities) # 保留质量最高的max_keyframes个关键帧 new_keyframes = [] for i in sorted_indices[-max_keyframes:]: new_keyframes.append(self.keyframes[i]) self.keyframes = new_keyframes print(f"Pruned keyframes: {len(sorted_indices)} -> {len(new_keyframes)}") 
5. 实际应用示例
5.1 在SLAM系统中使用
class SimpleSLAM: def __init__(self): self.map = SimpleMap(dist_thresh=0.5, angle_thresh=15.0, min_matches=30) self.current_pose = None self.frame_id = 0 def process_frame(self, frame): # 1. 提取特征 kpts, des = self.extract_features(frame) # 2. 计算当前帧位姿（通过特征匹配和PnP） rvec, tvec, match_count = self.estimate_pose(kpts, des) # 3. 更新当前位姿 self.current_pose = (rvec, tvec) # 4. 判断是否添加关键帧 if self.map.should_add_keyframe(rvec, tvec, match_count): self.map.add_keyframe(kpts, des, rvec, tvec) # 5. 可选：定期剔除多余关键帧 if self.frame_id % 100 == 0: self.map.prune_keyframes(max_keyframes=20) self.frame_id += 1 def extract_features(self, frame): ""提取ORB特征"" orb = cv2.ORB_create(nfeatures=1000) kpts, des = orb.detectAndCompute(frame, None) return kpts, des def estimate_pose(self, kpts, des): ""通过特征匹配估计相机位姿"" # 与上一个关键帧匹配 if self.map.keyframes: last_kf = self.map.keyframes[-1] # 特征匹配 bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) matches = bf.match(last_kf.descriptors, des) match_count = len(matches) if match_count > 20: # 提取匹配点 src_pts = np.float32([last_kf.keypoints[m.queryIdx].pt for m in matches]) dst_pts = np.float32([kpts[m.trainIdx].pt for m in matches]) # 计算单应性矩阵 H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) if H is not None: # 从单应性矩阵分解出旋转和平移 # 这里简化处理，实际应用中应该使用PnP _, rvec, tvec, _ = cv2.decomposeHomographyMat(H, self.camera_matrix) return rvec[0], tvec[0], match_count return np.zeros(3), np.zeros(3), 0 
5.2 可视化关键帧
def visualize_map(self, frame): ""在当前帧上可视化关键帧位置"" for kf in self.map.keyframes: # 将关键帧位置投影到当前帧 img_pts, _ = cv2.projectPoints( np.array([[0, 0, 0]]), # 关键帧位置 kf.rvec, kf.tvec, self.camera_matrix, self.dist_coeffs ) x, y = int(img_pts[0][0][0]), int(img_pts[0][0][1]) # 绘制关键帧位置 cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) cv2.putText(frame, f"KF{kf.id}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) return frame 
6. 性能优化建议
6.1 使用空间索引加速
from scipy.spatial import KDTree class SpatialIndexMap(SimpleMap): def __init__(self): super().__init__() self.kdtree = None self.kdtree_points = [] def update_spatial_index(self): ""更新关键帧空间索引"" points = [] for kf in self.keyframes: points.append(kf.tvec.flatten()) if points: self.kdtree = KDTree(points) self.kdtree_points = points def find_nearest_keyframes(self, tvec, k=3): ""找到最近的k个关键帧"" if self.kdtree is None: return [] distances, indices = self.kdtree.query(tvec.flatten(), k=k) return [(self.keyframes[i], distances[j]) for j, i in enumerate(indices)] 
6.2 并行特征提取
import threading from queue import Queue class ParallelSLAM(SimpleSLAM): def __init__(self): super().__init__() self.feature_queue = Queue() self.result_queue = Queue() self.worker_thread = None def start_worker(self): ""启动工作线程进行并行特征提取"" self.worker_thread = threading.Thread(target=self.feature_worker) self.worker_thread.daemon = True self.worker_thread.start() def feature_worker(self): ""工作线程：提取特征"" while True: frame = self.feature_queue.get() if frame is None: # 结束信号 break kpts, des = self.extract_features(frame) self.result_queue.put((kpts, des)) def process_frame_parallel(self, frame): ""并行处理帧"" # 将当前帧放入队列 self.feature_queue.put(frame) # 处理上一帧的结果（如果存在） if not self.result_queue.empty(): kpts, des = self.result_queue.get() # 继续处理... 
这个关键帧和地图管理系统是视觉SLAM的核心组件，它决定了系统的内存使用效率、计算效率和建图质量。通过合理的关键帧选择策略，可以在保持高精度定位的同时，实现长时间、大规模的建图。
"""