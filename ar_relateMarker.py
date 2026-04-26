import cv2
import cv2.aruco as aruco
import numpy as np

class WorldSpaceAR:
    """
    世界空间增强现实系统
    采用"基于已知标记的定位"策略，实现相机在预先构建的世界坐标系中的定位
    
    核心思想：通过已知标记定位相机，再通过相机位置计算新标记的世界坐标
    这种方法比固定原点标记更灵活，适用于动态、大规模AR场景
    """
    
    def __init__(self, camera_matrix, dist_coeffs):
        """
        初始化世界空间AR系统
        
        参数:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 相机畸变系数
        """
        # 相机参数
        self.K = camera_matrix
        self.D = dist_coeffs
        
        # 原点标记ID（可选，但可以作为参考点）
        self.origin_id = 0
        
        # 世界地图：存储每个标记在世界坐标系中的4x4变换矩阵
        # 4x4变换矩阵格式：
        # [R11, R12, R13, tx]
        # [R21, R22, R23, ty]
        # [R31, R32, R33, tz]
        # [0,   0,   0,   1 ]
        # 其中R是3x3旋转矩阵，t是3x1平移向量
        # 初始时，原点标记在自己的坐标系中是单位矩阵（位置为0，不旋转）
        self.world_map = {self.origin_id: np.eye(4)}
        
        # 相机在世界坐标系中的当前位姿（4x4变换矩阵）
        # None表示相机位置未知，需要重新定位
        self.camera_pose_in_world = None

    def make_matrix(self, rvec, tvec):
        """
        将旋转向量和平移向量转换为4x4齐次变换矩阵
        
        参数:
        rvec: 旋转向量（3个值），表示方向
        tvec: 平移向量（3个值），表示位置
        
        返回: 4x4齐次变换矩阵
        """
        # 将旋转向量转换为3x3旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 创建4x4单位矩阵
        M = np.eye(4)
        
        # 设置旋转部分
        M[:3, :3] = R
        
        # 设置平移部分
        M[:3, 3] = tvec.flatten()
        
        return M

    def update(self, ids, rvecs, tvecs):
        """
        更新系统状态：相机定位和世界地图扩展
        
        参数:
        ids: 检测到的标记ID数组
        rvecs: 检测到的标记的旋转向量数组
        tvecs: 检测到的标记的平移向量数组
        """
        # 如果没有检测到任何标记
        if ids is None:
            self.camera_pose_in_world = None
            return

        # 将ID数组展平为1D数组
        ids = ids.flatten()
        
        # 标记是否找到已知的定位标记
        found_localized_marker = False

        # --- 阶段 1：重定位 (Relocalization) ---
        # 寻找当前画面中，是否有任何一个标记已经存在于我们的"世界地图"中
        # 只要有1个已知标记，我们就能计算相机在世界坐标系中的位置
        for i, m_id in enumerate(ids):
            if m_id in self.world_map:
                # 获取当前标记相对于相机的变换矩阵
                # T_cm: 从标记坐标系到相机坐标系的变换
                T_cm = self.make_matrix(rvecs[i], tvecs[i])
                
                # 获取这个标记在世界坐标系中的预存位置
                # T_wm: 从标记坐标系到世界坐标系的变换
                T_wm = self.world_map[m_id]
                 
                # 【核心接力公式】：计算相机在世界坐标系的位置
                # 已知：标记在世界中的位置 T_wm
                # 已知：标记相对于相机的位置 T_cm
                # 求：相机在世界中的位置 T_wc
                
                # 公式推导：
                # 1. 标记在世界中的位置：P_w = T_wm * P_m
                # 2. 标记在相机中的位置：P_c = T_cm * P_m
                # 3. 相机在世界中的位置：P_w = T_wc * P_c
                # 由1和3得：T_wm * P_m = T_wc * T_cm * P_m
                # 所以：T_wm = T_wc * T_cm
                # 因此：T_wc = T_wm * inv(T_cm)
                
                self.camera_pose_in_world = T_wm @ np.linalg.inv(T_cm)
                found_localized_marker = True
                break  # 只要找到一个已知标记就可以定位相机

        # --- 阶段 2：建图接力 (Mapping/Extension) ---
        # 如果定位成功，就把画面中其他还没录入的新标记"钉"在世界坐标系里
        if found_localized_marker:
            for i, m_id in enumerate(ids):
                if m_id not in self.world_map:
                    # 计算新标记相对于相机的变换矩阵
                    T_c_new = self.make_matrix(rvecs[i], tvecs[i])
                    
                    # 【接力公式】：计算新标记在世界坐标系中的位置
                    # 已知：相机在世界中的位置 T_wc
                    # 已知：新标记相对于相机的位置 T_c_new
                    # 求：新标记在世界中的位置 T_w_new
                    
                    # 公式：T_w_new = T_wc * T_c_new
                    T_w_new = self.camera_pose_in_world @ T_c_new
                    
                    # 将新标记加入世界地图
                    self.world_map[m_id] = T_w_new
                    
                    print(f"坐标接力：成功通过 ID {ids[0]} 锚定了新 ID {m_id}")

    def draw_world_origin(self, frame):
        """
        无论相机在哪，只要定位成功，就在世界原点画坐标轴
        
        参数:
        frame: 要绘制的图像帧
        """
        if self.camera_pose_in_world is not None:
            # OpenCV渲染需要的是"世界相对于相机"的变换 (T_cw)
            # 但我们有"相机相对于世界"的变换 (T_wc)
            # 所以需要计算逆变换：T_cw = inv(T_wc)
            T_cw = np.linalg.inv(self.camera_pose_in_world)
            
            # 从变换矩阵中提取旋转和平移
            rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
            tvec = T_cw[:3, 3]
            
            # 在世界原点位置画出长度为10厘米的坐标轴
            # 红色：X轴，绿色：Y轴，蓝色：Z轴
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)

# 主程序示例
if __name__ == "__main__":
    # 相机参数（示例值，实际应用中需要标定）
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)
    
    dist_coeffs = np.zeros((4, 1))
    
    # 创建世界空间AR系统
    ar_system = WorldSpaceAR(camera_matrix, dist_coeffs)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 定义ARUco字典
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 检测ARUco标记
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # 2. 如果有标记被检测到
        if ids is not None:
            # 计算标记的3D姿态
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, camera_matrix, dist_coeffs
            )
            
            # 3. 更新系统状态
            ar_system.update(ids, rvecs, tvecs)
            
            # 4. 在世界原点绘制坐标轴
            ar_system.draw_world_origin(frame)
            
            # 5. 绘制检测到的标记
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                rvecs[i][0], tvecs[i][0], 0.05)
        
        # 显示结果
        cv2.imshow('World Space AR', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


""" 
系统架构详细解析
1. 核心概念：变换矩阵表示
1.1 齐次坐标变换矩阵
4x4变换矩阵格式： [R11 R12 R13 tx] [R21 R22 R23 ty] [R31 R32 R33 tz] [0 0 0 1 ] 作用： 1. 将点从本地坐标系变换到世界坐标系 2. 表示刚体运动（旋转+平移） 3. 便于进行矩阵连乘运算 
1.2 常用变换表示
T_wc：从相机坐标系到世界坐标系的变换
T_cm：从标记坐标系到相机坐标系的变换
T_wm：从标记坐标系到世界坐标系的变换
2. 核心算法：两步定位法
步骤1：重定位（已知标记定位相机）
输入：检测到的标记ID、位置 过程： 1. 查找是否有已知标记（在world_map中） 2. 如果有，计算相机在世界中的位置 3. 公式：T_wc = T_wm * inv(T_cm) 
步骤2：建图接力（相机定位新标记）
输入：新检测到的标记 过程： 1. 用已知的相机位置 2. 计算新标记在世界中的位置 3. 公式：T_wm_new = T_wc * T_cm_new 
3. 数学推导详细说明
3.1 坐标变换基础
设： P_w: 点在世界坐标系中的坐标 P_c: 点在相机坐标系中的坐标 P_m: 点在标记坐标系中的坐标 变换关系： P_w = T_wm * P_m (1) 标记到世界 P_c = T_cm * P_m (2) 标记到相机 P_w = T_wc * P_c (3) 相机到世界 
3.2 相机定位推导
由(1)和(3)得： T_wm * P_m = T_wc * P_c 代入(2)： T_wm * P_m = T_wc * T_cm * P_m 消去P_m： T_wm = T_wc * T_cm 所以： T_wc = T_wm * inv(T_cm) 
3.3 新标记定位推导
由(3)得： P_w = T_wc * P_c 代入(2)： P_w = T_wc * T_cm_new * P_m_new 所以： T_wm_new = T_wc * T_cm_new 
4. 系统优势
4.1 灵活性
不依赖固定原点标记
只要有任意已知标记就能定位
支持动态环境
4.2 可扩展性
可以逐步扩展世界地图
支持大规模AR场景
便于多人协作
4.3 鲁棒性
单个标记丢失不影响系统
自动重新定位
容错能力强
5. 实际应用场景
5.1 室内导航
场景：大型商场、机场、医院 工作流程： 1. 在关键位置放置AR标记 2. 系统自动构建世界地图 3. 用户通过手机相机实时定位 4. 显示导航路径和目的地 
5.2 工业维护
场景：工厂设备维护 工作流程： 1. 在设备关键部件上放置标记 2. 建立设备数字孪生模型 3. 维修人员通过AR眼镜查看维修指导 4. 虚拟信息与实际设备精确对齐 
5.3 教育培训
场景：实验室、教室 工作流程： 1. 在实验设备上放置标记 2. 建立虚拟实验环境 3. 学生通过AR设备观察实验过程 4. 虚拟信息与实验设备交互 
6. 扩展功能
6.1 持久性存储
def save_world_map(self, filename): ""保存世界地图到文件"" data = {} for marker_id, transform_matrix in self.world_map.items(): data[str(marker_id)] = transform_matrix.tolist() import json with open(filename, 'w') as f: json.dump(data, f) def load_world_map(self, filename): ""从文件加载世界地图"" import json with open(filename, 'r') as f: data = json.load(f) self.world_map = {} for marker_id_str, matrix_data in data.items(): marker_id = int(marker_id_str) self.world_map[marker_id] = np.array(matrix_data) 
6.2 多标记优化
def optimize_with_multiple_markers(self, ids, rvecs, tvecs): "" 使用多个已知标记优化相机位置 提高定位精度 "" if ids is None or self.camera_pose_in_world is None: return ids = ids.flatten() # 收集所有已知标记的对应关系 object_points = [] image_points = [] for i, marker_id in enumerate(ids): if marker_id in self.world_map: # 获取标记在世界坐标系中的位置 T_wm = self.world_map[marker_id] marker_world_position = T_wm[:3, 3] # 获取标记在图像中的位置 # 这里需要标记的角点坐标 # 简化：使用标记中心 object_points.append(marker_world_position) image_points.append([320, 240]) # 图像中心，需要实际计算 if len(object_points) >= 3: # 使用solvePnP重新计算相机位置 object_points = np.array(object_points, dtype=np.float32) image_points = np.array(image_points, dtype=np.float32) success, rvec, tvec = cv2.solvePnP( object_points, image_points, self.K, self.D ) if success: self.camera_pose_in_world = self.make_matrix(rvec, tvec) 
6.3 虚拟物体放置
def place_virtual_object(self, frame, object_position_in_world): "" 在世界坐标系中放置虚拟物体 并投影到图像平面 "" if self.camera_pose_in_world is None: return # 虚拟物体在世界坐标系中的位置 # object_position_in_world: 4x4变换矩阵 # 计算虚拟物体相对于相机的变换 T_cw = np.linalg.inv(self.camera_pose_in_world) T_co = T_cw @ object_position_in_world # 提取旋转和平移 rvec, _ = cv2.Rodrigues(T_co[:3, :3]) tvec = T_co[:3, 3] # 绘制虚拟物体 # 这里可以绘制任何3D模型 cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1) # 可以绘制更复杂的3D模型 self.draw_3d_model(frame, rvec, tvec) 
7. 调试和验证
7.1 可视化世界地图
def visualize_world_map(self): ""可视化世界地图中所有标记的位置"" import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D fig = plt.figure() ax = fig.add_subplot(111, projection='3d') for marker_id, transform_matrix in self.world_map.items(): # 提取位置 position = transform_matrix[:3, 3] # 提取旋转 rotation_matrix = transform_matrix[:3, :3] # 绘制位置点 ax.scatter(position[0], position[1], position[2], label=f'Marker {marker_id}') # 绘制坐标轴方向 axis_length = 0.1 for i, color in enumerate(['r', 'g', 'b']): axis = rotation_matrix[:, i] * axis_length ax.quiver(position[0], position[1], position[2], axis[0], axis[1], axis[2], color=color, arrow_length_ratio=0.1) ax.set_xlabel('X (m)') ax.set_ylabel('Y (m)') ax.set_zlabel('Z (m)') ax.legend() ax.set_title('World Map Visualization') plt.show() 
7.2 精度验证
def verify_accuracy(self, test_markers): "" 验证系统精度 test_markers: 已知精确位置的测试标记 "" errors = [] for marker_id, true_position in test_markers.items(): if marker_id in self.world_map: estimated_position = self.world_map[marker_id][:3, 3] error = np.linalg.norm(estimated_position - true_position) errors.append(error) print(f"Marker {marker_id}: True={true_position}, " f"Est={estimated_position}, Error={error:.3f}m") if errors: mean_error = np.mean(errors) max_error = np.max(errors) print(f"Mean error: {mean_error:.3f}m") print(f"Max error: {max_error:.3f}m") return mean_error, max_error 
8. 性能优化
8.1 使用KD树加速搜索
from scipy.spatial import KDTree def build_spatial_index(self): ""构建空间索引，加速标记搜索"" positions = [] ids = [] for marker_id, transform_matrix in self.world_map.items(): position = transform_matrix[:3, 3] positions.append(position) ids.append(marker_id) if positions: self.kdtree = KDTree(positions) self.kdtree_ids = np.array(ids) else: self.kdtree = None 
8.2 增量式更新
def incremental_update(self, new_observations): "" 增量式更新世界地图 避免每次重新计算所有标记 "" for marker_id, new_transform in new_observations.items(): if marker_id in self.world_map: # 加权平均更新 old_transform = self.world_map[marker_id] weight = 0.2 # 新观测权重 updated_transform = (1 - weight) * old_transform + weight * new_transform self.world_map[marker_id] = updated_transform else: # 新增标记 self.world_map[marker_id] = new_transform 
这个系统提供了一个灵活、可扩展的世界空间AR框架，可以适应各种实际应用场景。
""" 


""" 
 1. 初始化（The Seed）
 * **理解**：把最初的 Marker 作为初始锚点。
 * **架构定义**：**建立全局坐标系 (Global Coordinate Frame)**。通过设定 ID 0 为 (0,0,0)，为原本虚无的 3D 空间打下了第一根“桩”。
2. 知识扩张（The Mapping）
 * **理解**：不断推算其他 Marker 的位姿并存入字典。
 * **架构定义**：**动态建图 (Online Mapping)**。
   * 当相机成为“桥梁”（同时看到 A 和 B），它就把 A 的已知身份传递给了 B。
   * **存储矩阵**的意义在于：它不仅存了位置，还存了**姿态（Orientation）**。
3. 循环检索与重定位（The Relocalization）
 * **理解**：通过 for 循环找字典，有已存在的就继续推算。
 * **架构定义**：**重定位与闭环检测 (Relocalization & Loop Closure)**。
   * 这个 for 循环其实就是系统在问：“在这个陌生的画面里，有没有我熟悉的‘老朋友’？”
   * 只要有一个“老朋友”被认出来，相机就能通过矩阵逆运算（np.linalg.inv）找回自己的位置。
### 这套逻辑在实际运行中的“接力棒”效应：
想象在走廊里贴了 ID 0, 1, 2, 3...
 1. **第一步**：看到 0 和 1，1 被激活了（存入字典）。
 2. **第二步**：往前走，0 消失了，但 1 还在。此时系统靠 1 维持世界坐标。
 3. **第三步**：看到 1 和 2，2 被激活了。
 4. **第四步**：即使走到了 100 米外，只要看到了 ID 50，而 50 是通过 49、48...一路接力过来的，依然能算回最初 ID 0 的位置。
### 这种架构的“优缺点”：
 * **优点（极其适合现在的研究）**：
   * **极简**：不需要处理复杂的点云，只处理 ID。
   * **稳定**：只要 Marker 不被撕掉，坐标关系就是永久的。
 * **挑战（你未来的优化方向）**：
   * **误差漂移 (Drift)**：矩阵相乘是有误差的。从 0 传到 1 有 1% 误差，传到 100 时，误差可能会让坐标飘出去好几米。
   * **静态假设**：如果有人把 ID 1 偷偷挪了位置，整个世界坐标系就会“分崩离析”。
""" 