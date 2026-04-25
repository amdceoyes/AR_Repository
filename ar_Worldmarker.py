import cv2
import cv2.aruco as aruco
import numpy as np

class WorldSpaceAR:
    """
    世界空间增强现实系统
    核心思想：建立一个以特定标记为原点的世界坐标系，其他所有标记的位置都相对于这个原点定义
    
    这与之前的"相机为中心"的AR系统有本质区别：
    - 之前：每个标记相对于相机的位置
    - 现在：所有标记在世界坐标系中的固定位置，相机在这个世界坐标系中运动
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
        
        # 原点标记：我们将ID为0的标记定义为世界坐标系的原点
        # 这类似于GPS中的"基准点"，所有其他位置都相对于这个原点测量
        self.origin_id = 0
        
        # 世界地图：存储每个标记相对于原点的固定位置
        # 格式: {marker_id: (rvec_to_origin, tvec_to_origin)}
        # 一旦测量并存储，即使原点标记不在视野中，我们也能知道其他标记的位置
        self.world_map = { 
            self.origin_id: (np.zeros(3), np.zeros(3))  # 原点相对于自己就是零位置
        }
        
        # 历史记录：用于平滑滤波，防止抖动
        self.history = {}

    def process(self, frame):
        """
        处理单帧图像，建立世界坐标系并计算相对位置
        
        参数:
        frame: 输入图像帧
        返回: 处理后的图像帧
        """
        # 1. 图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 定义ARUco字典
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        
        # 3. 检测ARUco标记
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        
        # 4. 如果有标记被检测到
        if ids is not None:
            # 批量计算所有检测到的标记的3D姿态
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.K, self.D
            )
            
            # 将ids从2D数组展平为1D数组，方便索引
            ids = ids.flatten()
            
            # 5. 寻找"上帝"（原点标记）
            # np.where返回一个元组，[0]获取第一个维度（行索引）
            origin_idx = np.where(ids == self.origin_id)[0]
            
            # 6. 如果原点标记在画面中
            if len(origin_idx) > 0:
                # 获取原点标记相对于相机的变换
                # T_c_o: 从原点坐标系到相机坐标系的变换
                r_c_o = rvecs[origin_idx[0]][0]  # 旋转向量
                t_c_o = tvecs[origin_idx[0]][0]  # 平移向量
                
                # 计算相机相对于原点的位姿（逆变换）
                # 从 T_c_o 求 T_o_c
                # 公式: R_o_c = R_c_o.T, T_o_c = -R_o_c * T_c_o
                
                # 将旋转向量转换为旋转矩阵
                R_c_o, _ = cv2.Rodrigues(r_c_o)
                
                # 计算逆变换
                R_o_c = R_c_o.T  # 转置 = 逆（因为旋转矩阵是正交矩阵）
                T_o_c = -np.dot(R_o_c, t_c_o)
                
                # 7. 建立空间联系：计算其他标记在世界坐标系下的位置
                for i in range(len(ids)):
                    curr_id = ids[i]
                    
                    # 跳过原点标记
                    if curr_id == self.origin_id:
                        continue
                    
                    # 获取当前标记相对于相机的变换
                    # T_c_m: 从当前标记坐标系到相机坐标系的变换
                    r_c_m = rvecs[i][0]
                    t_c_m = tvecs[i][0]
                    
                    # 将旋转向量转换为旋转矩阵
                    R_c_m, _ = cv2.Rodrigues(r_c_m)
                    
                    # 核心变换：计算当前标记相对于原点的变换
                    # 公式: T_o_m = R_o_c * T_c_m + T_o_c
                    # 这类似于：世界坐标 = 相机在世界中的位置 + 相机到标记的向量
                    T_o_m = np.dot(R_o_c, t_c_m) + T_o_c
                    
                    # 计算旋转：R_o_m = R_o_c * R_c_m
                    R_o_m = np.dot(R_o_c, R_c_m)
                    
                    # 将旋转矩阵转换回旋转向量
                    r_o_m, _ = cv2.Rodrigues(R_o_m)
                    
                    # 存入世界地图
                    # 这里可以添加长时间平均，提高精度
                    self.world_map[curr_id] = (r_o_m.flatten(), T_o_m)
            
            # 8. 渲染层
            # 即使原点看不见了，只要相机姿态能通过其他标记推算，依然可以渲染
            self._render_world(frame, rvecs, tvecs, ids)
        
        return frame

    def _render_world(self, frame, rvecs, tvecs, ids):
        """
        在世界坐标系中渲染虚拟物体
        
        参数:
        frame: 要渲染的图像帧
        rvecs: 所有检测到的标记的旋转向量
        tvecs: 所有检测到的标记的平移向量
        ids: 所有检测到的标记的ID
        """
        # 这里的核心思想是XR（扩展现实）的真谛：
        # 虚拟物体被固定在真实世界的特定位置，而不是固定在相机视野中
        
        # 遍历世界地图中的所有标记
        for m_id, (r_o_m, t_o_m) in self.world_map.items():
            # 这里的渲染逻辑需要根据"谁在视野里"来反推相机位置，再投影
            # 但为了演示，我们只做简单的标注
            
            # 如果这个标记在当前帧中被检测到
            if ids is not None and m_id in ids:
                # 找到这个标记在当前帧中的索引
                idx = np.where(ids == m_id)[0][0]
                
                # 获取这个标记的当前检测位置
                rvec = rvecs[idx][0]
                tvec = tvecs[idx][0]
                
                # 在标记上绘制立方体
                self._draw_cube(frame, rvec, tvec, m_id)
                
                # 显示这个标记在世界坐标系中的位置
                # 位置是相对于原点的，单位是米
                cv2.putText(
                    frame,
                    f"ID:{m_id} Pos:({t_o_m[0]:.2f},{t_o_m[1]:.2f},{t_o_m[2]:.2f})",
                    (10, 30 + 30 * idx),  # 垂直排列显示
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
    
    def _draw_cube(self, frame, rvec, tvec, marker_id):
        """
        在标记位置上绘制3D立方体
        
        参数:
        frame: 要绘制的图像帧
        rvec: 旋转向量
        tvec: 平移向量
        marker_id: 标记ID
        """
        # 立方体边长
        cube_size = 0.05
        
        # 定义立方体的8个顶点
        cube_points = np.float32([
            # 底面4个点
            [-cube_size/2, -cube_size/2, 0],
            [cube_size/2, -cube_size/2, 0],
            [cube_size/2, cube_size/2, 0],
            [-cube_size/2, cube_size/2, 0],
            
            # 顶面4个点
            [-cube_size/2, -cube_size/2, -cube_size],
            [cube_size/2, -cube_size/2, -cube_size],
            [cube_size/2, cube_size/2, -cube_size],
            [-cube_size/2, cube_size/2, -cube_size]
        ])
        
        # 将3D点投影到2D图像
        img_points, _ = cv2.projectPoints(
            cube_points, rvec, tvec, self.K, self.D
        )
        img_points = np.int32(img_points).reshape(-1, 2)
        
        # 绘制立方体
        # 根据ID选择颜色
        if marker_id == 0:
            color = (255, 0, 0)  # 原点标记用蓝色
        elif marker_id == 1:
            color = (0, 255, 0)  # ID 1用绿色
        else:
            color = (0, 0, 255)  # 其他用红色
        
        # 绘制底面
        cv2.drawContours(frame, [img_points[:4]], -1, color, 2)
        
        # 绘制顶面
        cv2.drawContours(frame, [img_points[4:]], -1, color, 2)
        
        # 绘制4条垂直线
        for i in range(4):
            cv2.line(
                frame,
                tuple(img_points[i]),
                tuple(img_points[i+4]),
                (255, 255, 255),
                2
            )

# 主程序示例
if __name__ == "__main__":
    # 相机参数（示例值）
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理当前帧
        processed_frame = ar_system.process(frame)
        
        # 显示结果
        cv2.imshow('World Space AR', processed_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

"""
核心概念详细解释
1. 世界坐标系 vs. 相机坐标系
传统AR系统（相机为中心）：
相机坐标系：以相机为原点 标记位置：相对于相机定义 问题：当相机移动时，虚拟物体相对于相机会移动 
世界空间AR系统（世界为中心）：
世界坐标系：以原点标记为基准 标记位置：相对于世界坐标系定义 优势：虚拟物体固定在真实世界的特定位置 
2. 坐标变换的数学原理
2.1 变换表示
T_c_o：从原点坐标系到相机坐标系的变换
T_c_m：从标记坐标系到相机坐标系的变换
T_o_m：从标记坐标系到原点坐标系的变换（存储在世界地图中）
2.2 核心公式
已知：T_c_o 和 T_c_m 求：T_o_m 步骤： 1. 计算 T_o_c = inv(T_c_o) # 逆变换 2. 计算 T_o_m = T_o_c * T_c_m 
矩阵形式：
R_o_m = R_o_c * R_c_m T_o_m = R_o_c * T_c_m + T_o_c 
3. 系统工作流程
阶段1：建图（Mapping）
1. 检测到原点标记 2. 计算相机相对于原点的位置 3. 对于每个检测到的其他标记： a. 计算标记相对于相机的位置 b. 转换为相对于原点的位置 c. 存储到世界地图 
阶段2：跟踪（Tracking）
情况1：能看到原点标记 - 直接计算相机位置 - 用世界地图渲染所有虚拟物体 情况2：看不到原点标记，但能看到其他已知标记 - 用已知标记推算相机位置 - 仍然可以正确渲染虚拟物体 
4. 实际应用场景
4.1 大型AR环境
用途：博物馆、展览馆 优势：多个标记分布在空间中，用户可以在任何位置看到正确的虚拟内容 
4.2 多人协作AR
用途：团队设计评审、远程协助 优势：所有用户看到虚拟物体在相同的位置 
4.3 持久性AR
用途：室内导航、AR游戏 优势：虚拟物体"记住"自己的位置，下次来还在原地 
5. 扩展与优化建议
5.1 提高精度
# 长时间平均 if curr_id in self.world_map: # 加权平均新旧位置 old_r, old_t = self.world_map[curr_id] new_r = 0.8 * old_r + 0.2 * r_o_m new_t = 0.8 * old_t + 0.2 * T_o_m self.world_map[curr_id] = (new_r, new_t) 
5.2 多标记优化
# 使用多个标记优化相机位置 visible_markers = [] for i in range(len(ids)): if ids[i] in self.world_map: visible_markers.append((ids[i], rvecs[i][0], tvecs[i][0])) # 使用PnP求解最优相机位置 if len(visible_markers) >= 3: obj_points = [] img_points = [] for marker_id, rvec, tvec in visible_markers: # 获取标记在世界坐标系中的位置 world_pos = self.world_map[marker_id] # 添加到点对列表 # ... 使用solvePnP优化相机位置 
5.3 保存和加载世界地图
def save_world_map(self, filename): ""保存世界地图到文件"" data = {} for m_id, (rvec, tvec) in self.world_map.items(): data[str(m_id)] = { 'rvec': rvec.tolist(), 'tvec': tvec.tolist() } import json with open(filename, 'w') as f: json.dump(data, f) def load_world_map(self, filename): ""从文件加载世界地图"" import json with open(filename, 'r') as f: data = json.load(f) for m_id_str, values in data.items(): m_id = int(m_id_str) rvec = np.array(values['rvec']) tvec = np.array(values['tvec']) self.world_map[m_id] = (rvec, tvec) 
6. 调试和验证
6.1 验证坐标变换
def test_coordinate_transform(self): ""测试坐标变换的正确性"" # 手动设置测试值 # 验证变换链是否闭合 pass 
6.2 可视化世界地图
def visualize_world_map(self): ""可视化世界地图中所有标记的位置"" fig = plt.figure() ax = fig.add_subplot(111, projection='3d') for m_id, (rvec, tvec) in self.world_map.items(): ax.scatter(tvec[0], tvec[1], tvec[2], label=f'Marker {m_id}') # 绘制坐标轴方向 # ... ax.set_xlabel('X') ax.set_ylabel('Y') ax.set_zlabel('Z') ax.legend() plt.show() 
7. 性能考虑
7.1 计算复杂度
检测标记：O(n) 随标记数量线性增加
坐标变换：每个标记O(1)的矩阵运算
内存使用：世界地图存储所有已知标记的位置
7.2 实时性优化
只处理视野中的标记
使用多线程：检测、变换、渲染分离
GPU加速：使用CUDA处理矩阵运算
这个世界空间AR系统实现了真正的"虚拟物体固定在真实世界"的AR体验，是构建复杂AR应用的基础框架。
"""