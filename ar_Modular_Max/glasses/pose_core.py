# 导入必要的Python库
import cv2    # OpenCV库，用于计算机视觉和PnP求解
import numpy as np  # NumPy库，用于数值计算和矩阵操作

class PoseCore:
    """
    位姿核心类，负责通过PnP算法求解相机在3D空间中的位姿
    
    PnP（Perspective-n-Point）是计算机视觉中的经典问题：
    给定3D世界坐标点和对应的2D图像像素坐标，求解相机的位置和朝向
    
    这个类主要用于：
    1. AR系统的相机定位
    2. 机器人导航
    3. 三维重建
    4. 增强现实
    """
    
    def __init__(self, camera_matrix):
        """
        初始化位姿核心
        
        参数:
        camera_matrix: 相机内参矩阵，需要通过相机标定获取
                       3x3矩阵，格式为：
                       [[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]]
                       其中：
                       - fx, fy: 焦距（像素单位）
                       - cx, cy: 主点（图像中心）
        
        相机内参是相机的固有属性，不随相机移动而改变
        它描述了3D点到2D像素的投影关系
        """
        # 相机内参矩阵
        self.camera_matrix = camera_matrix
        
        # 畸变系数（通常假设为0，如果镜头畸变严重需要标定）
        # 畸变系数通常有5个参数：k1, k2, p1, p2, k3
        # 这里初始化为0，表示不考虑镜头畸变
        self.dist_coeffs = np.zeros((5, 1))
        
        # 打印初始化信息
        print("[Pose] PoseCore 已启动，视觉位姿解算模块就绪。")
        print(f"[Pose] 相机内参矩阵: \n{self.camera_matrix}")

    def solve_pose(self, object_points, image_points):
        """
        核心解算：根据世界坐标系中的点(object_points)和图像中的点(image_points)求解位姿
        
        参数:
        object_points: 3D世界坐标点，形状为(N,3)或(N,1,3)，N>=4
                       N是点的数量，每个点是[x, y, z]坐标
                       example: [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        image_points: 对应的2D图像像素坐标，形状为(N,2)或(N,1,2)
                      N是点的数量，每个点是[u, v]像素坐标
                      example: [[100, 200], [150, 200], [100, 250], [150, 250]]
        
        返回:
        4x4位姿矩阵，表示相机在世界坐标系中的位置和朝向
        如果解算失败，返回None
        
        这个函数使用OpenCV的solvePnP算法，它是解决PnP问题的标准方法
        PnP问题在AR中非常关键：将虚拟物体"钉"在真实世界中的特定位置
        """
        # 使用PnP算法求解相机位姿
        # cv2.solvePnP: 解决透视n点问题
        # 参数说明：
        #   object_points: 3D世界坐标点
        #   image_points: 对应的2D图像坐标点
        #   self.camera_matrix: 相机内参矩阵
        #   self.dist_coeffs: 镜头畸变系数
        #   flags=cv2.SOLVEPNP_ITERATIVE: 使用迭代法求解（最常用）
        # 返回值：
        #   success: 是否成功求解
        #   rvec: 旋转向量（3x1），表示相机朝向
        #   tvec: 平移向量（3x1），表示相机位置
        
        # 注意：object_points和image_points需要有相同的点数（N>=4）
        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # 如果求解成功
        if success:
            # 将旋转向量转换为旋转矩阵（3x3）
            # 旋转向量是旋转的紧凑表示（轴角表示法）
            # 旋转矩阵是标准的3x3正交矩阵
            R, _ = cv2.Rodrigues(rvec)
            
            # 将旋转矩阵和平移向量格式化为4x4变换矩阵
            return self._format_pose(R, tvec)
        else:
            # 求解失败，打印警告
            print("[Pose] 警告：位姿解算失败")
            return None

    def _format_pose(self, R, t):
        """
        将旋转矩阵R和平移向量t合并为标准4x4变换矩阵
        
        参数:
        R: 3x3旋转矩阵
        t: 3x1平移向量
        
        返回:
        4x4齐次变换矩阵
        
        齐次变换矩阵格式：
        [[R00, R01, R02, tx],
         [R10, R11, R12, ty],
         [R20, R21, R22, tz],
         [ 0,   0,   0,   1]]
        
        这个矩阵表示从世界坐标系到相机坐标系的变换
        在AR中，我们通常需要从相机坐标系到世界坐标系的变换
        所以实际使用时可能需要求逆
        """
        # 创建4x4单位矩阵
        pose_matrix = np.eye(4)
        
        # 将3x3旋转矩阵赋值到左上角
        pose_matrix[:3, :3] = R
        
        # 将平移向量赋值到右上角
        # t是3x1的列向量，flatten()将其变为一维数组(3,)
        pose_matrix[:3, 3] = t.flatten()
        
        return pose_matrix