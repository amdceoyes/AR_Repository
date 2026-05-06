import cv2
import numpy as np

class PnPSolver:
    def __init__(self, camera_matrix, dist_coeffs=None):
        # 这里的 K 是相机的内参（焦距、中心点）
        self.K = camera_matrix
        # D 是畸变参数，如果是普通摄像头，可以传 None 或者全 0
        self.D = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def solve(self, object_points, image_points):
        """
        [数据流]: 3D地图点 + 2D像素点 -> 6自由度位姿
        返回: 成功标志, 旋转向量(rvec), 平移向量(tvec)
        """
        if len(image_points) < 4:
            return False, None, None
        
        # 使用 RANSAC 算法排除误匹配的“脏数据”
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, 
            image_points, 
            self.K, 
            self.D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return success, rvec, tvec