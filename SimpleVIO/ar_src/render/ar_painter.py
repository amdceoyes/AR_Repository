import cv2
import numpy as np

class ARPainter:
    def __init__(self, camera_matrix, dist_coeffs=None):
        self.K = camera_matrix
        self.D = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def draw_cube(self, image, rvec, tvec):
        """
        在给定的图像上绘制一个 3D 立方体
        """
        # 定义一个 3D 立方体的 8 个顶点（假设边长为 0.1 单位，比如 10cm）
        axis = np.float32([
            [0,0,0], [0.1,0,0], [0.1,0.1,0], [0,0.1,0],
            [0,0,-0.1], [0.1,0,-0.1], [0.1,0.1,-0.1], [0,0.1,-0.1]
        ])

        # [数据流]: 3D 顶点 + 位姿 (R, T) -> 2D 像素坐标
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, self.D)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # 绘制底面
        image = cv2.drawContours(image, [imgpts[:4]], -1, (0, 255, 0), 3)
        # 绘制支柱
        for i, j in zip(range(4), range(4, 8)):
            image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # 绘制顶面
        image = cv2.drawContours(image, [imgpts[4:]], -1, (0, 0, 255), 3)

        return image

    def draw_status(self, image, state_name):
        """
        在左上角显示系统当前的状态（就是你刚才困惑的状态机名字）
        """
        color = (0, 255, 0) if state_name == "TRACKING" else (0, 0, 255)
        cv2.putText(image, f"STATE: {state_name}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image