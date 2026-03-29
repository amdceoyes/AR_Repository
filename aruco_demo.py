import cv2
import numpy as np
import sys

# 1. 初始化检测器
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 2. 估计相机参数 (防止轴乱飞)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))

# 3. 尝试开启摄像头 (DELL 优化版)
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    cap = cv2.VideoCapture(1) # 如果 0 不行就试 1

# 强制设置小分辨率，解决卡顿
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 正在唤醒摄像头，请稍候...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # 检测 Marker
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # 画边框
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # 计算并画出 3D 轴
        for i in range(len(ids)):
            obj_points = np.array([[-0.02, 0.02, 0], [0.02, 0.02, 0], 
                                  [0.02, -0.02, 0], [-0.02, -0.02, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

    # 显示画面
    cv2.imshow('SportCross AR - Press Q to Quit', frame)

    # 这里的 30ms 是解决卡顿的关键，给系统 UI 留出反应时间
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()