import cv2
import numpy as np
from ar_src.vision.extractor import ORBExtractor
from ar_src.pose.pnp_solver import PnPSolver
from ar_src.render.ar_painter import ARPainter

def main():
    extractor = ORBExtractor(n_features=1000)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    solver = PnPSolver(camera_matrix=K)
    painter = ARPainter(camera_matrix=K)

    # 预定义的 3D 立方体底座
    obj_points = np.float32([[0,0,0], [0.1,0,0], [0.1,0.1,0], [0,0.1,0]])
    
    # --- 关键：记忆模块 ---
    anchor_des = None       # 锚点描述子
    anchor_kps = None       # 锚点特征点位置
    last_rvec, last_tvec = None, None
    alpha = 0.2 

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        kps, des = extractor.extract(frame)

        # 1. 自动捕获第一个稳定的平面作为“锚点”
        if anchor_des is None:
            if kps is not None and len(kps) > 200:
                anchor_des = des
                anchor_kps = kps
                print(">>> 锚点已锁定！坐标系已固定。")
            cv2.imshow("AR", display_frame)
            continue

        # 2. 核心：通过 Match 寻找那些“老熟人”
        matches = extractor.match(anchor_des, des)
        
        if len(matches) > 50:
            # 3. 找回锚点建立时的坐标变换 (Homography)
            src_pts = np.float32([anchor_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # 4. 【解决滚动的终极方案】：
                # 我们不再取当前帧的随机点，而是取锚点帧的四个虚拟角，投射到当前帧
                # 这样 img_pts 的顺序永远是固定的！
                h, w = frame.shape[:2]
                # 定义锚点帧中的虚拟参考位置（比如屏幕中心的一个小区域）
                ref_corners = np.float32([[w/2-50, h/2-50], [w/2+50, h/2-50], 
                                         [w/2+50, h/2+50], [w/2-50, h/2+50]]).reshape(-1, 1, 2)
                
                img_pts = cv2.perspectiveTransform(ref_corners, M).reshape(4, 2)
                
                ok, rvec, tvec = solver.solve(obj_points, img_pts)
                
                if ok:
                    # 5. 平滑滤波
                    if last_rvec is not None:
                        rvec = alpha * rvec + (1 - alpha) * last_rvec
                        tvec = alpha * tvec + (1 - alpha) * last_tvec
                    last_rvec, last_tvec = rvec.copy(), tvec.copy()
                    
                    display_frame = painter.draw_cube(display_frame, rvec, tvec)

        cv2.imshow("AR", display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
    第一部分：之前错误（滚来滚去/面筋人）的核心原因
你之前的直觉是“提取的点越多越稳”，这在增加纹理对齐上是对的，但在位姿估计上却踩了一个大坑：忽略了“特征点的拓扑一致性”。
1. 致命的随机排序
之前代码最严重的错误在于这一行：
img_pts = np.float32([kp.pt for kp in plane_kps[:4]])
OpenCV 提取出的特征点集合（plane_kps）是一个无序列表，它通常是按照特征点的响应强度（ORB score）排序的。
现象：第一帧里，你告诉 Solver ，“这 1000 个点里最强的 4 个点，分别是虚拟世界的左上、右上、右下、左下角”。Solver 愉快地算出了位姿。
灾难：第二帧，因为光线变化或相机微小的移动，原本最强的 4 个点里的某一个变弱了，排到了第 5 位。原本的第 5 位跳到了前 4 位。
结果（Data Association Fail）：此时，你传给 Solver 的前 4 个点，虽然它们都在平面内，但它们在物理世界中的具体位置已经发生了互换或偏移。但 Solver 不知道，它依然按照固定的“四个角”逻辑去拟合。数学为了迎合这种错误的对应关系，解算出的 R（旋转矩阵）就会发生剧烈的扭曲和突变。反映在屏幕上，就是方块在滚，或者像面筋一样甩动。
2. “朝生暮死”的参考系
在那一版代码里，我们没有引入 Mapping 或 Anchor（锚点）的概念。
错误根源：每一帧都在重新定义“哪里是原点”。系统是“金鱼脑子”，完全没有上一帧的记忆。每一帧都在根据这一帧随机取出的前 4 个点试图建立一个全新的 3D 坐标系。
结果：因为没有统一、锁死的“世界坐标系”，每一帧算出来的坐标系轴向都在无规律抖动。
第二部分：我是怎么解决的（技术手段）
解决“滚动”的核心不是增加点的数量，而是**“固定点对在时间维度上的拓扑结构”**。
手段一：首帧锚定（Anchor Initialization）
我在代码里增加了一个判断：if anchor_des is None:。
解决方式：让系统在检测到足够多特征点的第一帧，强制把这 1000 个点记下来，作为整个项目的“圣经”（基准）。
作用：这就像给这 1000 人发了固定的“身份证号”。
手段二：描述子匹配（Descriptor Matching）
后续所有帧不再是从 1000 个点里随机抓，而是通过 extractor.match，用新画面里的描述子和锚点帧里的“身份证号”去一一比对。
解决方式：不管点在图像上怎么移动，只要“长相”（描述子）一样，我们就能确保这一帧拿到的第 i 号点，就是锚点帧里的那个第 i 号点。
作用：找回了点的拓扑顺序。这是 PnP 能够稳定解算的最根本前提。
手段三：单应性辅助（Homography Assistance）
为了进一步加强稳定性，我没有直接用匹配到的点去解 PnP。
解决方式：先用所有匹配点（比如 100 个）去算出平面的整体单应性变换 M（利用 RANSAC 剔除噪点），然后用 M 去投影一个顺序绝对不会出错的、虚拟的参考正方形 (ref_corners)。
作用：这为 Solver 提供了顺序严格正确、分布均匀、且极具连续性的 2D 特征点对。
第三部分：你以后在 SLAM 开发中需要注意什么
这是一次非常宝贵的经验。以下四点是你以后在写相关算法时必须注意的命门：
1. 数据关联（Data Association）是 SLAM 的灵魂
提取特征（Extraction）和计算描述子（Description）只是基础。如果你不能在下一帧、下十帧、甚至把相机转一圈再转回来时识别出同一个点，那么所有的点都是废纸。没有稳定的数据关联（匹配），就没有稳定的 SLAM。
2. 不要让 PnP 直接裸奔
纯粹的 R 和 T 对噪声极其敏感。你以后一定要把滤波加上：
简单的滤波：咱们用的 alpha * new + (1-alpha) * old 低通滤波，是成本最低的减震器。
高级的滤波：卡尔曼滤波（Kalman Filter）或者滑窗束调整（Sliding Window BA），它们不仅考虑上一帧，还考虑相机的运动模型。
3. 慎重处理单目尺度的二义性
单目相机其实不知道“10cm”是多远。在焦距 K 和物理物体大小对不上时，系统会产生多解歧义，导致方块上下颠倒或翻转。
注意：在做无参考定位时，一定要通过检测特定物体（如人脸、地板高度）或者使用 IMU 来获得尺度（Scale）。
4. 不要轻易挑战“无记忆追踪”
这种每帧都是“第一次见面”的追踪方式，只能用在极极少数特定的图像配准（Registration）任务中。对于凡是涉及到物体在屏幕上需要长期“吸附”的 AR 任务，必须建立局部的 Map（点云数据库）和关键帧机制，以此作为坐标系的锚点。
'''