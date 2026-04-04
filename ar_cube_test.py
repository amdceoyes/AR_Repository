import cv2#OpenCV计算机视觉库
import cv2.aruco as aruco#OpenCV的ArUco标记模块，用于检测和处理二维码标记
import numpy as np#科学计算库，处理数组和矩阵运算
import time

# --- 核心修改：如果没有 npy 文件，我们先用通用参数 ---
# 这是一个模拟的相机矩阵，虽然不完美，但能让立方体画出来
camera_matrix = np.array([[800, 0, 320],
                         [0, 800, 240],
                         [0, 0, 1]], dtype=np.float32)
#相机参数（内参矩阵）
#camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32) 
#相机内参矩阵的格式：
#[800, 0, 320]: 第一行
#800: 焦距（fx），以像素为单位
#0: 倾斜系数（通常为0）
#320: 主点x坐标（cx），通常是图像宽度的一半
#[0, 800, 240]: 第二行
#800: 焦距（fy），以像素为单位
#240: 主点y坐标（cy），通常是图像高度的一半
#[0, 0, 1]: 第三行，齐次坐标转换
#注意：这里的数值是假设的，实际应用需要通过相机标定获得准确的内参。

# 畸变系数设为 0
#畸变系数，设为0表示忽略镜头畸变。实际相机通常有径向和切向畸变。
dist_coeffs = np.zeros((5, 1))

# --- 定义 ArUco 配置 ---
#创建ArUco字典，使用6×6的标记，共有250个不同的标记ID。6×6表示每个标记由6×6的二进制网格组成。
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
#创建ArUco检测器的参数对象，使用默认检测参数。
parameters = aruco.DetectorParameters()

# 🚀 插入这一行：开启亚像素精细化，彻底干掉跳动
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX


# --- 定义立方体 (5cm) ---
size = 0.05 # 立方体边长0.05米（5厘米）
s = size / 2# 半边长 
cube_points = np.float32([
    [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],   # 底面4个顶点    # 底面
    [-s, -s, size], [s, -s, size], [s, s, size], [-s, s, size]     # 顶面4个顶点    # 顶面
])
#定义立方体的8个顶点（世界坐标）：
#坐标原点在标记中心
#前4个点：底面，z=0（在标记平面上）
#后4个点：顶面，z=-size（在标记上方，因为OpenCV坐标系z轴指向相机前方）

cap = cv2.VideoCapture(0)#打开默认摄像头（索引0）
# --- 新增：平滑处理需要的变量 ---
prev_rvec = None
prev_tvec = None
alpha = 0.4  # 平滑系数：0.1很稳但有延迟，0.5很灵敏但较抖。建议先用0.2。

prev_time = 0
while True:#循环读取视频帧，ret表示是否成功读取
    ret, frame = cap.read()
    if not ret: break

    #下面这部分代码块是为了检测采样率的
   # --- 【新增】FPS 计算逻辑 ---
    curr_time = time.time()
    # 这一行计算两帧之间的时间差并转为频率
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    # --- 【新增】在画面左上角显示 FPS ---
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转换为灰度图像，因为ArUco检测在灰度图上更高效
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#检测ArUco标记：
#corners: 检测到的标记四个角点的像素坐标（列表，每个标记4个点）
#ids: 检测到的标记ID（列表）
#rejected: 被拒绝的候选标记

# ... 你的 detectMarkers 代码 ...
#corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:#这个是用于打印角检测精度的
        # 👉 打印在这里！打印第一个检测到的 marker 的角点
        print("当前角点坐标：\n", corners[0]) 
        
        # ... 后面是你原本的 estimatePose 等代码 ...


    if ids is not None:
        # 估计姿态 (假设你的 marker 长度也是 0.05m)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
#估计每个标记的6自由度姿态：
#rvecs: 旋转向量（3×1），表示标记相对于相机的旋转
#tvecs: 平移向量（3×1），表示标记相对于相机的位置
#0.05: 标记的实际边长（米），必须与使用的物理标记一致


        for i in range(len(ids)):
            # 1. 获取当前帧原始位姿
            curr_rvec = rvecs[i]
            curr_tvec = tvecs[i]

            # 2. 核心平滑逻辑 (减震器)
            if prev_rvec is None:
                prev_rvec, prev_tvec = curr_rvec, curr_tvec
            else:
                prev_rvec = alpha * curr_rvec + (1 - alpha) * prev_rvec
                prev_tvec = alpha * curr_tvec + (1 - alpha) * prev_tvec

            # 3. 使用平滑后的位姿进行投影
            imgpts, _ = cv2.projectPoints(cube_points, prev_rvec, prev_tvec, camera_matrix, dist_coeffs)
            #cv2.projectPoints()是核心函数：
            #输入：3D点、旋转向量、平移向量、相机参数
            #输出：对应的2D图像坐标
            #实现了3D到2D的透视投影变换
            #imgpts = np.int32(imgpts).reshape(-1, 2)
            #将浮点数坐标转换为整数，并重塑为(8,2)数组，便于绘图
            

                        # --- 从这里开始替换 ---
            # 1. 把投影出来的点转为整数并去掉多余维度
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # 2. 定义立方体的 6 个面 (每组 4 个点的索引)
            # 索引说明: 0-3 是底面, 4-7 是顶面
            faces = [
                [imgpts[0], imgpts[1], imgpts[2], imgpts[3]], # 底面
                [imgpts[4], imgpts[5], imgpts[6], imgpts[7]], # 顶面
                [imgpts[0], imgpts[1], imgpts[5], imgpts[4]], # 侧面1
                [imgpts[1], imgpts[2], imgpts[6], imgpts[5]], # 侧面2
                [imgpts[2], imgpts[3], imgpts[7], imgpts[6]], # 侧面3
                [imgpts[3], imgpts[0], imgpts[4], imgpts[7]]  # 侧面4
            ]

            # 3. 创建一个覆盖层用于实现半透明 (让 AR 看起来更自然)
            overlay = frame.copy()

            # 4. 给面涂上不同的颜色 (B, G, R)
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

            for i, face in enumerate(faces):
                # 填充颜色
                cv2.fillConvexPoly(overlay, np.array(face), colors[i])

            # 5. 合并原图和覆盖层 (0.6是方块不透明度)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            # --- 替换结束 ---


            # 连线画立方体
            #cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2) # 1. 绘制底面（绿色矩形）imgpts[:4]是底面的4个点，绘制为绿色轮廓
            #for k in range(4):
            #    cv2.line(frame, tuple(imgpts[k]), tuple(imgpts[k+4]), (255, 0, 0), 2) # 立柱, 绘制4条垂直边（蓝色）连接底面和顶面对应的点，形成立方体的垂直边
            #cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2) # 绘制顶面（红色矩形）imgpts[4:]是顶面的4个点，绘制为红色轮廓

    cv2.imshow('AR Cube Test - Lab', frame)#显示带立方体的AR图像，按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()#释放摄像头资源，关闭所有OpenCV窗口