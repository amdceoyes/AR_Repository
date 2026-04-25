import cv2
import cv2.aruco as aruco
import numpy as np
import time
#cv2 : OpenCV，计算机视觉库
#aruco : OpenCV中的ARUco模块，专门处理AR标记
#numpy : 数学计算库
#time : 时间相关功能


#这个代码包含了：亚像素增强、多 ID 字典平滑、FPS 显示、以及多 Marker 独立涂色
#整体流程：摄像头读取 → 检测ARUco标记 → 计算3D位置 → 绘制立方体 → 显示画面

# 1. 基础配置（请确保这里的矩阵是你之前标定或填写的那个）
#相机内参矩阵：描述相机自身的特性
#这个矩阵告诉程序"相机如何看世界"
camera_matrix = np.array([[650, 0, 320], [0, 650, 240], [0, 0, 1]], dtype=float)
#800: 焦距（fx, fy），控制视野大小
#320, 240: 主点（cx, cy），图像中心点
#单位：像素
dist_coeffs = np.zeros((4, 1))
#畸变系数：修正镜头畸变
#这里全是0，表示"假设镜头是完美的，没有畸变"
#真实相机需要标定得到真实值


# 定义 5cm 的立方体顶点
size = 0.05
half = size / 2
cube_points = np.float32([
     [-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0], # 底面四点，中心在(0,0)
    [-half, -half, -size], [half, -half, -size], [half, half, -size], [-half, half, -size] # 顶面四点
    #[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
    #[0, 0, -size], [size, 0, -size], [size, size, -size], [0, size, -size]
])
#定义3D立方体的8个顶点坐标
#单位：米
#立方体边长5厘米（0.05米）
#坐标顺序：4个底面点 + 4个顶面点

#底面4个点：     顶面4个点（相对底面下移5cm）：
#(0,0,0)        (0,0,-0.05)
#(0.05,0,0)     (0.05,0,-0.05)
#(0.05,0.05,0)  (0.05,0.05,-0.05)
#(0,0.05,0)     (0,0.05,-0.05)



# 2. 初始化检测器
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#选择ARUco字典：决定使用哪种类型的标记
# DICT_4X4_50 : 4x4网格，有50种不同的标记
#你可以打印不同的标记图案

parameters = aruco.DetectorParameters()
# 开启亚像素精细化，干掉抖动
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# 3. 初始化多 Marker 记忆字典
marker_history = {} 
alpha = 0.2  # 平滑系数
prev_time = 0
#历史记录：存储每个标记之前的位置
#用于位置平滑，减少抖动
# alpha=0.2 : 新位置权重20%，旧位置80%
#相当于"惯性"效果


cap = cv2.VideoCapture(0)
#打开摄像头
#0表示默认摄像头
#可以改成1、2等使用其他摄像头


while True:
    ret, frame = cap.read()
    if not ret: break
#主循环：不断读取摄像头画面
# ret : 是否成功读取
# frame : 当前帧的图像
#如果读取失败就退出循环


    # 计算 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
#计算帧率：每秒处理多少帧
#计算两次循环的时间差
#帧率 = 1 / 时间间隔
#用于性能监控


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #转换为灰度图
    #ARUco检测只需要黑白信息
    #灰度图处理更快

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #检测ARUco标记：核心步骤
    #输入：灰度图像
    #输出：
    # corners : 每个标记的4个角点坐标
    # ids : 每个标记的ID编号
    # _ : 拒绝的标记（不需要）

    if ids is not None:
        # 批量解算位姿
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, size, camera_matrix, dist_coeffs)
        #如果有标记被检测到：
        #计算每个标记的3D位姿
        # rvecs : 旋转向量（3个值，表示方向）
        # tvecs : 平移向量（3个值，表示位置）
        #单位：米

        # 🚀 核心：遍历每一个检测到的标记，获取当前标记的ID，旋转，平移
        for i in range(len(ids)):
            curr_id = ids[i][0]
            curr_rvec = rvecs[i][0]
            curr_tvec = tvecs[i][0]

            # 独立的平滑记忆
            if curr_id not in marker_history:
                marker_history[curr_id] = [curr_rvec, curr_tvec]
            else:
                p_rvec, p_tvec = marker_history[curr_id]
                marker_history[curr_id][0] = alpha * curr_rvec + (1 - alpha) * p_rvec
                marker_history[curr_id][1] = alpha * curr_tvec + (1 - alpha) * p_tvec
                #位置平滑：减少抖动
                #如果是新标记：直接记录
                #如果是旧标记：新旧位置加权平均
                #公式： 新平滑位置 = 0.2*当前位置 + 0.8*上次位置 

            # 取出平滑后的位姿（旋转和平移）
            s_rvec, s_tvec = marker_history[curr_id]

            # 投影 3D 点到 2D 图像
            imgpts, _ = cv2.projectPoints(cube_points, s_rvec, s_tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            #3D到2D投影：关键步骤！
            #将立方体的8个3D点（世界坐标）投影到2D图像
            #结果 imgpts 是8个像素坐标
            #这样就可以在图片上画立方体了

            # --- 为不同 ID 设置不同颜色 ---
            # ID 0 用蓝色，ID 1 用橘色，其他用绿色
            if curr_id == 0:
                color = (255, 100, 0)
            elif curr_id == 1:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
                #不同标记用不同颜色
                #OpenCV颜色是BGR格式：
                #蓝色：(255, 0, 0)
                #绿色：(0, 255, 0)
                #红色：(0, 0, 255)
                #这里做了自定义

            # 绘制底面和侧面（复用你之前的绘制逻辑）
            cv2.drawContours(frame, [imgpts[:4]], -1, color, -1) 
            #绘制立方体底面
            # imgpts[:4] : 前4个点（底面4个点）
            # -1 : 填充整个区域
            #颜色：之前定义的颜色

            for j in range(4):
                cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255,255,255), 2)
                #绘制4条垂直线
                #连接底面和顶面对应点
                #颜色：白色，线宽2像素

                cv2.drawContours(frame, [np.array([imgpts[j], imgpts[(j+1)%4], imgpts[(j+1)%4+4], imgpts[j+4]])], -1, color, 1)
                #绘制4个侧面
                #每个侧面是一个四边形
                #只绘制边框，不填充
                #线宽1像素

    # 显示 FPS 和 结果
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #在图像左上角显示帧率
    #格式：保留1位小数
    #字体：HERSHEY_SIMPLEX
    #大小：0.7倍
    #颜色：绿色
    #线宽：2像素

    cv2.imshow('Multi-Marker AR', frame)
    #显示图像窗口
    #窗口标题：'Multi-Marker AR'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #检查是否按下了'q'键
    # waitKey(1) : 等待1毫秒
    #如果按'q'就退出循环

cap.release()
cv2.destroyAllWindows()
#释放摄像头资源
#关闭所有OpenCV窗口