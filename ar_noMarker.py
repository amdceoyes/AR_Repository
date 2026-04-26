import cv2
import numpy as np

class IntegratedMarkerlessAR:
    """
    集成无标记增强现实系统
    这个版本解决了之前代码中的"方块不方"问题，使用了更准确的3D姿态估计方法
    
    核心改进：
    1. 使用PnP（透视n点）算法计算真实的3D姿态
    2. 模拟相机内参，实现正确的透视投影
    3. 增加了多种防护逻辑，防止程序崩溃
    4. 使用BFMatcher的交叉验证提高匹配质量
    """
    
    def __init__(self):
        # 1. 初始化特征检测器：使用ORB算法
        # ORB是免费的快速特征检测算法，适合实时应用
        # nfeatures=1000: 最多检测1000个特征点
        # 减少特征点数量可以提高处理速度，但可能降低匹配精度
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # 2. 初始化特征匹配器：使用暴力匹配器(BFMatcher)
        # cv2.NORM_HAMMING: ORB是二进制描述子，使用汉明距离
        # crossCheck=True: 启用交叉验证，确保匹配的一致性
        # 交叉验证：匹配A→B和B→A，只有双向匹配的点才认为是有效匹配
        # 这能显著提高匹配质量，减少错误匹配
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 参考平面的特征点和描述子
        self.ref_kp = None  # 参考平面的关键点
        self.ref_des = None  # 参考平面的描述子
        
        # 目标平面的尺寸（宽度, 高度），单位是像素
        self.target_size = None
        
        # 相机内参矩阵（3×3）
        # 包含焦距(fx, fy)和主点(cx, cy)
        # 这是实现正确3D投影的关键
        self.K = None

    def set_target(self, frame, roi):
        """
        设置追踪目标（定义世界坐标系的原点）
        这是系统的初始化步骤，用户选择一个平面作为参考
        
        参数:
        frame: 当前摄像头帧
        roi: 感兴趣区域，格式为(x, y, width, height)
        返回: 无
        """
        # 提取ROI的坐标和尺寸
        x, y, w, h = roi
        
        # 防护：确保选择的区域足够大
        if w < 20 or h < 20:
            print("错误：选择的区域太小")
            return
        
        # 从原始帧中提取目标区域
        target_img = frame[y:y+h, x:x+w]
        
        # 保存目标平面的尺寸
        self.target_size = (w, h)
        
        # 在目标平面上提取ORB特征
        # 这些特征将用于后续的匹配和追踪
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(target_img, None)
        
        # 模拟相机内参：这是解决"方块不方"问题的核心
        # 在实际应用中，应该通过相机标定得到准确的内参
        # 这里我们使用经验值：
        # 假设焦距f = 图像宽度（近似）
        # 主点(cx, cy) = 图像中心
        f = frame.shape[1]  # 使用图像宽度作为焦距
        cx, cy = frame.shape[1] / 2, frame.shape[0] / 2  # 图像中心点
        
        # 构建内参矩阵
        # [[fx, 0, cx],
        #  [0, fy, cy],
        #  [0, 0,  1]]
        self.K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print("✅ 目标已锁定，3D坐标系已建立")
        print(f"  目标尺寸: {w}×{h} 像素")
        print(f"  特征点数量: {len(self.ref_kp)}")
        print(f"  相机内参: f={f:.0f}, cx={cx:.0f}, cy={cy:.0f}")

    def process(self, frame):
        """
        处理单帧图像，进行特征匹配和3D姿态估计
        
        参数:
        frame: 输入图像帧
        返回: 处理后的图像帧
        """
        # --- 防护逻辑 1：还没选目标时不运行 ---
        # 如果还没有设置参考平面，显示提示信息
        if self.ref_des is None:
            cv2.putText(frame, "Press 's' to Select Surface", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        # 提取当前帧的特征点和描述子
        kp, des = self.orb.detectAndCompute(frame, None)
        
        # --- 防护逻辑 2：当前画面没特征点时直接跳过，不退出 ---
        if des is None:
            return frame

        # 使用暴力匹配器进行特征匹配
        # 匹配参考平面和当前帧的特征点
        matches = self.bf.match(self.ref_des, des)
        
        # 对匹配结果按距离排序，取前50个最佳匹配
        # 距离越小，匹配质量越高
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        # --- 防护逻辑 3：匹配点太少时不计算，防止强制计算导致的扭曲和闪退 ---
        if len(matches) > 15:
            # 获取目标平面的宽度和高度
            w, h = self.target_size
            
            # 准备PnP算法需要的坐标
            # 3D参考点：平面的四个角（在目标平面的局部坐标系中）
            # 格式：[[x, y, z], ...]
            # 假设目标平面在z=0平面上
            obj_pts = np.float32([
                [0, 0, 0],    # 左上角
                [w, 0, 0],    # 右上角
                [w, h, 0],    # 右下角
                [0, h, 0]     # 左下角
            ])
            
            # 2D对应点：当前画面中匹配到的坐标
            # 从参考平面提取源点
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches])
            # 从当前帧提取目标点
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

            # 4. 利用单应性矩阵和RANSAC剔除错误点
            # 单应性矩阵H：描述两个平面之间的投影变换
            # RANSAC：随机采样一致性算法，用于排除异常点
            # 5.0：RANSAC重投影误差阈值，单位是像素
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # --- 防护逻辑 4：只有找到有效的平面变换才继续 ---
            if H is not None and H.shape == (3, 3):
                try:
                    # 将平面四个角投射到当前帧
                    # 使用单应性矩阵H将参考平面的角点映射到当前帧
                    rect_pts = cv2.perspectiveTransform(obj_pts[:, :2].reshape(-1, 1, 2), H)
                    
                    # 5. 【解决方块不方】使用solvePnP计算真实的相机位姿
                    # solvePnP：从3D-2D点对应关系求解相机姿态
                    # 输入：3D物体点，对应的2D图像点，相机内参，畸变系数
                    # 输出：旋转向量(rvec)，平移向量(tvec)
                    # 这些向量描述了相机相对于目标平面的位置和方向
                    _, rvec, tvec = cv2.solvePnP(
                        obj_pts,      # 3D物体点
                        rect_pts,     # 对应的2D图像点
                        self.K,       # 相机内参
                        np.zeros((4, 1))  # 畸变系数（假设没有畸变）
                    )

                    # 6. 定义真正的3D立方体顶点
                    # 在目标平面上方定义一个立方体
                    size = w // 2  # 立方体高度设为平面宽度的一半
                    cube_3d = np.float32([
                        # 底面4个点（z=0）
                        [0, 0, 0],
                        [w, 0, 0],
                        [w, h, 0],
                        [0, h, 0],
                        # 顶面4个点（z=-size）
                        [0, 0, -size],
                        [w, 0, -size],
                        [w, h, -size],
                        [0, h, -size]
                    ])
                    
                    # 7. 【关键函数】利用相机模型投影3D点到2D屏幕
                    # projectPoints：将3D点投影到2D图像平面
                    # 使用从solvePnP得到的相机姿态
                    # 这个函数能正确处理透视投影，使立方体看起来更真实
                    img_pts, _ = cv2.projectPoints(
                        cube_3d,     # 3D立方体顶点
                        rvec,        # 旋转向量
                        tvec,        # 平移向量
                        self.K,      # 相机内参
                        np.zeros((4, 1))  # 畸变系数
                    )
                    img_pts = np.int32(img_pts).reshape(-1, 2)

                    # 8. 绘制方块
                    # 绘制底面（绿色）
                    cv2.polylines(frame, [img_pts[:4]], True, (0, 255, 0), 2)
                    
                    # 绘制顶面（红色）
                    cv2.polylines(frame, [img_pts[4:]], True, (0, 0, 255), 2)
                    
                    # 绘制4条垂直边（青色）
                    for i in range(4):
                        cv2.line(frame, 
                                tuple(img_pts[i]),      # 底面顶点
                                tuple(img_pts[i+4]),    # 对应顶面顶点
                                (255, 255, 0),  # 青色
                                2)
                    
                    # 显示跟踪状态
                    cv2.putText(frame, "TRACKING", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                except Exception as e:
                    # 如果计算过程中出现错误，静默处理
                    # 这样可以防止程序崩溃
                    pass
        else:
            # 匹配点太少，显示跟踪丢失
            cv2.putText(frame, "LOST TARGET", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
        return frame

# --- 稳定的启动逻辑 ---
if __name__ == "__main__":
    """
    主程序入口
    提供稳定的交互界面
    """
    
    # 打开摄像头
    # 参数0表示默认摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        exit(1)
    
    # 创建AR系统实例
    ar = IntegratedMarkerlessAR()
    
    # 窗口名称
    win_name = "Integrated AR"
    
    # 显示使用说明
    print("=== 集成无标记AR系统 ===")
    print("按 's' 键：选择要跟踪的平面")
    print("按 'q' 键：退出程序")
    print("选择平面时，用鼠标拖动选择矩形区域")
    print("=================================")
    
    # 主循环
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取视频帧")
            break
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按's'键：开始选择目标平面
        if key == ord('s'):
            # 弹出选择框，让用户选择ROI
            # cv2.selectROI参数说明：
            # - win_name: 窗口名称
            # - frame: 要选择的图像
            # - False: 是否显示十字线
            # 返回值: (x, y, width, height) 或 (0,0,0,0)（如果取消）
            roi = cv2.selectROI(win_name, frame, False)
            
            # 如果选择了有效的ROI（宽度和高度大于0）
            if roi[2] > 0 and roi[3] > 0:
                ar.set_target(frame, roi)
        
        # 按'q'键：退出程序
        elif key == ord('q'):
            break
        
        # 处理当前帧
        # 使用try-except包装，确保即使处理出错程序也不会崩溃
        try:
            display = ar.process(frame)
            cv2.imshow(win_name, display)
        except Exception as e:
            # 如果处理出错，只显示原始帧
            cv2.imshow(win_name, frame)
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

"""
1. 解决“方块畸变/不方”问题 (3D Perspective Distortion)
现象：在初始版本中，虚拟立方体在相机倾斜时会变成扭曲的平行四边形或不规则多边形，无法维持正方体形状。
原因：
维度缺失：初始代码使用了 单应性矩阵 (Homography) 进行 2D 投影。这种方法本质上是将参考图“贴”在平面上，它假设物体没有深度（Z轴）。
透视忽视：单应性变换不考虑相机的焦距和光学中心。当视角倾斜时，它只进行线性拉伸，而不会根据物理透视规律进行收缩。
解决方案：
引入 PnP 算法 (Perspective-n-Point)：通过 cv2.solvePnP 计算相机在 3D 空间中相对于目标的真实 旋转向量 (rvec) 和 平移向量 (tvec)。
姿态解算：配合模拟的相机内参矩阵 (Camera Matrix)，使用 cv2.projectPoints 将 3D 立方体顶点投影到 2D 屏幕。这确保了方块遵循“近大远小”和“透视收缩”的物理规律。
2. 解决“失去平面自动退出”问题 (Program Robustness)
现象：当摄像头被遮挡、移开或特征点匹配失败时，程序会直接报错并强制关闭窗口。
原因：
空值操作 (NoneType Error)：当算法找不到足够的匹配点（matches < 4）时，计算单应性矩阵或位姿的函数会返回 None。如果代码直接对 None 进行矩阵运算，Python 会抛出异常。
逻辑中断：缺乏防御性编程，将“追踪失败”这一常态现象误认为是程序致命错误，导致主循环 while 被 break 或因未捕获的异常而中止。
解决方案：
防御性条件判断：在计算前增加阈值检查（例如 if len(matches) > 15），只有在数据充足时才执行数学运算。
异常捕获 (Try-Except)：使用 try...except 块包裹核心视觉处理逻辑，确保即使在数学计算出现奇异矩阵或溢出时，程序也只是跳过当前帧的渲染，保持 while 循环和摄像头预览的持续运行。
状态反馈：将“退出程序”改为“状态提示”，在界面上实时显示 LOST TARGET，提升了交互体验。
"""
"""
系统架构与技术细节详解
1. 核心改进：解决"方块不方"问题
问题分析：之前的代码使用单应性矩阵直接绘制3D物体，导致：
立方体变形，不符合透视原理
侧面不垂直，看起来像平行四边形
快速移动时抖动严重
解决方案：使用真正的3D姿态估计
# 关键步骤： 1. 使用solvePnP计算相机姿态 (rvec, tvec) 2. 使用projectPoints进行正确的3D到2D投影 3. 使用相机内参矩阵实现透视投影 
2. 相机模型与投影原理
相机内参矩阵 K：
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] 
fx, fy：焦距（像素单位），控制视野大小
cx, cy：主点（图像中心），通常是图像宽度/2和高度/2
3D到2D投影公式：
s * [u, v, 1]^T = K * [R|t] * [X, Y, Z, 1]^T 
u, v：图像像素坐标
X, Y, Z：3D世界坐标
R, t：相机旋转和平移（外参）
s：尺度因子
3. 特征匹配优化
交叉验证（crossCheck）：
# 传统匹配：A中的点找B中的最近邻 # 交叉验证：A→B和B→A双向验证 # 只有双向匹配的点才认为是有效匹配 # 这能显著减少错误匹配 
距离排序与筛选：
# 按匹配距离排序，取前N个最佳匹配 matches = sorted(matches, key=lambda x: x.distance)[:50] # 为什么是50？ # 太多匹配点：计算量大，可能包含错误匹配 # 太少匹配点：姿态估计不稳定 # 50是一个经验值，平衡了精度和速度 
4. 鲁棒性设计：多层防护逻辑
防护逻辑1：检查是否已设置参考平面
if self.ref_des is None: return frame # 直接返回，不处理 
防护逻辑2：检查当前帧是否有特征点
if des is None: return frame # 跳过处理 
防护逻辑3：检查匹配点数量
if len(matches) > 15: # 至少需要15个匹配点 # 进行计算 else: # 显示"跟踪丢失" 
防护逻辑4：检查单应性矩阵有效性
if H is not None and H.shape == (3, 3): # 继续计算 
防护逻辑5：异常处理
try: # 可能出错的计算 _, rvec, tvec = cv2.solvePnP(...) except Exception: pass # 静默处理，不崩溃 
5. 实际应用建议
5.1 提高跟踪稳定性
# 添加历史帧平滑 def smooth_pose(self, new_rvec, new_tvec): ""平滑相机姿态，减少抖动"" if self.last_rvec is None: self.last_rvec = new_rvec self.last_tvec = new_tvec else: # 使用指数加权移动平均 alpha = 0.3 # 平滑系数 self.last_rvec = (1 - alpha) * self.last_rvec + alpha * new_rvec self.last_tvec = (1 - alpha) * self.last_tvec + alpha * new_tvec return self.last_rvec, self.last_tvec 
5.2 添加交互控制
# 在process方法中添加键盘控制 def process(self, frame): # ... 之前的代码 ... # 添加键盘控制 key = cv2.waitKey(1) & 0xFF if key == ord('w'): # 增大立方体 self.cube_scale *= 1.1 elif key == ord('s'): # 减小立方体 self.cube_scale *= 0.9 elif key == ord('c'): # 切换颜色 self.cube_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) return frame 
5.3 性能优化
# 调整ORB参数 self.orb = cv2.ORB_create( nfeatures=800, # 减少特征点数量 scaleFactor=1.2, # 金字塔尺度因子 nlevels=8, # 金字塔层数 edgeThreshold=31, # 边缘阈值 firstLevel=0, # 第一层 WTA_K=2, # 产生描述子的点数 scoreType=cv2.ORB_HARRIS_SCORE, # 评分类型 patchSize=31, # 描述子块大小 fastThreshold=20 # FAST阈值 ) 
6. 故障排除
问题1：立方体位置不正确
可能原因： 1. 相机内参不准确 2. 目标平面尺寸测量错误 3. 特征匹配质量差 解决方案： 1. 使用相机标定获取准确内参 2. 准确测量目标平面尺寸 3. 增加特征点数量，改善光照条件 
问题2：跟踪抖动严重
可能原因： 1. 特征点数量不足 2. 平面纹理单一 3. 光照变化大 解决方案： 1. 选择纹理丰富的平面 2. 增加特征点数量 3. 添加姿态平滑 
问题3：快速移动时跟踪丢失
可能原因： 1. 处理速度跟不上 2. 特征变化太快 解决方案： 1. 降低图像分辨率 2. 减少特征点数量 3. 使用光流辅助跟踪 
7. 扩展功能
7.1 添加增强现实内容
def draw_ar_content(self, frame, rvec, tvec): ""绘制多种AR内容"" # 绘制坐标轴 cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1) # 绘制文字 img_pts, _ = cv2.projectPoints( np.float32([[w/2, h/2, -50]]), # 文字位置 rvec, tvec, self.K, self.D ) cv2.putText(frame, "AR Text", (int(img_pts[0][0][0]), int(img_pts[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # 绘制3D模型 # 可以加载.obj等3D模型文件 
7.2 多平面跟踪
class MultiPlaneAR: def __init__(self): self.planes = {} # 存储多个平面 def add_plane(self, name, frame, roi): ""添加新的跟踪平面"" plane = IntegratedMarkerlessAR() plane.set_target(frame, roi) self.planes[name] = plane def track_all(self, frame): ""跟踪所有平面"" for name, plane in self.planes.items(): frame = plane.process(frame) return frame 
这个集成的无标记AR系统提供了一个稳定、准确的AR体验，通过正确的3D姿态估计和多重防护机制，能够处理各种实际场景。
"""
"""
## 1. 为什么 ORB 会失败？（特征提取的局限）
ORB 本质上是在找**“像素亮度的剧烈变化”**。它失败通常有三个原因：
 * **环境信息熵不足（纹理缺失）**：如果你对着一面白墙、一张纯色桌面或者磨砂金属表面，ORB 找不到足够的“角点”。没有点，就没有指纹。
 * **运动模糊（Motion Blur）**：当你快速晃动摄像头时，画面会变糊。像素点散开了，ORB 无法在糊成一片的图像中提取出稳定的特征描述子。
 * **光照极端变化**：ORB 对亮度虽然有一定鲁棒性，但在极暗或过曝（反光）的情况下，像素对比度消失，特征点会大量丢失。
## 2. 为什么 PnP 会漂？（数学解算的误差）
你看到的“方块在抖”或者“位置对不准”就是**漂移（Drift）**。
 * **噪声传递**：ORB 提取的点位置不是 100% 准确的，可能由于像素重采样有 1-2 像素的偏差。PnP 是一个数学方程，**输入的 2D 坐标有微小抖动，解出来的 3D 旋转和位移就会被放大成剧烈的跳动。**
 * **相机内参不准**：我们在代码里模拟了相机内参 K。如果这个矩阵（焦距、中心点）和你的真实摄像头硬件不匹配，PnP 算出来的深度就会错位，导致方块看起来“浮”在平面上或者比例不对。
 * **退化构型**：如果你选取的特征点几乎挤在一起，或者都在一条直线上，PnP 就会解出多个可能的答案（多解问题），导致方块在几个位置之间反复横跳。
## 3. 为什么 RANSAC 能救系统？（概率论的胜利）
RANSAC（随机采样一致性）是视觉算法里的**“防抖过滤器”**。
 * **剔除离群点（Outliers）**：在匹配过程中，总会有“猪队友”。比如书面上的一颗灰尘被识别成了特征点，但下一秒它动了，或者它被误认为是背景里的点。如果直接把这些错点带入 PnP 计算，方块会瞬间飞出屏幕。
 * **寻找“共识集”**：RANSAC 的逻辑非常流氓但也非常有效：
   1. 它随机拔取一小部分点（比如 4 个点）算出位姿。
   2. 看看剩下的点里，有多少点符合这个位姿。
   3. 重复成百上千次，**找出支持人数最多的那个答案**。
 * **救命效果**：它把那些乱跳的错误匹配点（离群点）直接视为无效，只用最靠谱的“核心群众”来算位置。所以，即便有 30% 的点匹配错了，只要剩下的点是准的，系统依然能稳住。
"""