import cv2
import numpy as np
import time

class ARSystemManager:
    def __init__(self):
        """
        初始化增强现实(AR)系统管理器
        
        这个类管理整个AR系统的状态，包含以下核心功能：
        1. 视觉特征提取与匹配
        2. 系统状态机管理
        3. 视觉惯性融合(VIO)位姿估计
        4. 关键帧管理与重定位
        """
        
        # 1. 初始化视觉处理模块
        # ORB特征提取器：提取图像中的关键点和描述子
        # nfeatures=1000: 每帧最多提取1000个特征点
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # 暴力匹配器：用于匹配当前帧与关键帧的特征点
        # crossCheck=True: 双向匹配，确保匹配的鲁棒性
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 关键帧列表：系统的"空间记忆"
        # 每个关键帧包含：特征描述子、相机位姿、时间戳
        self.keyframes = []  
        
        # 2. 系统状态机 - 定义系统可能的状态
        # INIT:     系统初始化，寻找第一个可追踪目标
        # TRACKING: 正常追踪状态，视觉定位准确
        # LOST:     视觉丢失状态，使用IMU预测
        # RECOVERING: 重定位状态，尝试找回位置
        self.state = "INIT"  
        
        # 3. 位姿信息存储
        # rvec: 旋转向量(3x1)，表示相机朝向
        # tvec: 平移向量(3x1)，表示相机位置
        self.rvec = np.zeros((3, 1))
        self.tvec = np.zeros((3, 1))
        
        # 4. 时间管理
        # 用于计算帧间时间差，IMU积分需要精确的时间间隔
        self.last_time = time.time()

    def run_step(self, frame, imu_gyro):
        """
        主循环函数：处理每一帧图像和IMU数据
        
        参数:
            frame: 当前视频帧 (numpy数组, BGR格式)
            imu_gyro: IMU陀螺仪数据，包含角速度信息(3x1向量)
        
        处理流程:
            1. 提取当前帧特征
            2. 根据当前状态执行对应逻辑
            3. 更新系统状态
            4. 渲染AR效果
        """
        # 1. 特征提取：检测关键点并计算描述子
        # kps: 关键点位置列表
        # des: 特征描述子矩阵
        kps, des = self.orb.detectAndCompute(frame, None)
        
        # ===== 状态机核心逻辑 =====
        # 根据当前状态执行不同的处理流程
        
        if self.state == "INIT":
            """
            初始化状态：
            目标：找到第一个可追踪的标记或特征点
            条件：需要足够多的匹配特征点才能初始化成功
            """
            success, r, t = self.try_pnp_tracking(kps, des)
            if success:
                # 初始化成功，记录初始位姿
                self.rvec, self.tvec = r, t
                # 切换到追踪状态
                self.state = "TRACKING"
                print(">>> [INIT -> TRACKING] 系统启动成功")
                print(f"   初始位姿: 旋转={r.flatten()}, 平移={t.flatten()}")

        elif self.state == "TRACKING":
            """
            正常追踪状态：
            目标：使用视觉方法持续追踪相机位姿
            失败处理：视觉丢失时切换到IMU预测模式
            """
            success, r, t = self.try_pnp_tracking(kps, des)
            if success:
                # 视觉追踪成功，更新位姿
                self.rvec, self.tvec = r, t
                
                # 管理关键帧：如果移动足够远，保存当前帧为关键帧
                self.manage_keyframes(des, r, t)
                
                # 更新时间戳，用于后续IMU积分
                self.last_time = time.time()
                
                # 可选：输出调试信息
                # print(f"追踪成功 - 位姿更新")
            else:
                # 视觉追踪失败，切换到丢失状态
                self.state = "LOST"
                print(">>> [TRACKING -> LOST] 视觉丢失，切入IMU预测模式")
                print("    原因: 特征点匹配不足或PnP求解失败")

        elif self.state == "LOST":
            """
            视觉丢失状态：
            目标：使用IMU数据预测相机运动
            同时尝试通过视觉重定位找回位置
            """
            # 使用IMU陀螺仪数据进行运动预测
            # 公式: 当前角度 = 上一帧角度 + 角速度 × 时间差
            self.rvec = self.predict_with_imu(self.rvec, imu_gyro)
            
            # 尝试通过视觉重定位
            success, r, t = self.try_pnp_tracking(kps, des)
            if success:
                # 重定位成功，恢复视觉追踪
                # 注意：这里可以进行位姿融合，使过渡更平滑
                self.rvec, self.tvec = r, t
                self.state = "TRACKING"
                print(">>> [LOST -> TRACKING] 视觉重定位成功，恢复正常追踪")
                
                # 可选：卡尔曼滤波融合IMU预测和视觉测量
                # self.fuse_pose_estimates(imu_predicted, visual_measured)

        # 4. 渲染AR效果
        # 在图像上绘制虚拟物体，并显示当前系统状态
        self.render_ar(frame, self.rvec, self.tvec)

    def try_pnp_tracking(self, kps, des):
        """
        尝试使用PnP(透视n点)算法进行视觉位姿估计
        
        参数:
            kps: 当前帧的关键点
            des: 当前帧的特征描述子
            
        返回:
            success: 是否成功估计位姿 (bool)
            r: 旋转向量 (3x1 numpy数组，成功时返回)
            t: 平移向量 (3x1 numpy数组，成功时返回)
        
        PnP算法原理:
            通过2D-3D点对应关系，求解相机位姿
            需要至少4个匹配点对
        """
        # 检查是否有足够的特征点
        if des is not None and len(des) > 20:
            # 实际实现步骤：
            # 1. 与关键帧特征匹配
            # 2. 筛选好的匹配对 (使用比率测试或距离阈值)
            # 3. 获取对应的3D点 (从关键帧或已知地图)
            # 4. 调用cv2.solvePnP()求解位姿
            # 5. 使用RANSAC剔除异常值
            
            # 示例：这里简化返回成功和零位姿
            # 实际应该返回真实的PnP计算结果
            return True, np.random.randn(3,1)*0.1, np.random.randn(3,1)*0.1
        return False, None, None

    def predict_with_imu(self, last_r, gyro):
        """
        使用IMU陀螺仪数据预测相机旋转
        
        参数:
            last_r: 上一帧的旋转向量
            gyro: 当前陀螺仪测量值 (角速度，弧度/秒)
            
        返回:
            predicted_r: 预测的旋转向量
            
        物理公式:
            θ_current = θ_previous + ω * Δt
            其中: θ是角度，ω是角速度，Δt是时间间隔
        """
        # 计算时间差
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # 物理积分：角度 = 上一帧角度 + 角速度 × 时间
        # 注意：这是简化模型，实际中需要考虑坐标系变换和噪声
        predicted_r = last_r + gyro * dt
        
        # 实际系统中可能需要：
        # 1. 陀螺仪零偏校正
        # 2. 四元数或旋转矩阵表示
        # 3. 数值积分方法(如龙格-库塔)
        
        return predicted_r

    def manage_keyframes(self, des, r, t):
        """
        关键帧管理：决定何时保存新的关键帧
        
        参数:
            des: 当前帧的特征描述子
            r: 当前帧的旋转向量
            t: 当前帧的平移向量
            
        关键帧选择策略:
            1. 距离上一个关键帧足够远(平移或旋转)
            2. 当前帧有足够的特征点
            3. 特征点分布均匀
        """
        if len(self.keyframes) == 0:
            # 第一个关键帧
            keyframe = {
                'descriptors': des,
                'rvec': r.copy(),
                'tvec': t.copy(),
                'time': time.time()
            }
            self.keyframes.append(keyframe)
            print("    保存第一个关键帧")
        else:
            # 计算与最后一个关键帧的距离
            last_kf = self.keyframes[-1]
            trans_distance = np.linalg.norm(t - last_kf['tvec'])
            
            # 如果移动超过阈值，保存为新的关键帧
            if trans_distance > 0.1:  # 10厘米阈值
                keyframe = {
                    'descriptors': des,
                    'rvec': r.copy(),
                    'tvec': t.copy(),
                    'time': time.time()
                }
                self.keyframes.append(keyframe)
                print(f"    保存关键帧 #{len(self.keyframes)}，移动距离: {trans_distance:.3f}m")

    def render_ar(self, frame, r, t):
        """
        渲染AR效果：在图像上绘制虚拟内容和状态信息
        
        参数:
            frame: 要绘制的图像帧
            r: 相机旋转向量
            t: 相机平移向量
        """
        # 1. 显示当前系统状态
        # 使用不同颜色表示不同状态
        color_map = {
            "INIT": (0, 255, 255),    # 黄色
            "TRACKING": (0, 255, 0),  # 绿色
            "LOST": (0, 0, 255),      # 红色
            "RECOVERING": (0, 165, 255)  # 橙色
        }
        
        color = color_map.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"STATE: {self.state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 2. 显示位姿信息
        # 格式化显示旋转和平移值
        r_str = f"R: [{r[0,0]:.2f}, {r[1,0]:.2f}, {r[2,0]:.2f}]"
        t_str = f"T: [{t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f}]"
        
        cv2.putText(frame, r_str, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, t_str, (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 3. 显示关键帧数量
        cv2.putText(frame, f"Keyframes: {len(self.keyframes)}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        # 4. 绘制虚拟物体（示例：坐标系或立方体）
        # 这里可以调用你的3D模型渲染代码
        # 例如：draw_coordinate_axes(frame, r, t, camera_matrix, dist_coeffs)
        
        # 5. 显示图像
        cv2.imshow("AR System Manager", frame)
        
        # 6. 检查退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("系统正常退出")
            cv2.destroyAllWindows()
            exit(0)

# ===== 使用示例 =====
if __name__ == "__main__":
    """
    使用这个AR系统的示例代码
    """
    # 创建AR系统管理器
    ar_system = ARSystemManager()
    
    # 模拟主循环
    # 实际使用时，这里应该从摄像头和IMU读取真实数据
    print("AR系统初始化完成")
    print("等待摄像头输入...")
    print("按'q'键退出程序")
    
    # 示例：模拟10帧数据
    for i in range(10):
        # 模拟一帧图像（黑色背景）
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 模拟IMU陀螺仪数据（轻微旋转）
        imu_gyro = np.array([[0.01], [0.02], [0.01]])  # 弧度/秒
        
        # 处理当前帧
        ar_system.run_step(frame, imu_gyro)
        
        # 模拟帧率
        time.sleep(0.1)




"""
最底层（def们）：是你的工具箱。它们只负责干脏活累活（算数学、找特征、滤噪声）。它们不关心现在是哪一帧，只关心输入和输出。
中间层（状态机 elif）：是你的指挥官。它决定此时此刻该派哪个“工具”上场。
最顶层（run_step 循环）：是你的生命线。它保证系统不停地跳动，处理每一帧进来的图像。

1. 关于“初始化（第一帧）”
你提到的“确认第一帧”非常关键。在 SLAM 里这叫 Initialization（初始化）。
为什么重要：没有第一帧，你就没有坐标原点。这就像你出生在世界上，得先睁开眼看一眼，才知道哪是北。
你的任务：确保第一帧的特征点足够多且稳，否则整个坐标系后面全是歪的。
2. 关于“循环外的 def”
你发现它们在外面，这在软件工程里叫 解耦（Decoupling）。
好处：如果以后你想把“陀螺仪纠错”的算法从简单的相加改成复杂的卡尔曼滤波，你只需要改那个 def 里的逻辑，而不需要动状态机的 elif。这就是为什么大项目不容易崩的原因。
3. 关于“打印与反馈”
你提到的“画画并投影”，在专业领域叫 Visualization（可视化）。
深度理解：它不仅仅是给用户看的，更是给你（开发者）调试用的。通过在屏幕上画出特征点和当前的 State，你能一眼看出是哪个 def 出了 Bug。
"""
"""
第一帧用于初始化参考状态（initial pose / initial map anchor）

状态机 = 系统控制逻辑
它决定：
用视觉还是IMU
是 tracking 还是 lost
是否更新 keyframe
是否执行回环
更本质一句话：
状态机不是代码结构，是系统大脑

函数不是“附属品”，而是：
不同子系统的实现单元
比如：
ORB → 感知系统
PnP → 几何系统
IMU融合 → 状态估计系统
render → 表达系统
不是：
状态机调用函数
而是：
状态机在协调多个“独立子系统”

输入层（Camera / IMU）
    ↓
感知层（ORB）
    ↓
几何层（PnP）
    ↓
状态层（State Machine）
    ↓
融合层（IMU / Filter）
    ↓
输出层（Render）
"""
"""
1. 状态：VISION_STABLE (视觉稳定)
触发条件：ORB 特征匹配点 > 阈值，PnP 解算误差在正常范围内。
执行动作：
主导位姿：完全信任视觉位姿（PnP）。
校准 IMU：利用当前的视觉位姿来反向修正 IMU 的零偏（Bias）。
地图维护：根据位移决定是否保存 Keyframe。
回环检测：在后台异步进行 Loop Detection。
2. 状态：LOST_PREDICTION (追踪丢失/惯性预测)
触发条件：视觉特征点暴跌，或 PnP 无法解算出有效解。
执行动作：
主导位姿：立即切换到 IMU 积分预测模式，利用 \theta_{new} = \theta_{old} + \omega \cdot \Delta t 维持方块位置。
功能降级：暂时关闭回环检测和关键帧保存，全力保证渲染帧率。
超时处理：如果 2 秒后视觉仍未找回，进入重定位模式。
3. 状态：RELOCALIZING (重定位/恢复模式)
触发条件：视觉丢失后重新看到特征。
执行动作：
暴力匹配：拿当前帧去检索历史 Keyframe。
平滑修正：一旦找回位置，不要让方块“瞬移”，而是通过低通滤波平滑地从 IMU 预测位姿拉回到视觉位姿。
重置状态：成功后跳回 VISION_STABLE。
"""
"""
为了完成整合，代码需要进行以下结构性调整：
统一接口：创建一个 AR_Manager 类，所有的零件（ORB, PnP, IMU, Keyframe）都作为它的私有变量。
主循环重写：update() 函数中不再直接写算法逻辑，而是写 if self.state == ... 的分支判断。
引入时间戳同步：所有输入数据（图像和陀螺仪）必须带有 timestamp，这是计算 \Delta t 和多端同步的基础。
分层错误处理：为每个模块设置 Try-Except 或错误码返回，任何模块的 Bug 都会触发状态机向“降级模式”切换，而不是程序崩溃。
"""



"""
AR系管理器代码详细解析
第一部分：类定义与初始化
class ARSystemManager: def __init__(self): "" 初始化增强现实(AR)系统管理器 这个类管理整个AR系统的状态，包含以下核心功能： 1. 视觉特征提取与匹配 2. 系统状态机管理 3. 视觉惯性融合(VIO)位姿估计 4. 关键帧管理与重定位 ""
区块1：视觉特征处理模块
# 1. 初始化视觉处理模块 # ORB特征提取器：提取图像中的关键点和描述子 # nfeatures=1000: 每帧最多提取1000个特征点 self.orb = cv2.ORB_create(nfeatures=1000) # 暴力匹配器：用于匹配当前帧与关键帧的特征点 # crossCheck=True: 双向匹配，确保匹配的鲁棒性 self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
作用：
建立系统的"眼睛" - 让计算机能够"看到"和理解图像内容
ORB用于从图像中提取独特的特征点（类似人眼识别的关键点）
BFMatcher用于比较不同图像间的特征点是否匹配
实现原理：
ORB特征提取：通过FAST角点检测+BRIEF描述子算法
检测图像中的角点、边缘等显著位置
为每个特征点生成256位的二进制描述符
二进制描述符的优势：计算快、匹配效率高
暴力匹配器：
比较所有特征点对的汉明距离
crossCheck确保A匹配B的同时B也匹配A，避免误匹配
为什么是这个格式：
使用类成员变量（self.orb, self.bf）保存特征提取器
一次初始化，多次使用，避免重复创建开销
nfeatures=1000平衡了精度和计算效率
区块2：空间记忆存储
# 关键帧列表：系统的"空间记忆" # 每个关键帧包含：特征描述子、相机位姿、时间戳 self.keyframes = [] 
作用：
建立系统的"记忆" - 记住曾经"看过"的场景
为后续的跟踪和重定位提供参考
实现原理：
列表结构简单灵活，支持动态添加
每个关键帧存储：
descriptors：特征描述子（场景的"指纹"）
rvec/tvec：相机位置和朝向（空间定位）
time：时间戳（用于时序分析）
为什么是这个格式：
列表适用于顺序存储和索引访问
字典结构方便存储多种类型数据
.copy()确保数据独立性，避免引用问题
区块3：状态机与位姿管理
# 2. 系统状态机 - 定义系统可能的状态 # INIT: 系统初始化，寻找第一个可追踪目标 # TRACKING: 正常追踪状态，视觉定位准确 # LOST: 视觉丢失状态，使用IMU预测 # RECOVERING: 重定位状态，尝试找回位置 self.state = "INIT" # 3. 位姿信息存储 # rvec: 旋转向量(3x1)，表示相机朝向 # tvec: 平移向量(3x1)，表示相机位置 self.rvec = np.zeros((3, 1)) self.tvec = np.zeros((3, 1)) 
作用：
管理系统的"大脑状态" - 知道当前处于什么模式
记录相机的"位置和朝向" - 空间定位核心
实现原理：
状态机：用字符串表示状态
INIT：系统启动，寻找初始位置
TRACKING：正常工作，视觉定位稳定
LOST：视觉丢失，IMU临时接管
状态转移由条件触发（如视觉成功/失败）
位姿表示：
rvec：旋转向量(3x1)，Rodrigues旋转表示
tvec：平移向量(3x1)，世界坐标系下的位置
使用numpy数组便于数学运算
为什么是这个格式：
字符串状态直观易懂
3x1向量符合OpenCV的PnP函数要求
零初始化确保系统有确定起始值
区块4：时间管理系统
# 4. 时间管理 # 用于计算帧间时间差，IMU积分需要精确的时间间隔 self.last_time = time.time() 
作用：
提供时间基准，计算物理运动
控制IMU积分的准确性
实现原理：
保存上一帧处理完成的时间戳
当前时间减上一帧时间得到时间差Δt
IMU积分：角度变化 = 角速度 × Δt
为什么是这个格式：
time.time()返回秒为单位的浮点数
高精度时间戳支持精确的物理模拟
简单的时间差计算，实时性好
第二部分：主处理循环
区块5：特征提取预处理
def run_step(self, frame, imu_gyro): "" 主循环函数：处理每一帧图像和IMU数据 "" # 1. 特征提取：检测关键点并计算描述子 # kps: 关键点位置列表 # des: 特征描述子矩阵 kps, des = self.orb.detectAndCompute(frame, None) 
作用：
从原始图像中提取可用于匹配的信息
将图像转换为计算机可处理的数值特征
实现原理：
detect：找出图像中的特征点
使用FAST算法检测角点
计算方向信息，使特征具有旋转不变性
compute：生成特征描述子
在特征点周围采样生成二进制模式
每个特征点对应一个256位的二进制串
工作流程：
输入图像 → FAST角点检测 → 计算方向 → BRIEF描述子 → 输出(kps, des) 
为什么是这个格式：
返回(kps, des)元组，结构清晰
kps包含位置、尺度、方向等属性
des是N×256的uint8数组，N为特征点数量
区块6：状态机核心逻辑
# ===== 状态机核心逻辑 ===== # 根据当前状态执行不同的处理流程 
子区块6.1：INIT状态
if self.state == "INIT": success, r, t = self.try_pnp_tracking(kps, des) if success: self.rvec, self.tvec = r, t self.state = "TRACKING" 
作用：
系统启动时的初始化阶段
寻找第一个可靠的定位点
逻辑流程：
INIT状态 → 尝试PnP跟踪 → 成功 → 记录位姿 → 切换到TRACKING → 失败 → 保持INIT状态 
为什么这样设计：
需要成功定位一次才能进入跟踪状态
避免在定位不准确时开始跟踪
提供系统启动的明确信号
子区块6.2：TRACKING状态
elif self.state == "TRACKING": success, r, t = self.try_pnp_tracking(kps, des) if success: self.rvec, self.tvec = r, t self.manage_keyframes(des, r, t) self.last_time = time.time() else: self.state = "LOST" 
作用：
正常的视觉跟踪工作状态
持续更新相机位姿
工作流程：
视觉跟踪成功 → 更新位姿 → 管理关键帧 → 更新时间戳 视觉跟踪失败 → 切换到LOST状态 
关键设计：
成功时：更新位姿并保存关键信息
失败时：立即降级到LOST状态，避免错误累积
关键帧管理：选择性保存重要帧，节省内存
子区块6.3：LOST状态
elif self.state == "LOST": self.rvec = self.predict_with_imu(self.rvec, imu_gyro) success, r, t = self.try_pnp_tracking(kps, des) if success: self.rvec, self.tvec = r, t self.state = "TRACKING" 
作用：
视觉丢失时的容错处理
使用IMU维持短期定位
容错机制：
视觉丢失 → IMU预测维持 → 尝试重定位 → 成功恢复TRACKING → 失败保持LOST 
设计优势：
IMU预测：提供短时运动估计
持续重定位：不断尝试找回视觉
平滑恢复：成功后可无缝切换回视觉跟踪
第三部分：核心算法模块
区块7：PnP位姿估计
def try_pnp_tracking(self, kps, des): "" 尝试使用PnP算法进行视觉位姿估计 "" if des is not None and len(des) > 20: return True, np.random.randn(3,1)*0.1, np.random.randn(3,1)*0.1 return False, None, None 
作用：
通过2D-3D点对应关系求解相机位姿
实现视觉定位的核心算法
PnP算法原理：
已知：3D空间点坐标 + 对应的2D图像点坐标 求解：相机在世界坐标系中的位置和朝向 数学：最小化重投影误差 
实际实现应包含：
特征匹配：当前帧与关键帧特征点匹配
3D点获取：从关键帧获取匹配点的3D坐标
solvePnP调用：OpenCV的PnP求解函数
RANSAC滤波：剔除异常匹配对
为什么用这个接口：
返回(success, rvec, tvec)三元组
success标志表示算法是否可靠
使用随机数模拟真实位姿变化
区块8：IMU运动预测
def predict_with_imu(self, last_r, gyro): "" 使用IMU陀螺仪数据预测相机旋转 "" now = time.time() dt = now - self.last_time self.last_time = now predicted_r = last_r + gyro * dt return predicted_r 
作用：
在视觉丢失时提供短时运动估计
基于物理定律的预测
物理原理：
角度积分公式：θ_t = θ_{t-1} + ω × Δt 其中：θ为角度，ω为角速度，Δt为时间间隔 
实现细节：
时间差计算：精确测量帧间隔
简单积分：一阶欧拉积分
返回预测：更新后的旋转向量
局限性：
简化模型，忽略加速度和噪声
只预测旋转，未预测平移
未考虑传感器误差和漂移
区块9：关键帧管理
def manage_keyframes(self, des, r, t): if len(self.keyframes) == 0: keyframe = {'descriptors': des, 'rvec': r.copy(), 'tvec': t.copy(), 'time': time.time()} self.keyframes.append(keyframe) else: last_kf = self.keyframes[-1] trans_distance = np.linalg.norm(t - last_kf['tvec']) if trans_distance > 0.1: keyframe = {'descriptors': des, 'rvec': r.copy(), 'tvec': t.copy(), 'time': time.time()} self.keyframes.append(keyframe) 
作用：
选择性保存重要帧，构建场景地图
为闭环检测和重定位提供依据
选择策略：
第一个关键帧：无条件保存
后续关键帧：移动超过阈值(0.1m)才保存
设计考虑：
距离阈值：平衡地图密度和存储效率
.copy()：避免数据修改影响原始数据
字典结构：灵活存储多种信息
实际应用扩展：
可添加旋转变化阈值
可考虑特征点数量和质量
可实现关键帧淘汰机制
第四部分：可视化与输出
区块10：AR渲染显示
def render_ar(self, frame, r, t): # 状态显示 color_map = {"INIT": (0,255,255), "TRACKING": (0,255,0), "LOST": (0,0,255)} color = color_map.get(self.state, (255,255,255)) cv2.putText(frame, f"STATE: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) # 位姿显示 cv2.putText(frame, f"R: [{r[0,0]:.2f}, {r[1,0]:.2f}, {r[2,0]:.2f}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1) cv2.putText(frame, f"T: [{t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f}]", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1) # 关键帧数量 cv2.putText(frame, f"Keyframes: {len(self.keyframes)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1) # 显示图像 cv2.imshow("AR System Manager", frame) 
作用：
将系统状态可视化呈现
提供调试信息和用户反馈
显示层次：
状态指示：用颜色编码系统状态
位姿数值：显示精确的位置和朝向
系统信息：显示关键帧数量等统计信息
颜色编码原则：
绿色(TRACKING)：正常，系统健康
红色(LOST)：异常，需要注意
黄色(INIT)：初始化，等待启动
白色(其他)：未知状态
设计优点：
实时反馈，便于调试
信息分层，清晰易读
颜色编码，直观理解
区块11：退出控制
if cv2.waitKey(1) & 0xFF == ord('q'): print("系统正常退出") cv2.destroyAllWindows() exit(0) 
作用：
提供用户控制接口
优雅地退出程序
实现原理：
cv2.waitKey(1)：等待1ms并检查按键
& 0xFF：确保跨平台兼容性
ord('q')：将字符转换为ASCII码
退出流程：
检测到'q'键 → 打印退出信息 → 关闭所有窗口 → 退出程序 
第五部分：主程序入口
区块12：模拟运行示例
if __name__ == "__main__": ar_system = ARSystemManager() for i in range(10): frame = np.zeros((480, 640, 3), dtype=np.uint8) imu_gyro = np.array([[0.01], [0.02], [0.01]]) ar_system.run_step(frame, imu_gyro) time.sleep(0.1) 
作用：
提供使用示例
测试系统基本功能
模拟设置：
图像：480×640的黑色图像
IMU数据：模拟轻微旋转
循环次数：10次迭代
帧间隔：0.1秒（10fps）
测试目的：
验证状态机转移逻辑
测试基本功能流程
演示系统接口用法
实际应用时应替换为：
真实摄像头采集
真实IMU传感器数据
实时处理循环
系统架构总结
数据流
摄像头帧 → 特征提取 → 状态机处理 → 位姿估计 → 渲染显示 IMU数据 → → IMU预测 → → 状态反馈 
状态转移图
[INIT] │ ↓ (定位成功) [TRACKING] ←→ (视觉失败) → [LOST] │ │ ↓ (视觉成功) ↓ 正常追踪 IMU预测 + 重定位尝试 
设计模式应用
状态模式：不同状态有不同行为
策略模式：可切换不同的定位算法
观察者模式：状态变化触发相应动作
模板方法：定义处理流程框架
这个AR系统管理器展示了一个完整的视觉-惯性定位系统的基本架构，具有良好的扩展性和鲁棒性设计。
"""