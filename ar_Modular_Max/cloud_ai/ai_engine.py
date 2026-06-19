# 导入必要的Python库
import onnxruntime as ort  # ONNX运行时，用于加载和运行ONNX格式的模型
import numpy as np         # NumPy库，用于高效的数值计算和数组操作
import cv2                 # OpenCV库，用于图像处理和编解码
import json                # JSON库，用于处理JSON格式的数据

class AIEngine:
    """
    AI引擎类（ONNX版本），使用ONNX Runtime进行推理
    
    这个类封装了ONNX模型的推理流程：
    1. 加载ONNX模型
    2. 对输入的图像进行预处理
    3. 运行ONNX模型推理
    4. 返回JSON格式的结果
    """
    
    def __init__(self, model_path):
        """
        初始化：载入 ONNX 模型
        
        参数:
        model_path: ONNX模型文件的路径（.onnx格式）
        """
        print("[AI] 正在载入 ONNX 模型...")
        
        # 创建ONNX Runtime推理会话
        # ort.InferenceSession: 加载ONNX模型并创建推理会话
        # 这个会话用于运行模型推理
        # ONNX Runtime会自动选择可用的执行提供者（CPU、CUDA、TensorRT等）
        self.session = ort.InferenceSession(model_path)
        
        # 获取模型输入的名称
        # get_inputs(): 返回模型的所有输入信息
        # [0]: 取第一个输入（假设模型只有一个输入）
        # .name: 获取输入的名称
        self.input_name = self.session.get_inputs()[0].name
        
        # 可选：打印模型输入输出信息，便于调试
        print(f"[AI] 模型输入名称: {self.input_name}")
        print(f"[AI] 模型输入形状: {self.session.get_inputs()[0].shape}")
        print(f"[AI] 模型输出数量: {len(self.session.get_outputs())}")
        
        print("[AI] 模型载入成功，准备就绪。")

    def predict(self, image_bytes):
        """
        核心推理：接收二进制图片数据，返回识别结果
        
        参数:
        image_bytes: 图片的二进制数据（通常是JPEG格式的字节流）
        
        返回:
        推理结果的JSON字符串
        成功时返回: {"result": "success", "data": [预测结果列表]}
        失败时返回: {"error": 错误信息}
        """
        try:
            # 1. 图像解码与缩放
            
            # 将字节数据转换为NumPy数组
            # np.frombuffer: 从字节缓冲区创建数组
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # 解码JPEG图片
            # cv2.imdecode: 从内存缓冲区解码图像
            # cv2.IMREAD_COLOR: 以彩色模式读取（BGR格式）
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 检查图片是否成功解码
            if img is None:
                return json.dumps({"error": "图片解码失败"})
            
            # 调整图片大小
            # 假设模型需要224x224的输入，根据实际模型调整
            img = cv2.resize(img, (224, 224))
            
            # 2. 预处理：将图片转换为模型需要的格式
            
            # 将像素值从0-255归一化到0-1之间
            # 这是大多数深度学习模型的标准预处理步骤
            input_data = img.astype(np.float32) / 255.0
            
            # 转换维度顺序
            # OpenCV的默认格式是(H, W, C) [高度, 宽度, 通道]
            # 转换为(C, H, W) [通道, 高度, 宽度]
            # 这是大多数深度学习框架（如PyTorch、ONNX）的格式
            input_data = np.transpose(input_data, (2, 0, 1))
            
            # 添加批次维度
            # 从(C, H, W)转换为(1, C, H, W) [批次大小, 通道, 高度, 宽度]
            # 批次维度表示一次处理多少张图片
            input_data = np.expand_dims(input_data, axis=0)
            
            # 3. 运行推理
            
            # 运行ONNX模型
            # self.session.run(): 执行模型推理
            # 参数:
            #   None: 表示返回所有输出
            #   {self.input_name: input_data}: 输入字典，键是输入名称，值是输入数据
            # 返回值: 包含所有输出的列表
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # 4. 返回结果
            
            # 创建结果字典
            # 假设模型只有一个输出，取outputs[0]
            # .tolist(): 将NumPy数组转换为Python列表（可序列化为JSON）
            result_dict = {
                "result": "success",  # 表示推理成功
                "data": outputs[0].tolist()  # 推理结果
            }
            
            # 将字典转换为JSON字符串
            return json.dumps(result_dict)
            
        except Exception as e:
            # 捕获并处理所有异常
            error_msg = f"推理过程中发生错误: {str(e)}"
            print(f"[AI] {error_msg}")
            
            # 返回错误信息的JSON字符串
            return json.dumps({"error": error_msg})