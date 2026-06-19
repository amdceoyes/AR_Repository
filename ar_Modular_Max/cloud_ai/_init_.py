# 统一管理云端配置
HOST = '0.0.0.0'
PORT = 8888
MODEL_PATH = 'models/mobilenet.onnx'

from .ai_engine import AIEngine
from .FSM_cloud import CloudFSM
from .Transport_cloud import CloudTransport