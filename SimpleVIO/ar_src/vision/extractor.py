import cv2
import numpy as np

class ORBExtractor:
    def __init__(self, n_features=1000):
        # 初始化 ORB 探测器，n_features 是提取特征点的上限
        # 大二学生优化建议：手机跑不动可以调低到 500
        self.orb = cv2.ORB_create(nfeatures=n_features)
        
        # 定义暴力匹配器 (Brute-Force Matcher)
        # NORM_HAMMING 是 ORB 描述子专用的匹配度量方式
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, image):
        """
        [数据流]: 原始图像 -> 特征点 + 描述子
        """
        if image is None:
            return None, None
        
        # 检测特征点并计算描述子
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match(self, des1, des2):
        """
        [数据流]: 两个描述子 -> 匹配结果对
        """
        if des1 is None or des2 is None:
            return []
        
        # 进行匹配并按距离（相似度）排序
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches