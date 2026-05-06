import cv2
import numpy as np

class MapDatabase:
    def __init__(self):
        # 存储关键帧的列表，每个关键帧包含：描述子、3D点、当时的位姿
        self.keyframes = []
        # 定义匹配器，用于重定位时的比对
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def add_keyframe(self, descriptors, pose, points_3d):
        """
        [数据流]: 描述子 + 当前位姿 + 3D点 -> 存入记忆库
        """
        # 为了节省内存，我们只在特征点足够多时才存为关键帧
        if descriptors is not None and len(descriptors) > 50:
            self.keyframes.append({
                'des': descriptors,
                'pose': pose,
                'pts_3d': points_3d
            })
            print(f">>> [Mapping]: 新增关键帧，当前记忆库容量: {len(self.keyframes)}")

    def relocalize(self, current_des):
        """
        [数据流]: 当前帧描述子 -> 在记忆库中搜寻最匹配的历史瞬间
        返回: 匹配度最高的那一帧的位姿（如果有的话）
        """
        if not self.keyframes or current_des is None:
            return None

        best_match_idx = -1
        max_good_matches = 0

        # 遍历记忆库，寻找最像的那一帧
        for i, kf in enumerate(self.keyframes):
            matches = self.bf.match(current_des, kf['des'])
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_idx = i

        # 如果匹配点够多（比如超过30个），认为找回了位置
        if max_good_matches > 30:
            print(f">>> [重定位成功]: 匹配到第 {best_match_idx} 号关键帧")
            return self.keyframes[best_match_idx]['pose']
        
        return None