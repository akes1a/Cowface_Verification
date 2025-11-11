import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    def __init__(self, method="cosine"):
        self.method = method
        
    def compute_similarity(self, features1, features2):
        """计算特征相似度"""
        if self.method == "cosine":
            return self._cosine_similarity(features1, features2)
        elif self.method == "euclidean":
            return self._euclidean_similarity(features1, features2)
        else:
            raise ValueError(f"Unsupported similarity method: {self.method}")
    
    def _cosine_similarity(self, features1, features2):
        """余弦相似度"""
        return cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    
    def _euclidean_similarity(self, features1, features2):
        """基于欧氏距离的相似度"""
        distance = np.linalg.norm(features1 - features2)
        return 1 / (1 + distance)  # 转换为相似度