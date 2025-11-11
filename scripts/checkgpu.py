import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.feature_extra.resnet import ResNetFeatureExtractor
import torch, torchvision.transforms as transforms
from PIL import Image
print("CUDA available:", torch.cuda.is_available()) 
fe = ResNetFeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
print("Feature extractor device:", fe.device)
# 创建测试的图像张量 (1,3,224,224)
x = torch.zeros(1,3,224,224)
# 尝试提取特征
feat = fe.extract_features(x)
print("feature shape:", feat.shape)