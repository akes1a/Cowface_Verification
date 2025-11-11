import torch
import torchvision.models as models
import torch.nn as nn

class ResNetFeatureExtractor:
    def __init__(self, model_name="resnet50", pretrained=True, device="cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = self._build_model().to(self.device)
        self._freeze_layers()
        
    def _build_model(self):
        """构建ResNet模型"""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == "resnet101":
            model = models.resnet101(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # 移除最后的分类层
        model = nn.Sequential(*list(model.children())[:-1])
        return model
    
    def _freeze_layers(self):
        """冻结模型的某些层以避免训练时更新权重"""
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, image_batch):
        """提取特征向量"""
        image_batch = image_batch.to(self.device)  # 确保输入张量与模型在同一设备上
        with torch.no_grad():
            features = self.model(image_batch)
            return features.squeeze()