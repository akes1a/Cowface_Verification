class CowFaceModel:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = None
        self.similarity_calculator = None
        self.classifier = None
        self._build_model()
        
    def _build_model(self):
        """构建完整模型"""
        from feature_extra.resnet import ResNetFeatureExtractor
        from classifiers.simu_cal import SimilarityCalculator
        from classifiers.logi import LogisticClassifier
        
        self.feature_extractor = ResNetFeatureExtractor(
            self.config['model']['feature_extractor']
        )
        self.similarity_calculator = SimilarityCalculator(
            self.config['model']['similarity']
        )
        self.classifier = LogisticClassifier(
            self.config['model']['classifier']
        )
        
    def train(self, train_pairs, train_labels):
        """训练完整模型"""
        pass
        
    def predict(self, image_pairs):
        """预测图像对"""
        pass