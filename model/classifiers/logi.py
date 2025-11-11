from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

class LogisticClassifier:
    def __init__(self, params=None):
        self.params = params or {
            'C': 1.0,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000
        }
        self.model = LogisticRegression(**self.params)
        self.threshold = 0.5
        
    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X.reshape(-1, 1), y)
        
    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X.reshape(-1, 1))[:, 1]
    
    def predict(self, X, threshold=None):
        """预测类别"""
        threshold = threshold or self.threshold
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save_model(self, filepath):
        """保存模型"""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        self.model = joblib.load(filepath)