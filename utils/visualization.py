import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

def plot_similarity_distribution(similarities, labels, save_path=None):
    """绘制相似度分布图"""
    plt.figure(figsize=(10, 6))
    
    pos_similarities = [sim for sim, label in zip(similarities, labels) if label == 1]
    neg_similarities = [sim for sim, label in zip(similarities, labels) if label == 0]
    
    plt.hist(pos_similarities, bins=50, alpha=0.7, label='Positive Pairs', color='green')
    plt.hist(neg_similarities, bins=50, alpha=0.7, label='Negative Pairs', color='red')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution for Positive/Negative Pairs')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_proba, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_threshold_analysis(y_true, y_proba, save_path=None):
    """绘制阈值分析图"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision-Recall vs Threshold')
    
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, f1_scores[:-1], label='F1-score')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.title('F1-score vs Threshold')
    
    plt.subplot(1, 3, 3)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()