import os
import sys
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import joblib
from utils.visualization import plot_confusion_matrix, plot_roc_curve

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 从feature_extra和classifiers导入模块
from model.feature_extra.resnet import ResNetFeatureExtractor
from model.classifiers.simu_cal import SimilarityCalculator

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_data(config):
    """加载训练数据"""
    data_root = config['data']['root_path']
    train_dir = os.path.join(data_root, 'archive', 'train', 'train')
    
    image_paths = []
    labels = []
    
    # 遍历每个牛的文件夹
    for cow_folder in os.listdir(train_dir):
        cow_path = os.path.join(train_dir, cow_folder)
        if os.path.isdir(cow_path):
            # 获取该牛的所有图片
            for img_file in os.listdir(cow_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cow_path, img_file)
                    image_paths.append(img_path)
                    labels.append(cow_folder)  # 使用文件夹名作为标签
    
    return image_paths, labels

def create_training_pairs(image_paths, labels, num_pairs_per_class=1000):
    """创建训练图像对"""
    import random
    from itertools import combinations
    
    # 按标签分组
    label_to_images = {}
    for img_path, label in zip(image_paths, labels):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(img_path)
    
    positive_pairs = []
    negative_pairs = []
    
    # 生成正样本对（同一头牛）
    for label, images in label_to_images.items():
        if len(images) >= 2:
            # 生成所有可能的组合
            possible_pairs = list(combinations(images, 2))
            # 随机选择指定数量的对
            selected_pairs = random.sample(
                possible_pairs, 
                min(num_pairs_per_class, len(possible_pairs))
            )
            positive_pairs.extend(selected_pairs)
    
    # 生成负样本对（不同牛）
    all_labels = list(label_to_images.keys())
    for _ in range(len(positive_pairs)):
        label1, label2 = random.sample(all_labels, 2)
        img1 = random.choice(label_to_images[label1])
        img2 = random.choice(label_to_images[label2])
        negative_pairs.append((img1, img2))
    
    # 合并并创建标签
    all_pairs = positive_pairs + negative_pairs
    pair_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    return all_pairs, pair_labels

def extract_features_batch(image_paths, feature_extractor, batch_size=32):
    """批量提取特征"""
    features = []
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 分批处理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                batch_images.append(image)
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                # 添加一个空白图像作为占位符
                batch_images.append(torch.zeros(3, 224, 224))
        
        # 转换为张量
        batch_tensor = torch.stack(batch_images)
        
        # 提取特征
        with torch.no_grad():
            batch_features = feature_extractor.extract_features(batch_tensor)
            features.extend(batch_features.cpu().numpy())
    
    return features

def main():
    # 加载配置
    config = load_config()
    print("配置加载完成")
    
    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化特征提取器和相似度计算器
    feature_extractor = ResNetFeatureExtractor(device=device)
    similarity_calculator = SimilarityCalculator()
    
    # 加载训练数据
    print("加载训练数据...")
    image_paths, labels = load_training_data(config)
    print(f"找到 {len(image_paths)} 张训练图像，属于 {len(set(labels))} 头牛")
    
    # 创建训练对
    print("创建训练图像对...")
    train_pairs, train_labels = create_training_pairs(
        image_paths, 
        labels, 
        num_pairs_per_class=config['training'].get('num_pairs_per_class', 1000)
    )
    print(f"创建了 {len(train_pairs)} 个训练对 ({sum(train_labels)} 正样本, {len(train_labels)-sum(train_labels)} 负样本)")
    
    # 提取所有图像特征
    print("提取图像特征...")
    all_image_paths = list(set([path for pair in train_pairs for path in pair]))
    features_dict = {}
    
    features_list = extract_features_batch(
        all_image_paths, 
        feature_extractor,
        batch_size=config['training'].get('batch_size', 32)
    )
    
    for img_path, feature in zip(all_image_paths, features_list):
        features_dict[img_path] = feature
    
    # 计算训练对的相似度
    print("计算训练对相似度...")
    X_train = []
    for img1_path, img2_path in tqdm(train_pairs):
        feat1 = features_dict[img1_path]
        feat2 = features_dict[img2_path]
        similarity = similarity_calculator.compute_similarity(feat1, feat2)
        X_train.append(similarity)
    
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(train_labels)
    
    # 检查数据是否为空
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("训练数据为空，请检查数据加载和对生成步骤。")
    
    # 划分训练集和验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, 
        test_size=config['training'].get('validation_ratio', 0.2), 
        random_state=config['training'].get('random_seed', 42), 
        stratify=y_train
    )
    
    # 将逻辑回归模型移动到设备
    classifier_config = config['model']['classifier']['params']
    lr_model = LogisticRegression(**classifier_config)

    # 在训练和验证时确保张量在正确的设备上
    X_tr = torch.tensor(X_tr, device=device)
    X_val = torch.tensor(X_val, device=device)
    y_tr = torch.tensor(y_tr, device=device)
    y_val = torch.tensor(y_val, device=device)
    
    # 确保在将数据传递给 LogisticRegression 模型之前，将张量移动到 CPU 并转换为 NumPy 数组
    X_tr = X_tr.cpu().numpy()
    X_val = X_val.cpu().numpy()
    y_tr = y_tr.cpu().numpy()
    y_val = y_val.cpu().numpy()
    
    # 训练逻辑回归模型
    print("训练逻辑回归模型...")
    lr_model.fit(X_tr, y_tr)
    
    # 验证集评估
    y_pred = lr_model.predict(X_val)
    y_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    val_accuracy = accuracy_score(y_val, y_pred)
    
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    # 可视化混淆矩阵
    plot_confusion_matrix(y_val, y_pred, save_path='results/confusion_matrix_train.png')
    print("混淆矩阵已保存到 results/confusion_matrix_train.png")

    # 可视化 ROC 曲线
    plot_roc_curve(y_val, y_pred_proba, save_path='results/roc_curve_train.png')
    print("ROC 曲线已保存到 results/roc_curve_train.png")
    
    # 保存模型和特征字典
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'cowface_model.pkl')
    
    joblib.dump({
        'lr_model': lr_model,
        'feature_extractor': feature_extractor,
        'similarity_calculator': similarity_calculator,
        'features_dict': features_dict,  # 保存训练图像特征，便于后续分析
        'config': config
    }, model_path)
    
    print(f"模型已保存到 {model_path}")
    
    # 保存训练图像路径和特征的映射（用于测试时参考）
    features_mapping_path = os.path.join('saved_models', 'train_features_mapping.pkl')
    joblib.dump(features_dict, features_mapping_path)
    print(f"训练特征映射已保存到 {features_mapping_path}")

if __name__ == '__main__':
    main()