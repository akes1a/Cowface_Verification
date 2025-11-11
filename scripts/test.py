# scripts/test.py
import os
import sys
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.visualization import plot_similarity_distribution, plot_confusion_matrix

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_data(config):
    """加载测试数据"""
    data_root = config['data']['root_path']
    test_csv_path = os.path.join(data_root, 'test-0930.csv')
    test_image_dir = os.path.join(data_root, 'archive', 'test', 'test')
    
    # 读取测试CSV文件
    test_df = pd.read_csv(test_csv_path)
    
    # 构建测试图像对
    test_pairs = []
    pair_ids = []
    
    for _, row in test_df.iterrows():
        # 根据CSV文件的实际列名调整
        if 'ID_ID' in row:
            pair_id = row['ID_ID']
        elif 'pair_id' in row:
            pair_id = row['pair_id']
        else:
            # 如果没有找到标准列名，使用第一列
            pair_id = row.iloc[0]
            
        img1_id, img2_id = pair_id.split('_')
        
        # 构建图像路径
        img1_path = os.path.join(test_image_dir, f"{img1_id}.jpg")
        img2_path = os.path.join(test_image_dir, f"{img2_id}.jpg")
        
        test_pairs.append((img1_path, img2_path))
        pair_ids.append(pair_id)
    
    return test_pairs, pair_ids, test_df

def extract_features_batch(image_paths, feature_extractor, batch_size=32):
    """批量提取特征"""
    features = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 分批处理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取测试图像特征"):
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
        batch_tensor = torch.stack(batch_images).to(feature_extractor.device)
        
        # 提取特征
        with torch.no_grad():
            batch_features = feature_extractor.extract_features(batch_tensor)
            features.extend(batch_features.cpu().numpy())
    
    return features

def main():
    # 加载配置
    config = load_config()
    print("配置加载完成")
    
    # 加载训练好的模型
    model_path = os.path.join('saved_models', 'cowface_model.pkl')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在，请先运行 train.py")
        return
    
    print("加载训练好的模型...")
    model_data = joblib.load(model_path)
    
    lr_model = model_data['lr_model']
    feature_extractor = model_data['feature_extractor']
    similarity_calculator = model_data['similarity_calculator']
    
    print("模型加载完成")
    
    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 将特征提取器移动到设备
    feature_extractor.device = device
    feature_extractor.model = feature_extractor.model.to(device)
    
    # 加载测试数据
    print("加载测试数据...")
    test_pairs, pair_ids, test_df = load_test_data(config)
    print(f"找到 {len(test_pairs)} 个测试对")
    
    # 提取测试图像特征
    test_image_paths = list(set([path for pair in test_pairs for path in pair]))
    test_features_dict = {}
    
    test_features_list = extract_features_batch(
        test_image_paths, 
        feature_extractor,
        batch_size=config['training'].get('batch_size', 32)
    )
    
    for img_path, feature in zip(test_image_paths, test_features_list):
        test_features_dict[img_path] = feature
    
    # 预测测试对
    print("预测测试对...")
    predictions = []
    prediction_probas = []
    
    for img1_path, img2_path in tqdm(test_pairs):
        feat1 = test_features_dict[img1_path]
        feat2 = test_features_dict[img2_path]
        similarity = similarity_calculator.compute_similarity(feat1, feat2)
        prob = lr_model.predict_proba([[similarity]])[0][1]
        prediction_probas.append(prob)
        predictions.append(1 if prob > 0.5 else 0)
    
    # 生成提交文件
    submission_df = test_df.copy()
    submission_df['TARGET'] = predictions
    #submission_df['probability'] = prediction_probas  # 可选：保存概率值
    
    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)
    submission_path = os.path.join('results', 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"提交文件已生成: {submission_path}")
    
    # 输出预测统计
    print(f"\n预测统计:")
    print(f"正样本 (1): {sum(predictions)}")
    print(f"负样本 (0): {len(predictions) - sum(predictions)}")
    
    # 保存测试特征（可选，用于后续分析）
    test_features_path = os.path.join('saved_models', 'test_features.pkl')
    joblib.dump(test_features_dict, test_features_path)
    print(f"测试特征已保存到 {test_features_path}")
    
    # 可视化相似度分布
    plot_similarity_distribution(prediction_probas, predictions, save_path='results/similarity_distribution_test.png')
    print("相似度分布图已保存到 results/similarity_distribution_test.png")

    # 可视化混淆矩阵
    plot_confusion_matrix(predictions, submission_df['TARGET'], save_path='results/confusion_matrix_test.png')
    print("混淆矩阵已保存到 results/confusion_matrix_test.png")

if __name__ == '__main__':
    main()