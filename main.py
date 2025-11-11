import argparse
from scripts.train import main as train_main
from scripts.test import main as test_main

def main():
    parser = argparse.ArgumentParser(description='牛脸识别系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--model', required=True, help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_main()
    elif args.command == 'test':
        test_main()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()