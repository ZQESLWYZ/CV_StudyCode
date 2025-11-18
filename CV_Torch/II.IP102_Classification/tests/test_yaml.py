import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.config import read_yaml

if __name__ == "__main__":
    # print(__file__)
    # print(os.path.abspath(__file__))
    # print(os.path.dirname(os.path.abspath(__file__)))
    # print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    file_path = os.path.join(project_root, "configs", "train_config.yaml")
    train_config = read_yaml(file_path)
    print(train_config)