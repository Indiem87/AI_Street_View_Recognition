import json
import os
from collections import Counter

def analyze_json(json_path, dataset_name):
    print(f"--- 分析 {dataset_name} 数据集 ---")
    
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"图片总数: {len(data)}")
    
    # 统计字符长度分布
    length_counts = Counter()
    max_len = 0
    min_len = float('inf')
    
    for img_name, info in data.items():
        labels = info['label']
        # 确保 labels 是列表
        if not isinstance(labels, list):
            labels = [labels]
            
        length = len(labels)
        length_counts[length] += 1
        
        if length > max_len:
            max_len = length
        if length < min_len:
            min_len = length
            
    print(f"字符数量最少: {min_len}")
    print(f"字符数量最多: {max_len}")
    print("字符长度分布:")
    for length, count in sorted(length_counts.items()):
        print(f"  长度为 {length} 的图片有: {count} 张 ({count/len(data)*100:.2f}%)")
    print("\n")

if __name__ == '__main__':
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 tcdata 目录的绝对路径
    tcdata_dir = os.path.abspath(os.path.join(current_dir, '../../tcdata'))
    
    train_json = os.path.join(tcdata_dir, 'mchar_train.json')
    val_json = os.path.join(tcdata_dir, 'mchar_val.json')
    
    analyze_json(train_json, "训练集 (Train)")
    analyze_json(val_json, "验证集 (Validation)")
