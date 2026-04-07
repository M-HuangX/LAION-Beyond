import json
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_classname(path):
    return path.split('/images/')[-1].split('/')[0]

def analyze_pseudo_labels(json_file_path, laion_beyond_root, subfield):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    true_labels = []
    pseudo_labels = []
    class_matches = defaultdict(lambda: {'correct': 0, 'total': 0})

    # 提取标签并计算每个类别的正确匹配
    for item in data:
        true_class = extract_classname(item['image_path'])
        pseudo_class = item['pseudo_classname']
        
        true_labels.append(true_class)
        pseudo_labels.append(pseudo_class)
        
        class_matches[true_class]['total'] += 1
        if true_class == pseudo_class:
            class_matches[true_class]['correct'] += 1

    # 计算总体准确率
    accuracy = accuracy_score(true_labels, pseudo_labels)
    

    # 计算每个类别的准确率
    class_accuracies = {}
    for class_name, counts in class_matches.items():
        class_accuracy = counts['correct'] / counts['total']
        class_accuracies[class_name] = class_accuracy
        print(f"Accuracy for class {class_name}: {class_accuracy:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    unique_classes = sorted(set(true_labels + pseudo_labels))
    conf_matrix = confusion_matrix(true_labels, pseudo_labels, labels=unique_classes)

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(laion_beyond_root, f'{subfield}_confusion_matrix.png'))
    plt.close()

    # 绘制类别准确率条形图
    plt.figure(figsize=(12, 6))
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    plt.bar(classes, accuracies)
    plt.title('Class-wise Accuracies')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(laion_beyond_root, f'{subfield}_assign_class_accuracies.png'))
    plt.close()

if __name__ == "__main__":
    laion_beyond_root = "/202331510162/llm/hx/LAION_Beyond/"
    subfield = 'Pokemon'
    json_file_path = os.path.join(laion_beyond_root, f'{subfield}_clustering_results.json')  # 请替换为您的JSON文件路径
    analyze_pseudo_labels(json_file_path, laion_beyond_root, subfield)
    