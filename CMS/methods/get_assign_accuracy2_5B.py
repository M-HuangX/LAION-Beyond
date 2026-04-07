import json
import os
import csv
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

    # 计算混淆矩阵
    unique_classes = sorted(set(true_labels + pseudo_labels))
    conf_matrix = confusion_matrix(true_labels, pseudo_labels, labels=unique_classes)

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Confusion Matrix - {subfield}')
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
    plt.title(f'Class-wise Accuracies - {subfield}')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(laion_beyond_root, f'{subfield}_assign_class_accuracies.png'))
    plt.close()

    return accuracy, len(data)

def analyze_all_subfields(laion_beyond_root, subfields):
    results = []
    for subfield in subfields:
        json_file_path = os.path.join(laion_beyond_root, f'{subfield}_clustering_results.json')
        if os.path.exists(json_file_path):
            accuracy, num_samples = analyze_pseudo_labels(json_file_path, laion_beyond_root, subfield)
            results.append({
                'Subfield': subfield,
                'Accuracy': accuracy,
                'Number of Samples': num_samples
            })
            print(f"Subfield: {subfield}, Accuracy: {accuracy:.4f}, Number of Samples: {num_samples}")
        else:
            print(f"Warning: JSON file for {subfield} not found.")

    # 保存结果到CSV文件
    csv_file_path = os.path.join(laion_beyond_root, 'classname_assign_accuracies.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Subfield', 'Accuracy', 'Number of Samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {csv_file_path}")

if __name__ == "__main__":
    laion_beyond_root = "/202331510162/llm/hx/LAION_Beyond_5B/"
    subfields = [
        'Animals', 'Insects_Spiders', 'Plants_Fugi', 'Pokemon'
    ]
    analyze_all_subfields(laion_beyond_root, subfields)