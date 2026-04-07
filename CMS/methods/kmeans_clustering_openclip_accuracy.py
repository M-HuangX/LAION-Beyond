import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import Counter
from copy import deepcopy

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from project_utils.general_utils import str2bool
from models.openclip_original import OpenCLIP_original
from config import laion_beyond_root

import pandas as pd
from datetime import datetime
import csv


def extract_features(model, loader, device):
    features = []
    targets = []
    uq_idxs = []
    i_paths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images, label, batch_uq_idxs, batch_i_paths = batch
            images = images.to(device)
            feat = model(images).cpu().numpy()
            features.append(feat)
            targets.extend(label.numpy())
            uq_idxs.extend(batch_uq_idxs.numpy())
            i_paths.extend(batch_i_paths)
    
    return np.vstack(features), np.array(targets), np.array(uq_idxs), np.array(i_paths)

def kmeans_clustering(features, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(features)#输出每个样本被分配到的簇的编号，后续需要与其对应target来做比较

def subsample_dataset(dataset, indices):
    new_dataset = deepcopy(dataset)
    new_dataset.data = [dataset.data[i] for i in indices]
    new_dataset.targets = [dataset.targets[i] for i in indices]
    new_dataset.uq_idxs = [dataset.uq_idxs[i] for i in indices]
    return new_dataset



def cluster_acc(y_true, y_pred):
    """
    计算聚类准确度
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确度
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = confusion_matrix(y_true, y_pred, labels=range(D))
    '''
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 真实标签
    y_pred = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1, 1])  # 预测标签
    混淆矩阵解释:
    w = [
        [0, 0, 3],  # 真实标签0的3个样本被预测为标签2
        [3, 0, 0],  # 真实标签1的3个样本被预测为标签0
        [0, 4, 0]   # 真实标签2的4个样本被预测为标签1
    ]
    '''
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def evaluate_clustering(features, targets, n_clusters):
    """
    执行聚类并评估结果
    :param features: 特征
    :param targets: 真实标签
    :param n_clusters: 聚类数量
    :return: 聚类预测结果和准确度
    """
    preds = kmeans_clustering(features, n_clusters)
    acc = cluster_acc(targets, preds)
    return preds, acc

def evaluate_split_clustering(preds, targets, num_labeled_classes):
    """
    分别评估旧类和新类的聚类准确度
    :param preds: 聚类预测结果
    :param targets: 真实标签
    :param num_labeled_classes: 旧类的数量
    :return: 旧类准确度和新类准确度
    """
    # 创建旧类和新类的掩码
    old_mask = targets < num_labeled_classes
    new_mask = ~old_mask
    
    # 计算旧类准确度
    if np.any(old_mask):
        old_acc = cluster_acc(targets[old_mask], preds[old_mask])
    else:
        old_acc = 0.0
    
    # 计算新类准确度
    if np.any(new_mask):
        new_acc = cluster_acc(targets[new_mask], preds[new_mask])
    else:
        new_acc = 0.0
    
    return old_acc, new_acc



def save_experiment_results(args, results_dict, log_dir="/U_PZL2021KF0012/hx/GNCD_DATA/log"):
    """
    保存实验结果
    :param args: 参数
    :param results_dict: 包含实验结果的字典
    :param log_dir: 日志目录
    """
    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建实验结果字典，包含所有需要记录的信息
    experiment_data = {
        'timestamp': timestamp,
        'subfield': args.subfield,
        'dataset': args.dataset_name,
        'num_labeled_classes': args.num_labeled_classes,
        'num_unlabeled_classes': args.num_unlabeled_classes,
        'total_accuracy': results_dict['total_acc'],
        'old_classes_accuracy': results_dict['old_acc'],
        'new_classes_accuracy': results_dict['new_acc'],
        'old_only_accuracy': results_dict['old_only_acc'],
        'new_only_accuracy': results_dict['new_only_acc']
    }
    
    # 为当前实验创建单独的CSV文件
    individual_file = os.path.join(log_dir, f"{args.subfield}_{timestamp}.csv")
    pd.DataFrame([experiment_data]).to_csv(individual_file, index=False)
    print(f"\nExperiment results saved to: {individual_file}")
    
    # 更新汇总CSV文件
    summary_file = os.path.join(log_dir, 'experiments_summary.csv')
    
    # 检查汇总文件是否存在
    if not os.path.exists(summary_file):
        # 如果不存在，创建新文件并写入表头
        pd.DataFrame([experiment_data]).to_csv(summary_file, index=False)
    else:
        # 如果存在，追加数据
        with open(summary_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=experiment_data.keys())
            writer.writerow(experiment_data)
    
    print(f"Results appended to summary file: {summary_file}")



def main(args):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # Model setup
    model = OpenCLIP_original(args)
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # state_dict = torch.load(args.model_name)
    # model.load_state_dict(state_dict['model_state_dict'], strict=False)

    # Dataset setup
    _, test_transform = get_transform(args.transform, image_size=224, args=args)
    _, test_dataset, _, _ = get_datasets(args.dataset_name, test_transform, test_transform, args)
    test_dataset.uq_idxs = list(range(len(test_dataset.uq_idxs)))

    ##划分总体数据的5/6的IV数据，和OOV数据
    old_indices = [idx for idx, target in enumerate(test_dataset.targets) if target < args.num_labeled_classes]
    new_indices = [idx for idx, target in enumerate(test_dataset.targets) if target >= args.num_labeled_classes]
    
    old_test_dataset = subsample_dataset(test_dataset, old_indices)
    new_test_dataset = subsample_dataset(test_dataset, new_indices)
    
    # 对完整测试集进行聚类---------------------------------------------------------
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_features, test_targets, test_uq_idxs, test_i_paths = extract_features(model, test_loader, device)

    #128 for Pokemon
    n_clusters = args.num_labeled_classes + args.num_unlabeled_classes
    print(f"Performing K-means++ clustering with {n_clusters} clusters...")
    preds = kmeans_clustering(test_features, n_clusters)
    
    # 在完整测试集上评估三种聚类准确率
    total_acc = cluster_acc(test_targets, preds)#计算被正确分配的标签的概率
    old_acc, new_acc = evaluate_split_clustering(preds, test_targets, args.num_labeled_classes)
    
    print("\nClustering Results on Complete Test Set:")
    print(f"Overall Accuracy: {total_acc:.4f}")
    print(f"Old Classes Accuracy: {old_acc:.4f}")
    print(f"New Classes Accuracy: {new_acc:.4f}")
    
    # 评估旧类测试集----------------------------------------------------------------
    old_loader = DataLoader(old_test_dataset, batch_size=args.batch_size, 
                          num_workers=args.num_workers, shuffle=False)
    old_features, old_targets, _, _ = extract_features(model, old_loader, device)
    _, old_only_acc = evaluate_clustering(old_features, old_targets, args.num_labeled_classes)
    
    print("\nClustering Results on Old Classes Only:")
    print(f"Accuracy: {old_only_acc:.4f}")
    
    # 评估新类测试集
    new_loader = DataLoader(new_test_dataset, batch_size=args.batch_size, 
                          num_workers=args.num_workers, shuffle=False)
    new_features, new_targets, _, _ = extract_features(model, new_loader, device)
    _, new_only_acc = evaluate_clustering(new_features, new_targets, args.num_unlabeled_classes)
    
    print("\nClustering Results on New Classes Only:")
    print(f"Accuracy: {new_only_acc:.4f}")

    # 在完成所有评估后，收集结果
    results_dict = {
        'total_acc': total_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'old_only_acc': old_only_acc,
        'new_only_acc': new_only_acc
    }
    
    # 保存实验结果
    save_experiment_results(args, results_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-means++ clustering', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pretrain_path', type=str, default='/U_PZL2021KF0012/hx/GCD_DATA/models')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='/U_PZL2021KF0012/hx/GCD_DATA/log/cms/aircraft/Animals/checkpoints/model_best.pt')
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--feat_dim', default=768, type=int)
    parser.add_argument('--openclip_model', type=str, default='ViT-B-16', help='OpenCLIP model name')
    parser.add_argument('--openclip_pretrained', type=str, default='laion400m_e32', help='OpenCLIP pretrained dataset')
    parser.add_argument('--subfield', type=str, default='Animals')

    args = parser.parse_args()
    args = get_class_splits(args)
    
    args.num_labeled_classes = len(args.train_classes)#IV的类别标签0~39
    args.num_unlabeled_classes = len(args.unlabeled_classes)#OOV的类别标签39~128
    args.interpolation = 3
    args.crop_pct = 0.875
    args.prop_train_labels = 0.5
    args.image_size = 224

    print(args)

    main(args)