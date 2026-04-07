import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from collections import Counter

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from project_utils.general_utils import str2bool
from models.openclip_original import OpenCLIP_original
from config import laion_beyond_root

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
    return kmeans.fit_predict(features)

def save_filtered_data(preds, targets, uq_idxs, i_paths, args):
    # 找出旧类预测中出现频次最高的标签
    old_class_mask = np.isin(targets, range(args.num_labeled_classes))
    old_class_preds = preds[old_class_mask]
    old_class_pred_counts = Counter(old_class_preds)
    most_common_old_labels = set([label for label, _ in old_class_pred_counts.most_common(args.num_labeled_classes)])

    # 筛选新类样本，且预测标签不在最常见的旧类标签中
    new_class_mask = ~old_class_mask
    valid_new_class_mask = new_class_mask & ~np.isin(preds, list(most_common_old_labels))

    # 根据筛选条件获取最终要保存的数据
    filtered_preds = preds[valid_new_class_mask]
    filtered_targets = targets[valid_new_class_mask]
    filtered_uq_idxs = uq_idxs[valid_new_class_mask]
    filtered_i_paths = i_paths[valid_new_class_mask]

    # 打印数据格式和统计信息
    print("Filtered preds format:", type(filtered_preds), filtered_preds.shape)
    print("Filtered targets format:", type(filtered_targets), filtered_targets.shape)
    print("Filtered uq_idxs format:", type(filtered_uq_idxs), filtered_uq_idxs.shape)
    print("Filtered i_paths format:", type(filtered_i_paths), filtered_i_paths.shape)
    print(f"Total samples: {len(preds)}")
    print(f"Old class samples: {np.sum(old_class_mask)}")
    print(f"New class samples: {np.sum(new_class_mask)}")
    print(f"Filtered new class samples: {np.sum(valid_new_class_mask)}")
    print(f"Most common old class labels: {most_common_old_labels}")
    print(f"Labels in filtered new class predictions: {np.unique(filtered_preds)}")
    print(f"Labels in filtered targets: {np.unique(filtered_targets)}")
    if len(filtered_i_paths) > 0:
        print(f"First few filtered i_paths: {filtered_i_paths[:5]}")

    # 将数据保存为 .npz 文件
    npz_path = os.path.join(laion_beyond_root, f'{args.subfield}_filtered_clustering_results_kmeans.npz')
    np.savez(npz_path, 
             preds=filtered_preds,
             targets=filtered_targets,
             uq_idxs=filtered_uq_idxs,
             i_paths=filtered_i_paths)

    print(f"Filtered data saved to {npz_path}")

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    # Model setup
    model = OpenCLIP_original(args)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # state_dict = torch.load(args.model_name)
    # model.load_state_dict(state_dict['model_state_dict'], strict=False)

    # Dataset setup
    _, test_transform = get_transform(args.transform, image_size=224, args=args)
    _, test_dataset, _, _ = get_datasets(args.dataset_name, test_transform, test_transform, args)
    test_dataset.uq_idxs = list(range(len(test_dataset.uq_idxs)))
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Extract features
    print("Extracting test set features...")
    test_features, test_targets, test_uq_idxs, test_i_paths = extract_features(model, test_loader, device)

    # Perform K-means++ clustering
    n_clusters = args.num_labeled_classes + args.num_unlabeled_classes
    print(f"Performing K-means++ clustering with {n_clusters} clusters...")
    preds = kmeans_clustering(test_features, n_clusters)
    save_filtered_data(preds, test_targets, test_uq_idxs, test_i_paths, args)

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
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.interpolation = 3
    args.crop_pct = 0.875
    args.prop_train_labels = 0.5
    args.image_size = 224

    print(args)

    main(args)