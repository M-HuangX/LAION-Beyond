import numpy as np
import json
import torch
import open_clip
from PIL import Image
from scipy.optimize import linear_sum_assignment
import os
from tqdm import tqdm
import argparse
import random
from torch.utils.data import Dataset, DataLoader

# 加载NPZ文件
def load_npz(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

# 从路径中提取类名
def extract_classname(path):
    return path.split('/images/')[-1].split('/')[0]

# 获取唯一的类名列表
def get_unique_classnames(paths):
    return list(set(extract_classname(path) for path in paths))

# 加载OpenCLIP模型
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model, preprocess, tokenizer, device

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx])
            return self.preprocess(image)
        except Exception as e:
            print(f"Error processing image {self.image_paths[idx]}: {e}")
            return None

# 计算图像embeddings
def compute_image_embeddings(model, preprocess, image_paths, cluster_ids, device):
    embeddings = []
    unique_cluster_ids = np.unique(cluster_ids)
    random.seed(42)  # 设置随机种子以确保结果可重复
    
    for cluster_id in tqdm(unique_cluster_ids, desc="Computing image embeddings"):
        cluster_paths = [path for path, cid in zip(image_paths, cluster_ids) if cid == cluster_id]
        
        # 如果聚类中的图片数量超过20，随机采样20张
        if len(cluster_paths) > 20:
            cluster_paths = random.sample(cluster_paths, 20)
        
        dataset = ImageDataset(cluster_paths, preprocess)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
        
        cluster_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is not None:
                    batch = batch.to(device)
                    embedding = model.encode_image(batch)
                    cluster_embeddings.append(embedding)
        
        if cluster_embeddings:
            cluster_embedding = torch.mean(torch.cat(cluster_embeddings, dim=0), dim=0)
            embeddings.append(cluster_embedding)
    
    return torch.stack(embeddings)

# 计算文本embeddings
def compute_text_embeddings(model, tokenizer, classnames, device):
    text = tokenizer([f"a photo of a {name}" for name in classnames]).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text)
    return text_embeddings

# 使用匈牙利算法进行匹配
def match_clusters_to_classnames(image_embeddings, text_embeddings):
    similarity = (image_embeddings @ text_embeddings.T).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-similarity)
    return row_ind, col_ind

# 主函数
def main(args):
    # 构建NPZ文件路径
    npz_file_path = os.path.join(args.laion_beyond_root, f'{args.subfield}_filtered_clustering_results_kmeans.npz')
    
    # 加载NPZ文件
    data = load_npz(npz_file_path)
    image_paths = data['i_paths'].tolist()  # 转换为 Python 列表
    cluster_ids = data['preds']
    
    # 获取唯一的类名
    classnames = get_unique_classnames(image_paths)
    
    ## 加载CLIP模型
    model, preprocess, tokenizer, device = load_clip_model()
    model.eval()
    
    # 计算embeddings
    image_embeddings = compute_image_embeddings(model, preprocess, image_paths, cluster_ids, device)
    text_embeddings = compute_text_embeddings(model, tokenizer, classnames, device)
    
    # 匹配聚类和类名
    row_ind, col_ind = match_clusters_to_classnames(image_embeddings, text_embeddings)
    
    # 创建聚类ID到类名的映射
    cluster_to_classname = {row: classnames[col] for row, col in zip(row_ind, col_ind)}
    
    # 为未匹配的聚类分配一个默认类名
    max_cluster_id = max(cluster_ids)
    for i in range(max_cluster_id + 1):
        if i not in cluster_to_classname:
            cluster_to_classname[i] = "unknown"
    
    # 计算每个类别的图片数量
    class_counts = {}
    for cluster_id in cluster_ids:
        classname = cluster_to_classname[cluster_id]
        if classname != "unknown":
            class_counts[classname] = class_counts.get(classname, 0) + 1
    
    # 找出符合条件的类别（至少20张图片）
    valid_classes = {classname for classname, count in class_counts.items() if count >= 20}
    
    # 创建结果列表，只包含非"unknown"且符合数量要求的结果
    results = []
    for path, cluster_id in zip(image_paths, cluster_ids):
        classname = cluster_to_classname[cluster_id]
        if classname != "unknown" and classname in valid_classes:
            results.append({
                "image_path": path,
                "pseudo_label": int(cluster_id),
                "pseudo_classname": classname
            })
    
    # 构建输出JSON文件路径
    output_json_path = os.path.join(args.laion_beyond_root, f'{args.subfield}_clustering_results_kmeans.json')
    
    # 保存为JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_json_path}")

# 运行脚本
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process clustering results and match with classnames.")
    parser.add_argument("--laion_beyond_root", type=str, default="/U_PZL2021KF0012/hx/LAION_Beyond_5B/", 
                        help="Root directory of LAION Beyond dataset")
    parser.add_argument("--subfield", type=str, required=True, 
                        help="Subfield name for processing (e.g., 'Pokemon')")
    
    args = parser.parse_args()
    main(args)