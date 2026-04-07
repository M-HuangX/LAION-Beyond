import os
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict
from torchvision.datasets import VisionDataset
from PIL import Image
import glob
from config import laion_beyond_root

class Custom_LAION_Beyond_Dataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(Custom_LAION_Beyond_Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        self.uq_idxs = []
        self.classnames = []

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[index], img_path

    def __len__(self):
        return len(self.data)

def find_dataset_info(root, category):
    oov_dir_pattern = os.path.join(root, f"{category}*_OOV")
    iv_dir_pattern = os.path.join(root, f"{category}*_IV")
    
    oov_dirs = glob.glob(oov_dir_pattern)
    iv_dirs = glob.glob(iv_dir_pattern)
    
    if not oov_dirs or not iv_dirs:
        raise ValueError(f"Cannot find OOV or IV directories for {category}")
    
    return oov_dirs[0], iv_dirs[0]

def load_merged_mapping(dir_path):
    with open(os.path.join(dir_path, 'merged_mapping.json'), 'r', encoding="utf-8") as f:
        return json.load(f)

def create_dataset(root, oov_dir, iv_dir):
    dataset = Custom_LAION_Beyond_Dataset(root)
    
    oov_mapping = load_merged_mapping(oov_dir)
    iv_mapping = load_merged_mapping(iv_dir)
    
    label = 0
    for mapping, is_oov in [(iv_mapping, False), (oov_mapping, True)]:
        for class_name, images in mapping.items():
            for img_name in images:
                img_path = os.path.join(oov_dir if is_oov else iv_dir, 'images', class_name, img_name)
                dataset.data.append(img_path)
                dataset.targets.append(label)
                dataset.uq_idxs.append(len(dataset.uq_idxs))
            dataset.classnames.append(class_name)
            label += 1
    
    return dataset

def split_dataset(dataset, split_ratio, by_class=True, seed=0):
    np.random.seed(seed)
    
    if by_class:
        class_indices = defaultdict(list)
        for idx, target in enumerate(dataset.targets):
            class_indices[target].append(idx)
        
        split_indices = []
        remaining_indices = []
        
        for indices in class_indices.values():
            np.random.shuffle(indices)
            split_point = int(len(indices) * split_ratio)
            split_indices.extend(indices[:split_point])
            remaining_indices.extend(indices[split_point:])
    else:
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split_point = int(len(indices) * split_ratio)
        split_indices = indices[:split_point]
        remaining_indices = indices[split_point:]
    
    return split_indices, remaining_indices

def subsample_dataset(dataset, indices):
    new_dataset = deepcopy(dataset)
    new_dataset.data = [dataset.data[i] for i in indices]
    new_dataset.targets = [dataset.targets[i] for i in indices]
    new_dataset.uq_idxs = [dataset.uq_idxs[i] for i in indices]
    return new_dataset

def get_laion_beyond_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0, subfield='General'):
    root = laion_beyond_root
    split_file = os.path.join(root, f'{subfield}_split_seed_{seed}.json')
    
    if os.path.exists(split_file):
        # Load existing split
        with open(split_file, 'r', encoding="utf-8") as f:
            split_info = json.load(f)
        
        oov_dir, iv_dir = find_dataset_info(root, subfield)
        full_dataset = create_dataset(root, oov_dir, iv_dir)
        
        test_dataset = subsample_dataset(full_dataset, split_info['test_indices'])
        train_labelled = subsample_dataset(full_dataset, split_info['labelled_indices'])
        train_unlabelled = subsample_dataset(full_dataset, split_info['unlabelled_indices'])
        
        if split_train_val:
            val_dataset = subsample_dataset(full_dataset, split_info['val_indices'])
        else:
            val_dataset = None
    else:
        # Create new split
        oov_dir, iv_dir = find_dataset_info(root, subfield)
        full_dataset = create_dataset(root, oov_dir, iv_dir)
        
        # Split into test (5/6) and remaining (1/6)
        test_indices, remaining_indices = split_dataset(full_dataset, 5/6, by_class=True, seed=seed)
        
        test_dataset = subsample_dataset(full_dataset, test_indices)
        remaining_dataset = subsample_dataset(full_dataset, remaining_indices)
        
        # Split remaining into labelled (old classes, 8/10) and unlabelled
        iv_mapping = load_merged_mapping(iv_dir)
        old_class_count = len(iv_mapping)
        old_indices = [idx for idx, target in enumerate(remaining_dataset.targets) if target < old_class_count]
        new_indices = [idx for idx, target in enumerate(remaining_dataset.targets) if target >= old_class_count]
        
        old_dataset = subsample_dataset(remaining_dataset, old_indices)
        labelled_indices, unlabelled_indices_old = split_dataset(old_dataset, 0.8, by_class=True, seed=seed)
        unlabelled_indices = unlabelled_indices_old + new_indices
        
        train_labelled = subsample_dataset(remaining_dataset, labelled_indices)
        train_unlabelled = subsample_dataset(remaining_dataset, unlabelled_indices)
        
        if split_train_val:
            train_indices, val_indices = split_dataset(train_labelled, 0.8, by_class=True, seed=seed)
            val_dataset = subsample_dataset(train_labelled, val_indices)
            train_labelled = subsample_dataset(train_labelled, train_indices)
        else:
            val_dataset = None
            val_indices = None
        
        # Save split information
        split_info = {
            'test_indices': test_indices,
            'labelled_indices': labelled_indices,
            'unlabelled_indices': unlabelled_indices,
            'val_indices': val_indices,
        }
        with open(split_file, 'w', encoding="utf-8") as f:
            json.dump(split_info, f)
    
    # Apply transforms
    train_labelled.transform = train_transform
    train_unlabelled.transform = train_transform
    if val_dataset:
        val_dataset.transform = test_transform
    test_dataset.transform = test_transform
    
    all_datasets = {
        'train_labelled': train_labelled,
        'train_unlabelled': train_unlabelled,
        'val': val_dataset,
        'test': test_dataset,
    }
    
    return all_datasets

def get_laion_beyond_datasets_with_gcdval(train_transform, test_transform, train_classes=range(80), 
                                   prop_train_labels=0.8, split_train_val=True, seed=0, val_split=0.1, subfield='General'):
    root = laion_beyond_root
    split_file = os.path.join(root, f'{subfield}_split_gcdval_seed_{seed}.json')
    
    if os.path.exists(split_file):
        # Load existing split
        with open(split_file, 'r', encoding="utf-8") as f:
            split_info = json.load(f)
        
        oov_dir, iv_dir = find_dataset_info(root, subfield)
        full_dataset = create_dataset(root, oov_dir, iv_dir)
        
        test_dataset = subsample_dataset(full_dataset, split_info['test_indices'])
        train_labelled = subsample_dataset(full_dataset, split_info['train_l_indices'])
        val_labelled = subsample_dataset(full_dataset, split_info['val_l_indices'])
        train_unlabelled = subsample_dataset(full_dataset, split_info['train_u_indices'])
        val_unlabelled = subsample_dataset(full_dataset, split_info['val_u_indices'])
    else:
        # Create new split
        oov_dir, iv_dir = find_dataset_info(root, subfield)
        full_dataset = create_dataset(root, oov_dir, iv_dir)
        
        # Split into test (5/6) and remaining (1/6)
        test_indices, remaining_indices = split_dataset(full_dataset, 1/6, by_class=True, seed=seed)
        
        test_dataset = subsample_dataset(full_dataset, test_indices)
        remaining_dataset = subsample_dataset(full_dataset, remaining_indices)
        
        # Split remaining into labelled (old classes, 8/10) and unlabelled
        iv_mapping = load_merged_mapping(iv_dir)
        old_class_count = len(iv_mapping)
        old_indices = [idx for idx, target in enumerate(remaining_dataset.targets) if target < old_class_count]
        new_indices = [idx for idx, target in enumerate(remaining_dataset.targets) if target >= old_class_count]
        
        old_dataset = subsample_dataset(remaining_dataset, old_indices)
        labelled_indices, unlabelled_indices_old = split_dataset(old_dataset, 0.8, by_class=True, seed=seed)
        unlabelled_indices = unlabelled_indices_old + new_indices
        
        train_labelled = subsample_dataset(remaining_dataset, labelled_indices)
        train_unlabelled = subsample_dataset(remaining_dataset, unlabelled_indices)
        
        # Further split train_labelled and train_unlabelled into train and val
        train_l_indices, val_l_indices = split_dataset(train_labelled, 0.9, by_class=True, seed=seed)
        train_u_indices, val_u_indices = split_dataset(train_unlabelled, 0.9, by_class=True, seed=seed)
        
        train_labelled = subsample_dataset(train_labelled, train_l_indices)
        val_labelled = subsample_dataset(train_labelled, val_l_indices)
        train_unlabelled = subsample_dataset(train_unlabelled, train_u_indices)
        val_unlabelled = subsample_dataset(train_unlabelled, val_u_indices)
        
        # Save split information
        split_info = {
            'test_indices': test_indices,
            'train_l_indices': train_l_indices,
            'val_l_indices': val_l_indices,
            'train_u_indices': train_u_indices,
            'val_u_indices': val_u_indices,
        }
        with open(split_file, 'w', encoding="utf-8") as f:
            json.dump(split_info, f)
    
    # Apply transforms
    train_labelled.transform = train_transform
    train_unlabelled.transform = train_transform
    val_labelled.transform = test_transform
    val_unlabelled.transform = test_transform
    test_dataset.transform = test_transform
    
    all_datasets = {
        'train_labelled': train_labelled,
        'train_unlabelled': train_unlabelled,
        'val': [val_labelled, val_unlabelled],
        'test': test_dataset,
    }
    
    return all_datasets

if __name__ == '__main__':
    # Example usage
    root = "/path/to/LAION_Beyond"
    train_transform = None  # Define your train transform
    test_transform = None   # Define your test transform
    
    datasets = get_laion_beyond_datasets(root, train_transform, test_transform, split_train_val=True, seed=42)
    
    for name, dataset in datasets.items():
        if dataset is not None:
            print(f"{name}: {len(dataset)}")
    
    datasets_gcdval = get_laion_beyond_datasets_with_gcdval(root, train_transform, test_transform, seed=42)
    
    for name, dataset in datasets_gcdval.items():
        if isinstance(dataset, list):
            print(f"{name}: {len(dataset[0])}, {len(dataset[1])}")
        elif dataset is not None:
            print(f"{name}: {len(dataset)}")