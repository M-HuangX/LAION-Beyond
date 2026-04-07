import numpy as np
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx, image_path = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx, image_path = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not]), image_path
    def reset_uq_idx(self):
        # # 重置 labelled_dataset 的 uq_idx
        # for i, (data, target, _, image_path) in enumerate(self.labelled_dataset):
        self.labelled_dataset.uq_idxs = list(range(len(self.labelled_dataset.uq_idxs)))

        # 重置 unlabelled_dataset 的 uq_idx，从 labelled_dataset 的长度开始
        start_idx = len(self.labelled_dataset)
     
        # for i, (data, target, _, image_path) in enumerate(self.unlabelled_dataset.data):
            # self.unlabelled_dataset.uq_idxs[i] = start_idx + i
        self.unlabelled_dataset.uq_idxs = list(range(start_idx, start_idx + len(self.unlabelled_dataset.uq_idxs)))
    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
