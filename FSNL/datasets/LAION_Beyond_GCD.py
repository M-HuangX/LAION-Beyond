import os
import json
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class LAION_Beyond_GCD(DatasetBase):

    def __init__(self, cfg):
        self.subfield = cfg.DATASET.SUBFIELD
        self.dataset_dir = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.json_path = os.path.join(self.dataset_dir, f'{self.subfield}_clustering_results.json')

        assert cfg.DATASET.SUB_CLASSES == "OOP", "Only OOP mode is supported for this dataset."

        train, test = self.process_json()

        super().__init__(train_x=train, val=test, test=test)

    def process_json(self):
        print(f"Reading data from {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Group images by classname
        class_images = defaultdict(list)
        for item in data:
            class_images[item['pseudo_classname']].append(item['image_path'])

        # Sort classnames and create label mapping
        classnames = sorted(class_images.keys())
        class_to_label = {classname: idx for idx, classname in enumerate(classnames)}

        train_data = []
        test_data = []

        for classname, images in class_images.items():
            label = class_to_label[classname]
            
            # Shuffle images to ensure random selection
            random.shuffle(images)
            
            # Select 16 images for training, rest for testing
            train_images = images[:16]
            test_images = images[16:]

            for impath in train_images:
                train_data.append(Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                ))

            for impath in test_images:
                test_data.append(Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                ))

        return train_data, test_data

    @staticmethod
    def get_classnames(data_source):
        return sorted(list(set([d.classname for d in data_source])))