import os
import pickle
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

"""
Dataset with OOP (Out-of-Pretraining) and IP (In-Pretraining) subsets.

``cfg.DATASET.SUB_CLASSES`` selects which subset to use: ``"OOP"``, ``"IP"``,
or ``"All"`` (merged). Training and evaluation default to OOP; IP is used for
cross-split evaluation only.

Key config fields:
  - ``cfg.DATASET.NUM_SHOTS``: number of training shots per class (int >= 1).
  - ``cfg.DATASET.USE_SUPERCLASS_F_ZS_TEST``: use superclass labels for
    zero-shot evaluation (rarely needed; can be ignored for standard runs).
"""

@DATASET_REGISTRY.register()
class Landmark59_30(DatasetBase):

    # OOP_dataset_dir = "Landmark59_OOP"
    # OOP_split_json = "split_Xin_Landmark59_OOP.json"
    # IP_dataset_dir = "Landmark30_IP"
    # IP_split_json = "split_Xin_Landmark30_IP.json"
    def find_dataset_info(self, cfg, category):
        import glob
        # Find OOP dataset directory
        oov_dir_pattern = os.path.join(cfg.DATASET.ROOT, f"{category}*_OOP")
        oov_dirs = glob.glob(oov_dir_pattern)
        if not oov_dirs:
            raise ValueError(f"No OOP directory found for {category}")
        OOP_dataset_dir = os.path.basename(oov_dirs[0])
    
        # Find OOP split JSON file
        oov_json_pattern = os.path.join(cfg.DATASET.ROOT, OOP_dataset_dir, "split_Xin*.json")
        oov_jsons = glob.glob(oov_json_pattern)
        if not oov_jsons:
            raise ValueError(f"No OOP split json found for {category}")
        OOP_split_json = os.path.basename(oov_jsons[0])
    
        # Find IP dataset directory
        iv_dir_pattern = os.path.join(cfg.DATASET.ROOT, f"{category}*_IP")
        iv_dirs = glob.glob(iv_dir_pattern)
        if not iv_dirs:
            raise ValueError(f"No IP directory found for {category}")
        IP_dataset_dir = os.path.basename(iv_dirs[0])
    
        # Find IP split JSON file
        iv_json_pattern = os.path.join(cfg.DATASET.ROOT, IP_dataset_dir, "split_Xin*.json")
        iv_jsons = glob.glob(iv_json_pattern)
        if not iv_jsons:
            raise ValueError(f"No IP split json found for {category}")
        IP_split_json = os.path.basename(iv_jsons[0])
    
        return OOP_dataset_dir, OOP_split_json, IP_dataset_dir, IP_split_json
    
    


    def __init__(self, cfg):
        # Auto-discover dataset directories and split files
        category = "Landmark"
        self.OOP_dataset_dir, self.OOP_split_json, self.IP_dataset_dir, self.IP_split_json = self.find_dataset_info(cfg, category)
        # Auto-discover dataset directories and split files
        sub = cfg.DATASET.SUB_CLASSES
        assert sub in ["All", "OOP", "IP"]
        if sub == "OOP":
            train, val, test = self.sub_init(cfg, self.OOP_dataset_dir, self.OOP_split_json)
        elif sub == "IP":
            train, val, test = self.sub_init(cfg, self.IP_dataset_dir, self.IP_split_json)
        elif sub == "All":
            OOP_train, OOP_val, OOP_test = self.sub_init(cfg, self.OOP_dataset_dir, self.OOP_split_json)
            IP_train, IP_val, IP_test = self.sub_init(cfg, self.IP_dataset_dir, self.IP_split_json)
            # TO DO
            train, val, test = self.OOP_IP_merged(OOP_train, OOP_val, OOP_test, IP_train, IP_val, IP_test)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def sub_init(self, cfg, sub_dataset_dir, split_json):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, sub_dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        self.split_path = os.path.join(
            self.dataset_dir, split_json)
        
        self.split_fewshot_dir = os.path.join(
            self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        use_superclass_for_zs_test = cfg.DATASET.USE_SUPERCLASS_F_ZS_TEST

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir, use_superclass_for_zs_test)
        else:
            print("Can not find split file.")

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(
                    f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(
                    train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(
                    val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return train, val, test
    
    
    # TO DO
    def OOP_IP_merged(self, OOP_train, OOP_val, OOP_test, IP_train, IP_val, IP_test):
        # Step 1: Create a global classname to label mapping
        classnames = sorted(set([datum.classname for datum in OOP_train + IP_train]))
        class_to_label = {classname: idx for idx, classname in enumerate(classnames)}

        # Step 2: Function to update dataset
        def update_dataset(dataset):
            updated_dataset = []
            for datum in dataset:
                new_label = class_to_label[datum.classname]
                updated_datum = Datum(
                    impath=datum.impath,
                    label=new_label,
                    classname=datum.classname,
                    caption=datum.caption
                )
                updated_dataset.append(updated_datum)
            return updated_dataset

        # Step 3: Update and merge datasets
        train = update_dataset(OOP_train) + update_dataset(IP_train)
        val = update_dataset(OOP_val) + update_dataset(IP_val)
        test = update_dataset(OOP_test) + update_dataset(IP_test)

        return train, val, test



    def read_split(self, filepath, path_prefix, use_superclass_for_zs_test):
        def _convert(items):
            out = []
            for index, item in items.items():
                impath = os.path.join(path_prefix, item["classname"], item["image_name"])
                if use_superclass_for_zs_test:
                    datum = Datum(impath=impath, label=int(item["s_label"]), classname=item["superclass"])
                else:
                    datum = Datum(impath=impath, label=int(item["label"]), classname=item["classname"], caption=item.get("INFO", {}).get("Refined_caption3", ""))
                out.append(datum)
            return out

        print(f"Reading split from {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            split = json.load(f)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test