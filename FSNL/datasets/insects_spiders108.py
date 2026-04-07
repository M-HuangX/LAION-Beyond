import os
import pickle
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing


@DATASET_REGISTRY.register()
class Insects_Spiders108(DatasetBase):

    dataset_dir = "Insects_Spiders108"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        self.split_path = os.path.join(
            self.dataset_dir, "split_x_Insects_Spiders108.json")
        
        self.split_fewshot_dir = os.path.join(
            self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
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

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for index, item in items.items():
                impath = os.path.join(path_prefix, item["classname"], item["image_name"])
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
