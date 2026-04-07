import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import os.path as osp
from .build import EVALUATOR_REGISTRY

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results




@EVALUATOR_REGISTRY.register()
class ClusteringAccuracy(EvaluatorBase):
    """Evaluator for clustering accuracy using Hungarian algorithm."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._total = 0
        self._per_class_res = None
        self._logits = []
        self._true_labels = []
        self._pred_labels = []
        self._class_names = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._total = 0
        self._logits = []
        self._true_labels = []
        self._pred_labels = []
        self._class_names = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, impath):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): pseudo labels [batch]
        # impath (list): list of image paths
        self._logits.append(mo.cpu())
        self._total += mo.shape[0]

        # Extract true labels from image paths
        true_labels = [osp.basename(osp.dirname(path)) for path in impath]
        self._true_labels.extend(true_labels)
        self._class_names.extend(true_labels)

        # Store predicted labels
        pred = mo.max(1)[1]
        self._pred_labels.extend(pred.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()

        # Concatenate all logits
        all_logits = torch.cat(self._logits, dim=0)

        # Get unique class names and create a mapping
        unique_classes = list(set(self._class_names))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        # Convert true labels to indices
        true_labels = np.array([class_to_idx[cls] for cls in self._true_labels])

        # Get the number of predicted classes
        num_pred_classes = all_logits.shape[1]
        num_true_classes = len(unique_classes)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_labels, self._pred_labels, 
                                       labels=range(max(num_pred_classes, num_true_classes)))

        # Ensure the confusion matrix is square by padding if necessary
        if num_pred_classes < num_true_classes:
            pad_width = ((0, 0), (0, num_true_classes - num_pred_classes))
            conf_matrix = np.pad(conf_matrix, pad_width, mode='constant', constant_values=0)
        elif num_pred_classes > num_true_classes:
            pad_width = ((0, num_pred_classes - num_true_classes), (0, 0))
            conf_matrix = np.pad(conf_matrix, pad_width, mode='constant', constant_values=0)

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        # Compute accuracy
        total_correct = conf_matrix[row_ind, col_ind].sum()
        accuracy = 100.0 * total_correct / self._total

        results["clustering_accuracy"] = accuracy

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {total_correct:,}\n"
            f"* clustering accuracy: {accuracy:.1f}%\n"
            f"* true classes: {num_true_classes}\n"
            f"* predicted classes: {num_pred_classes}"
        )

        # Compute per-class accuracy
        if self._per_class_res is not None:
            print("=> per-class result")
            per_class_acc = []

            for i, (true_idx, pred_idx) in enumerate(zip(row_ind, col_ind)):
                if true_idx < num_true_classes:  # Only consider actual classes
                    true_class = unique_classes[true_idx]
                    correct = conf_matrix[true_idx, pred_idx]
                    total = conf_matrix[true_idx].sum()
                    acc = 100.0 * correct / total if total > 0 else 0
                    per_class_acc.append(acc)
                    print(
                        f"* class: {true_class}\t"
                        f"total: {total:,}\t"
                        f"correct: {correct:,}\t"
                        f"acc: {acc:.1f}%"
                    )

            mean_acc = np.mean(per_class_acc)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        # Save confusion matrix if required
        if self.cfg.TEST.COMPUTE_CMAT:
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(conf_matrix, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results