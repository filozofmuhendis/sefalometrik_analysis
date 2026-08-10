"""Evaluation Metrics Module for Cephalometric Radiograph Classification.

Calculates comprehensive classification metrics for imbalanced datasets:
- Accuracy, Balanced Accuracy
- Macro Precision, Macro Recall, Macro F1, Weighted F1
- Per-Class Precision, Recall, F1-score
- Confusion Matrix (Matrix, CSV, PNG plot when matplotlib is available)
- Multi-class Macro ROC-AUC (when probabilities are supplied)
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class Evaluator:
    """Academic Evaluation Metrics Calculator & Report Generator."""

    @staticmethod
    def _labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.unique(np.concatenate([y_true, y_pred]))

    @staticmethod
    def _confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
        label_to_idx = {int(label): idx for idx, label in enumerate(labels.tolist())}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
            cm[label_to_idx[int(true_label)], label_to_idx[int(pred_label)]] += 1
        return cm

    @staticmethod
    def _per_class_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        return precision, recall, f1

    @staticmethod
    def _evaluate_numpy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        labels = Evaluator._labels(y_true, y_pred)
        cm = Evaluator._confusion_matrix_numpy(y_true, y_pred, labels)
        precision_per_cls, recall_per_cls, f1_per_cls = Evaluator._per_class_from_cm(cm)
        support = cm.sum(axis=1).astype(np.float64)
        total = max(1.0, float(cm.sum()))

        acc = float(np.trace(cm) / total)
        balanced_acc = float(np.mean(recall_per_cls)) if recall_per_cls.size > 0 else 0.0
        macro_precision = float(np.mean(precision_per_cls)) if precision_per_cls.size > 0 else 0.0
        macro_recall = float(np.mean(recall_per_cls)) if recall_per_cls.size > 0 else 0.0
        f1_macro = float(np.mean(f1_per_cls)) if f1_per_cls.size > 0 else 0.0
        f1_weighted = float(np.sum(f1_per_cls * support) / max(1.0, np.sum(support)))

        results: Dict[str, Any] = {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_precision": precision_per_cls.tolist(),
            "per_class_recall": recall_per_cls.tolist(),
            "per_class_f1": f1_per_cls.tolist(),
            "confusion_matrix": cm.tolist(),
        }

        if y_prob is not None:
            try:
                if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
                    results["macro_roc_auc"] = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                    )
                    results["macro_pr_auc"] = float(
                        average_precision_score(y_true, y_prob, average="macro")
                    )
            except Exception:
                pass

        return results

    @staticmethod
    def evaluate(
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Optional[Union[List[List[float]], np.ndarray]] = None,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for true vs predicted labels.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predicted class labels.
            y_prob: Optional predicted class probability matrix of shape (N, C).
            target_names: Optional list of class label names.

        Returns:
            Dictionary containing accuracy, balanced_accuracy, macro/weighted metrics,
            per-class metrics, confusion_matrix, and optional roc_auc.
        """
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        y_pr = np.array(y_prob) if y_prob is not None else None

        if not SKLEARN_AVAILABLE:
            return Evaluator._evaluate_numpy(y_t, y_p, y_pr)

        acc = float(accuracy_score(y_t, y_p))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))
        macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
        prec_per_cls, rec_per_cls, f1_per_cls, _ = precision_recall_fscore_support(
            y_t, y_p, zero_division=0
        )
        cm = confusion_matrix(y_t, y_p)

        results: Dict[str, Any] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_precision": prec_per_cls.tolist(),
            "per_class_recall": rec_per_cls.tolist(),
            "per_class_f1": f1_per_cls.tolist(),
            "confusion_matrix": cm.tolist(),
        }

        if y_pr is not None:
            try:
                unique_classes = np.unique(y_t)
                if len(unique_classes) > 1:
                    results["macro_roc_auc"] = float(
                        roc_auc_score(y_t, y_pr, multi_class="ovr", average="macro")
                    )
                    results["macro_pr_auc"] = float(
                        average_precision_score(y_t, y_pr, average="macro")
                    )
            except Exception:
                pass

        return results

    @staticmethod
    def save_confusion_matrix(
        cm_array: Union[List[List[int]], np.ndarray],
        output_prefix: Union[str, Path],
        class_names: Optional[List[str]] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Save confusion matrix as CSV and PNG image plot (if matplotlib is available).

        Args:
            cm_array: 2D confusion matrix array.
            output_prefix: Prefix path (e.g. 'reports/confusion_matrix_val').
            class_names: List of class label strings.

        Returns:
            Tuple of (csv_filepath, png_filepath or None).
        """
        cm = np.array(cm_array)
        prefix_path = Path(output_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

        csv_path = prefix_path.parent / f"{prefix_path.name}.csv"
        png_path = prefix_path.parent / f"{prefix_path.name}.png"

        num_classes = cm.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        # 1. Save CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["True\\Pred"] + class_names)
            for i, row in enumerate(cm):
                writer.writerow([class_names[i]] + list(row))

        print(f"[Evaluator] Saved Confusion Matrix CSV: {csv_path.resolve()}")

        # 2. Render and save PNG Plot if matplotlib is installed
        saved_png_path: Optional[Path] = None
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(
                xticks=np.arange(num_classes),
                yticks=np.arange(num_classes),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel="True Class",
                xlabel="Predicted Class",
                title=f"Confusion Matrix ({prefix_path.name})",
            )

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

            fig.tight_layout()
            plt.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            saved_png_path = png_path
            print(f"[Evaluator] Saved Confusion Matrix PNG: {png_path.resolve()}")
        except ImportError:
            print("[Evaluator] matplotlib module not found. Skipping PNG confusion matrix rendering.")

        return csv_path, saved_png_path
