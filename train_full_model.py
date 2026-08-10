"""Full Model Training Script for Cephalometric X-ray Classification.

Architecture:
    Real X-ray → Image Enhancement → EfficientNetV2-S (pretrained)
    → MultiScaleAFEM → CFFM (Fuzzy) → ACAM → GRLM → Classification Head

All proposed modules are ENABLED.
This trains the complete proposed architecture from scratch.

Usage:
    python -u train_full_model.py
"""

import sys
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from modules.image_enhancement import ImageEnhancement
from models.cephalometric_net import CephalometricAnalysisNet
from loss.weighted_cross_entropy import ClassWeightedCrossEntropyLoss
from metrics.evaluator import Evaluator
from utils.seed import set_global_seed
from utils.checkpoint import CheckpointManager


# ============================================================
# DATASET CLASS
# ============================================================

class CephalometricDataset(Dataset):
    """PyTorch Dataset for Cephalometric X-ray Classification.

    Loads images from CSV manifest, applies ImageEnhancement preprocessing,
    resizes to target_size, and normalizes with ImageNet statistics.
    """

    def __init__(
        self,
        manifest_path: str,
        apply_enhancement: bool = True,
        target_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.apply_enhancement = apply_enhancement
        self.target_size = target_size
        self.records: List[Dict[str, str]] = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

        if self.apply_enhancement:
            self.enhancer = ImageEnhancement(config=Config().image_enhancement)

        print(f"[Dataset] Loaded {len(self.records)} samples from {manifest_path}", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.records[idx]
        img_path = PROJECT_ROOT / row["path"]

        if self.apply_enhancement:
            # ImageEnhancement returns uint8 grayscale
            img = self.enhancer.process(str(img_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.imread(str(img_path))

        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        label_tensor = torch.tensor(int(row["class_index"]), dtype=torch.long)

        return img_tensor, label_tensor


# ============================================================
# PARAMETER LOGGING
# ============================================================

def log_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Log and return parameter counts by module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Per-component breakdown
    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_total = sum(p.numel() for p in model.head.parameters())
    
    # Custom modules breakdown
    afem_total = sum(p.numel() for p in model.afem.parameters()) if model.afem else 0
    cffm_total = sum(p.numel() for p in model.cffm.parameters()) if model.cffm else 0
    acam_total = sum(p.numel() for p in model.acam.parameters()) if model.acam else 0
    grlm_total = sum(p.numel() for p in model.grlm.parameters()) if model.grlm else 0

    print("=" * 65, flush=True)
    print("FULL MODEL PARAMETER SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"  Total Parameters     : {total:>12,}", flush=True)
    print(f"  Trainable Parameters : {trainable:>12,}", flush=True)
    print(f"  Frozen Parameters    : {frozen:>12,}", flush=True)
    print(f"  Backbone Total       : {backbone_total:>12,}", flush=True)
    print(f"  Backbone Trainable   : {backbone_trainable:>12,}", flush=True)
    print(f"  AFEM Parameters      : {afem_total:>12,}", flush=True)
    print(f"  CFFM Parameters      : {cffm_total:>12,}", flush=True)
    print(f"  ACAM Parameters      : {acam_total:>12,}", flush=True)
    print(f"  GRLM Parameters      : {grlm_total:>12,}", flush=True)
    print(f"  Classification Head  : {head_total:>12,}", flush=True)
    print("=" * 65, flush=True)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "backbone_parameters": backbone_total,
        "afem_parameters": afem_total,
        "cffm_parameters": cffm_total,
        "acam_parameters": acam_total,
        "grlm_parameters": grlm_total,
        "head_parameters": head_total,
    }


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float]:
    """Train model for one epoch. Returns (loss, accuracy, avg_grad_norm)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    steps = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=max_grad_norm,
        )
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_grad_norm += float(norm.item()) if hasattr(norm, "item") else float(norm)
        steps += 1

    return (
        running_loss / max(1, total),
        correct / max(1, total),
        total_grad_norm / max(1, steps),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], List[int], List[int], List[List[float]]]:
    """Evaluate model on a DataLoader. Returns (loss, metrics, y_true, y_pred, y_prob)."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_y_prob: List[List[float]] = []

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total += targets.size(0)
        all_y_true.extend(targets.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())
        all_y_prob.extend(probs.cpu().numpy().tolist())

    val_loss = running_loss / max(1, total)
    metrics = Evaluator.evaluate(all_y_true, all_y_pred, y_prob=all_y_prob)

    return val_loss, metrics, all_y_true, all_y_pred, all_y_prob


def update_progressive_phase(model: CephalometricAnalysisNet, epoch: int, config: Config) -> None:
    """Update backbone progressive fine-tuning phase based on epoch."""
    p1 = config.training.phase1_epochs
    p2 = config.training.phase2_epochs

    if epoch < p1:
        target_phase = 1
    elif epoch < p2:
        target_phase = 2
    else:
        target_phase = 3

    if model.backbone.config.progressive_phase != target_phase:
        print(f"\n[Progressive Fine-Tuning] Transitioning to Phase {target_phase} at Epoch {epoch + 1}", flush=True)
        model.backbone.set_progressive_phase(target_phase)

        # Count parameters after phase change
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)", flush=True)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    start_time = time.time()

    # ----------------------------------------------------------
    # 1. CONFIGURATION
    # ----------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Full Model Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--phase1", type=int, default=None)
    parser.add_argument("--phase2", type=int, default=None)
    args, _ = parser.parse_known_args()

    config = Config()

    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.patience is not None:
        config.training.early_stopping_patience = args.patience
    if args.phase1 is not None:
        config.training.phase1_epochs = args.phase1
    if args.phase2 is not None:
        config.training.phase2_epochs = args.phase2

    # FULL MODEL: Enable all proposed modules
    config.ablation.use_afem = True
    config.ablation.use_cffm = True
    config.ablation.use_acam = True
    config.ablation.use_grlm = True
    config.backbone.pretrained = True

    # Create output directories
    reports_dir = PROJECT_ROOT / "reports" / "full_model"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "full_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 2. REPRODUCIBILITY
    # ----------------------------------------------------------
    set_global_seed(config.ablation.global_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Training on: {device}", flush=True)

    # ----------------------------------------------------------
    # 3. LOAD CLASS WEIGHTS
    # ----------------------------------------------------------
    class_weights_path = PROJECT_ROOT / "configs" / "generated_class_weights.json"
    with open(class_weights_path, "r") as f:
        cw_data = json.load(f)
    class_weights = torch.tensor(cw_data["weights"], dtype=torch.float32)
    print(f"[Class Weights] Loaded from train split: {class_weights.tolist()}", flush=True)

    # ----------------------------------------------------------
    # 4. LOAD CLASS MAPPING
    # ----------------------------------------------------------
    class_mapping_path = PROJECT_ROOT / "reports" / "class_mapping.json"
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    print(f"[Class Mapping] {class_mapping}", flush=True)

    # ----------------------------------------------------------
    # 5. CREATE DATALOADERS
    # ----------------------------------------------------------
    train_manifest = str(PROJECT_ROOT / config.dataset.train_manifest)
    val_manifest = str(PROJECT_ROOT / config.dataset.val_manifest)
    test_manifest = str(PROJECT_ROOT / config.dataset.test_manifest)

    train_dataset = CephalometricDataset(train_manifest, apply_enhancement=True)
    val_dataset = CephalometricDataset(val_manifest, apply_enhancement=True)
    test_dataset = CephalometricDataset(test_manifest, apply_enhancement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    print(f"\n[DataLoaders] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", flush=True)

    # ----------------------------------------------------------
    # 6. CREATE MODEL (FULL MODEL)
    # ----------------------------------------------------------
    print("\n" + "=" * 65, flush=True)
    print("FULL MODEL ARCHITECTURE CONFIGURATION", flush=True)
    print("=" * 65, flush=True)
    print(f"  AFEM  : ENABLED", flush=True)
    print(f"  CFFM  : ENABLED (Fuzzy Scale Weights)", flush=True)
    print(f"  ACAM  : ENABLED", flush=True)
    print(f"  GRLM  : ENABLED (512-D L2 Normalized)", flush=True)
    print(f"  Backbone: EfficientNetV2-S (ImageNet pretrained)", flush=True)
    print(f"  Head: Dropout(0.3) -> FC(512->3)", flush=True)
    print("=" * 65, flush=True)

    model = CephalometricAnalysisNet(config=config)
    model = model.to(device)

    param_info = log_model_parameters(model)

    # ----------------------------------------------------------
    # 7. LOSS, OPTIMIZER, SCHEDULER
    # ----------------------------------------------------------
    criterion = ClassWeightedCrossEntropyLoss(class_weights=class_weights)
    print(f"\n[Loss] ClassWeightedCrossEntropyLoss with weights: {class_weights.tolist()}", flush=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    print(f"[Optimizer] AdamW (lr={config.training.learning_rate}, wd={config.training.weight_decay})", flush=True)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=config.training.eta_min,
    )
    print(f"[Scheduler] CosineAnnealingLR (T_max={config.training.num_epochs}, eta_min={config.training.eta_min})", flush=True)

    # Early stopping on Macro F1 (higher is better)
    best_val_f1 = -1.0
    patience = config.training.early_stopping_patience
    patience_counter = 0
    best_epoch = -1

    checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

    # ----------------------------------------------------------
    # 8. SAVE CONFIGURATION
    # ----------------------------------------------------------
    config_snapshot = {
        "experiment": "full_model",
        "description": "EfficientNetV2-S + Enhanced Image + AFEM + CFFM + ACAM + GRLM",
        "ablation": {
            "use_afem": True,
            "use_cffm": True,
            "use_acam": True,
            "use_grlm": True,
        },
        "backbone": {"model_name": config.backbone.model_name, "pretrained": True},
        "training": {
            "batch_size": config.training.batch_size,
            "num_epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "eta_min": config.training.eta_min,
            "max_grad_norm": config.training.max_grad_norm,
            "early_stopping_patience": patience,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "mixed_precision": False,
            "progressive_phases": {
                "phase1_epochs": config.training.phase1_epochs,
                "phase2_epochs": config.training.phase2_epochs,
            },
        },
        "dataset": {
            "train_manifest": config.dataset.train_manifest,
            "val_manifest": config.dataset.val_manifest,
            "test_manifest": config.dataset.test_manifest,
            "target_size": list(config.dataset.target_size),
            "imagenet_mean": list(config.dataset.imagenet_mean),
            "imagenet_std": list(config.dataset.imagenet_std),
        },
        "class_weights": class_weights.tolist(),
        "class_mapping": class_mapping,
        "seed": config.ablation.global_seed,
        "parameters": param_info,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }
    with open(reports_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=4)

    # ----------------------------------------------------------
    # 9. TRAINING LOOP
    # ----------------------------------------------------------
    print("\n" + "=" * 65, flush=True)
    print(f"STARTING FULL MODEL TRAINING ({config.training.num_epochs} epochs)", flush=True)
    print(f"Primary metric: Macro F1 (model selection)", flush=True)
    print(f"Early stopping patience: {patience} epochs", flush=True)
    print("=" * 65, flush=True)

    history_keys = [
        "epoch", "train_loss", "val_loss", "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "balanced_accuracy", "learning_rate",
    ]

    # CSV header
    with open(reports_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(history_keys)

    history_data: Dict[str, List[Any]] = {k: [] for k in history_keys}
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(config.training.num_epochs):
        epoch_start = time.time()

        # Progressive fine-tuning phase update
        update_progressive_phase(model, epoch, config)

        current_trainable = [p for p in model.parameters() if p.requires_grad]
        if len(list(optimizer.param_groups[0]["params"])) != len(current_trainable):
            optimizer = optim.AdamW(
                current_trainable,
                lr=optimizer.param_groups[0]["lr"] if epoch > 0 else config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs - epoch,
                eta_min=config.training.eta_min,
            )

        # Train
        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=config.training.max_grad_norm,
        )

        # Validate
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        epoch_num = epoch + 1

        # Log metrics
        row_data = {
            "epoch": epoch_num,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "accuracy": round(val_metrics["accuracy"], 6),
            "macro_precision": round(val_metrics["macro_precision"], 6),
            "macro_recall": round(val_metrics["macro_recall"], 6),
            "macro_f1": round(val_metrics["f1_macro"], 6),
            "weighted_f1": round(val_metrics["f1_weighted"], 6),
            "balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
            "learning_rate": round(current_lr, 8),
        }

        for k in history_keys:
            history_data[k].append(row_data[k])

        # Append to CSV
        with open(reports_dir / "training_history.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row_data[k] for k in history_keys])

        # Print epoch summary
        print(
            f"Epoch [{epoch_num:02d}/{config.training.num_epochs:02d}] "
            f"({epoch_time:.0f}s) "
            f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc*100:.1f}% | "
            f"VlLoss: {val_loss:.4f} | VlAcc: {val_metrics['accuracy']*100:.1f}% | "
            f"MacroF1: {val_metrics['f1_macro']:.4f} | BalAcc: {val_metrics['balanced_accuracy']*100:.1f}% | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        # Best model selection based on Macro F1
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch_num
            best_val_metrics = val_metrics.copy()
            best_val_metrics["val_loss"] = val_loss
            patience_counter = 0

            # Save best checkpoint
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                filename="best_checkpoint.pth",
                class_weights=class_weights,
                config=config_snapshot,
                seed=config.ablation.global_seed,
                dataset_split_manifest=str(config.dataset.train_manifest),
                class_mapping=class_mapping,
            )

            # Save val confusion matrix
            Evaluator.save_confusion_matrix(
                val_metrics["confusion_matrix"],
                output_prefix=str(reports_dir / "confusion_matrix_val"),
                class_names=class_names,
            )

            print(f"  -> NEW BEST Macro F1: {best_val_f1:.4f} (Epoch {best_epoch}) - checkpoint saved", flush=True)
        else:
            patience_counter += 1
            print(f"  (patience: {patience_counter}/{patience})", flush=True)

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.", flush=True)
            break

    training_time = time.time() - start_time

    print("\n" + "=" * 65, flush=True)
    print(f"TRAINING COMPLETED! Best Epoch: {best_epoch} | Best Val Macro F1: {best_val_f1:.4f}", flush=True)
    print(f"Total training time: {training_time/60:.1f} minutes", flush=True)
    print("=" * 65, flush=True)

    # ----------------------------------------------------------
    # 10. FINAL TEST EVALUATION
    # ----------------------------------------------------------
    print("\n" + "=" * 65, flush=True)
    print("FINAL TEST EVALUATION (Best Checkpoint)", flush=True)
    print("=" * 65, flush=True)

    # Load best checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint(
        model=model,
        filename="best_checkpoint.pth",
        device=device,
    )
    model = model.to(device)
    print(f"[Checkpoint] Loaded best model from epoch {checkpoint_data['epoch'] + 1}", flush=True)

    # Evaluate on test set
    test_loss, test_metrics, test_y_true, test_y_pred, test_y_prob = evaluate(
        model, test_loader, criterion, device,
    )

    # Print test results
    print(f"\nTest Loss          : {test_loss:.4f}", flush=True)
    print(f"Test Accuracy      : {test_metrics['accuracy']*100:.2f}%", flush=True)
    print(f"Balanced Accuracy  : {test_metrics['balanced_accuracy']*100:.2f}%", flush=True)
    print(f"Macro Precision    : {test_metrics['macro_precision']:.4f}", flush=True)
    print(f"Macro Recall       : {test_metrics['macro_recall']:.4f}", flush=True)
    print(f"Macro F1           : {test_metrics['f1_macro']:.4f}", flush=True)
    print(f"Weighted F1        : {test_metrics['f1_weighted']:.4f}", flush=True)

    if "macro_roc_auc" in test_metrics:
        print(f"Macro ROC-AUC      : {test_metrics['macro_roc_auc']:.4f}", flush=True)

    # Per-class results
    print("\nPer-Class Results:", flush=True)
    for i, name in enumerate(class_names):
        p = test_metrics["per_class_precision"][i] if i < len(test_metrics["per_class_precision"]) else 0
        r = test_metrics["per_class_recall"][i] if i < len(test_metrics["per_class_recall"]) else 0
        f = test_metrics["per_class_f1"][i] if i < len(test_metrics["per_class_f1"]) else 0
        print(f"  {name}: Precision={p:.4f} | Recall={r:.4f} | F1={f:.4f}", flush=True)

    # Save test confusion matrix
    Evaluator.save_confusion_matrix(
        test_metrics["confusion_matrix"],
        output_prefix=str(reports_dir / "confusion_matrix_test"),
        class_names=class_names,
    )

    # ----------------------------------------------------------
    # 11. SAVE ALL REPORTS
    # ----------------------------------------------------------

    # Classification report CSV
    with open(reports_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1"])
        for i, name in enumerate(class_names):
            p = test_metrics["per_class_precision"][i] if i < len(test_metrics["per_class_precision"]) else 0
            r = test_metrics["per_class_recall"][i] if i < len(test_metrics["per_class_recall"]) else 0
            f1 = test_metrics["per_class_f1"][i] if i < len(test_metrics["per_class_f1"]) else 0
            writer.writerow([name, round(p, 6), round(r, 6), round(f1, 6)])
        writer.writerow(["macro_avg",
                         round(test_metrics["macro_precision"], 6),
                         round(test_metrics["macro_recall"], 6),
                         round(test_metrics["f1_macro"], 6)])

    # Metrics JSON
    metrics_json = {
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_time, 2),
        "total_epochs_run": len(history_data["epoch"]),
        "validation": {
            "loss": round(best_val_metrics["val_loss"], 6) if best_val_metrics else None,
            "accuracy": round(best_val_metrics["accuracy"], 6) if best_val_metrics else None,
            "balanced_accuracy": round(best_val_metrics["balanced_accuracy"], 6) if best_val_metrics else None,
            "macro_precision": round(best_val_metrics["macro_precision"], 6) if best_val_metrics else None,
            "macro_recall": round(best_val_metrics["macro_recall"], 6) if best_val_metrics else None,
            "macro_f1": round(best_val_metrics["f1_macro"], 6) if best_val_metrics else None,
            "weighted_f1": round(best_val_metrics["f1_weighted"], 6) if best_val_metrics else None,
            "macro_roc_auc": round(best_val_metrics.get("macro_roc_auc", 0), 6) if best_val_metrics else None,
        },
        "test": {
            "loss": round(test_loss, 6),
            "accuracy": round(test_metrics["accuracy"], 6),
            "balanced_accuracy": round(test_metrics["balanced_accuracy"], 6),
            "macro_precision": round(test_metrics["macro_precision"], 6),
            "macro_recall": round(test_metrics["macro_recall"], 6),
            "macro_f1": round(test_metrics["f1_macro"], 6),
            "weighted_f1": round(test_metrics["f1_weighted"], 6),
            "macro_roc_auc": round(test_metrics.get("macro_roc_auc", 0), 6),
            "per_class_precision": [round(v, 6) for v in test_metrics["per_class_precision"]],
            "per_class_recall": [round(v, 6) for v in test_metrics["per_class_recall"]],
            "per_class_f1": [round(v, 6) for v in test_metrics["per_class_f1"]],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "parameters": param_info,
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    # ----------------------------------------------------------
    # 12. GENERATE FULL MODEL REPORT
    # ----------------------------------------------------------
    val_roc = round(best_val_metrics.get("macro_roc_auc", 0), 4) if best_val_metrics else "N/A"
    test_roc = round(test_metrics.get("macro_roc_auc", 0), 4)

    # Build per-class table
    per_class_rows = ""
    for i, name in enumerate(class_names):
        p = test_metrics["per_class_precision"][i] if i < len(test_metrics["per_class_precision"]) else 0
        r = test_metrics["per_class_recall"][i] if i < len(test_metrics["per_class_recall"]) else 0
        f1 = test_metrics["per_class_f1"][i] if i < len(test_metrics["per_class_f1"]) else 0
        per_class_rows += f"| {name} | {p:.4f} | {r:.4f} | {f1:.4f} |\n"

    # Confusion matrix table
    cm = test_metrics["confusion_matrix"]
    cm_header = "| True\\Pred | " + " | ".join(class_names) + " |"
    cm_sep = "|" + "|".join(["---"] * (len(class_names) + 1)) + "|"
    cm_rows = ""
    for i, name in enumerate(class_names):
        row_vals = " | ".join(str(cm[i][j]) for j in range(len(class_names)))
        cm_rows += f"| {name} | {row_vals} |\n"

    # Train class distribution
    from collections import Counter
    train_labels = [int(r["class_index"]) for r in train_dataset.records]
    val_labels = [int(r["class_index"]) for r in val_dataset.records]
    test_labels = [int(r["class_index"]) for r in test_dataset.records]
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    report = f"""# Full Proposed Model Training Report

## 1. Dataset
- **Dataset Root**: `dataset/`
- **Total Valid Samples**: {len(train_dataset) + len(val_dataset) + len(test_dataset)}
- **Preprocessing**: ImageEnhancement (16-step) + ImageNet normalization
- **Input Size**: 512 × 512 × 3

## 2. Split
| Split | Samples |
|-------|---------|
| Train | {len(train_dataset)} |
| Validation | {len(val_dataset)} |
| Test | {len(test_dataset)} |

## 3. Class Distribution
| Class | Train | Val | Test |
|-------|-------|-----|------|
| sinif1 (0) | {train_counts[0]} | {val_counts[0]} | {test_counts[0]} |
| sinif2 (1) | {train_counts[1]} | {val_counts[1]} | {test_counts[1]} |
| sinif3 (2) | {train_counts[2]} | {val_counts[2]} | {test_counts[2]} |

## 4. Class Weights
- Formula: weight_i = N / (C × n_i) (computed from TRAIN split only)
- sinif1: {class_weights[0]:.4f}
- sinif2: {class_weights[1]:.4f}
- sinif3: {class_weights[2]:.4f}

## 5. Model Architecture
```
Real X-ray
    ↓
Image Enhancement
    ↓
EfficientNetV2-S Backbone (ImageNet pretrained)
    ↓
MultiScaleAFEM (Adaptive Feature Enhancement)
    ↓
CFFM (Cross Scale Feature Fusion with Fuzzy Logic)
    ↓
ACAM (Adaptive Context Aggregation)
    ↓
GRLM (Global Representation Learning - 512-D L2 Normalized)
    ↓
Classification Head (Dropout 0.3 + Linear -> 3 classes)
```
- **AFEM**: ENABLED
- **CFFM**: ENABLED (Fuzzy Scale Weights)
- **ACAM**: ENABLED
- **GRLM**: ENABLED
- Total Parameters: {param_info['total_parameters']:,}
- Trainable Parameters: {param_info['trainable_parameters']:,}
- Frozen Parameters: {param_info['frozen_parameters']:,}

## 6. Transfer Learning Strategy
- Backbone: ImageNet pretrained EfficientNetV2-S
- Phase 1 (Epochs 1-{config.training.phase1_epochs}): Freeze Stem + Stage 1 + Stage 2
- Phase 2 (Epochs {config.training.phase1_epochs+1}-{config.training.phase2_epochs}): Unfreeze Stage 2
- Phase 3 (Epochs {config.training.phase2_epochs+1}+): Full fine-tuning
- Proposed attention/fusion modules remain trainable across all phases.

## 7. Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | {config.training.learning_rate} |
| Weight Decay | {config.training.weight_decay} |
| Scheduler | CosineAnnealingLR |
| Eta Min | {config.training.eta_min} |
| Batch Size | {config.training.batch_size} |
| Max Epochs | {config.training.num_epochs} |
| Gradient Clipping | {config.training.max_grad_norm} |
| Early Stopping | Patience={patience}, on Macro F1 |
| Seed | {config.ablation.global_seed} |
| Device | {device} |

## 8. Validation Results (Best Epoch)
| Metric | Value |
|--------|-------|
| Loss | {best_val_metrics['val_loss']:.4f} |
| Accuracy | {best_val_metrics['accuracy']*100:.2f}% |
| Balanced Accuracy | {best_val_metrics['balanced_accuracy']*100:.2f}% |
| Macro Precision | {best_val_metrics['macro_precision']:.4f} |
| Macro Recall | {best_val_metrics['macro_recall']:.4f} |
| Macro F1 | {best_val_metrics['f1_macro']:.4f} |
| Weighted F1 | {best_val_metrics['f1_weighted']:.4f} |
| Macro ROC-AUC | {val_roc} |

## 9. Best Epoch
- **Best Epoch**: {best_epoch}
- **Selection Metric**: Macro F1
- **Total Epochs Run**: {len(history_data['epoch'])}
- **Training Time**: {training_time/60:.1f} minutes

## 10. Final Test Results

| Metric | Validation | Test |
|--------|----------:|-----:|
| Accuracy | {best_val_metrics['accuracy']*100:.2f}% | {test_metrics['accuracy']*100:.2f}% |
| Balanced Accuracy | {best_val_metrics['balanced_accuracy']*100:.2f}% | {test_metrics['balanced_accuracy']*100:.2f}% |
| Macro Precision | {best_val_metrics['macro_precision']:.4f} | {test_metrics['macro_precision']:.4f} |
| Macro Recall | {best_val_metrics['macro_recall']:.4f} | {test_metrics['macro_recall']:.4f} |
| Macro F1 | {best_val_metrics['f1_macro']:.4f} | {test_metrics['f1_macro']:.4f} |
| Weighted F1 | {best_val_metrics['f1_weighted']:.4f} | {test_metrics['f1_weighted']:.4f} |
| Macro ROC-AUC | {val_roc} | {test_roc} |

## 11. Per-Class Results (Test)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
{per_class_rows}
## 12. Confusion Matrix (Test)
{cm_header}
{cm_sep}
{cm_rows}
## 13. Reproducibility Information
- Seed: {config.ablation.global_seed}
- Framework: PyTorch {torch.__version__}
- Device: {device}
- Checkpoint: `checkpoints/full_model/best_checkpoint.pth`
- Training History: `reports/full_model/training_history.csv`
- Manifests: train.csv / val.csv / test.csv
"""

    with open(reports_dir / "full_model_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # ----------------------------------------------------------
    # 13. FINAL SUMMARY
    # ----------------------------------------------------------
    print("\n" + "=" * 65, flush=True)
    print("FULL MODEL FINAL RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"\n| {'Metric':<20} | {'Validation':>12} | {'Test':>12} |", flush=True)
    print(f"|{'-'*22}|{'-'*14}|{'-'*14}|", flush=True)
    print(f"| {'Accuracy':<20} | {best_val_metrics['accuracy']*100:>11.2f}% | {test_metrics['accuracy']*100:>11.2f}% |", flush=True)
    print(f"| {'Balanced Accuracy':<20} | {best_val_metrics['balanced_accuracy']*100:>11.2f}% | {test_metrics['balanced_accuracy']*100:>11.2f}% |", flush=True)
    print(f"| {'Macro Precision':<20} | {best_val_metrics['macro_precision']:>12.4f} | {test_metrics['macro_precision']:>12.4f} |", flush=True)
    print(f"| {'Macro Recall':<20} | {best_val_metrics['macro_recall']:>12.4f} | {test_metrics['macro_recall']:>12.4f} |", flush=True)
    print(f"| {'Macro F1':<20} | {best_val_metrics['f1_macro']:>12.4f} | {test_metrics['f1_macro']:>12.4f} |", flush=True)
    print(f"| {'Weighted F1':<20} | {best_val_metrics['f1_weighted']:>12.4f} | {test_metrics['f1_weighted']:>12.4f} |", flush=True)
    print(f"| {'Macro ROC-AUC':<20} | {val_roc:>12} | {test_roc:>12} |", flush=True)
    print(f"\nBest Epoch: {best_epoch}", flush=True)
    print(f"Training Time: {training_time/60:.1f} minutes", flush=True)
    print(f"Parameters: {param_info['total_parameters']:,} total, {param_info['trainable_parameters']:,} trainable", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
