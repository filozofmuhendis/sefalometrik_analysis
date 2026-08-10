"""Complete Academic Training Engine with Progressive Fine-Tuning, AMP, and Checkpointing.

Implements full training loop:
1. Progressive Fine-Tuning across Phase 1, Phase 2, Phase 3.
2. AdamW Optimizer & CosineAnnealingLR Learning Rate Scheduler.
3. Class Weighted Cross Entropy Loss.
4. Mixed Precision Training (torch.amp.autocast & GradScaler).
5. Gradient Norm Clipping (clip_grad_norm_).
6. EarlyStopping callback & CheckpointManager.
7. TensorBoard event logging & CSV training history export (reports/training_history.csv).
8. Academic evaluation metrics (Accuracy, Macro/Weighted F1, Balanced Acc, Confusion Matrix).
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from configs.config import Config
from loss.weighted_cross_entropy import ClassWeightedCrossEntropyLoss
from metrics.evaluator import Evaluator
from models.cephalometric_net import CephalometricAnalysisNet
from utils.checkpoint import CheckpointManager
from utils.early_stopping import EarlyStopping


class Trainer:
    """Academic Training Manager for CephalometricAnalysisNet."""

    def __init__(
        self,
        model: CephalometricAnalysisNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config = Config(),
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        reports_dir: Union[str, Path] = "reports",
    ) -> None:
        """Initialize Trainer with model, datasets, optimizer, and callbacks."""
        self.config: Config = config
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.reports_dir: Path = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model: CephalometricAnalysisNet = model.to(self.device)

        # Loss function
        self.criterion = ClassWeightedCrossEntropyLoss(class_weights=class_weights)
        if class_weights is not None:
            print(f"[Trainer] Initialized ClassWeightedCrossEntropyLoss with class weights: {class_weights.tolist()}")
        else:
            print("[Trainer] Initialized Standard CrossEntropyLoss (Uniform class weights)")

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Scheduler: CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
            eta_min=self.config.training.eta_min,
        )

        # Mixed Precision AMP GradScaler
        self.use_amp: bool = self.config.training.use_mixed_precision and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # EarlyStopping & Checkpoint Manager
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.early_stopping_delta,
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.config.training.checkpoint_dir)

        # TensorBoard logger
        if SummaryWriter is not None:
            self.writer: Optional[Any] = SummaryWriter(log_dir=self.config.training.tensorboard_dir)
        else:
            self.writer = None

        self.csv_history_path = self.reports_dir / "training_history.csv"

    def _update_progressive_phase(self, epoch: int) -> None:
        """Update progressive fine-tuning phase based on epoch thresholds."""
        p1 = self.config.training.phase1_epochs
        p2 = self.config.training.phase2_epochs

        if epoch < p1:
            target_phase = 1
        elif epoch < p2:
            target_phase = 2
        else:
            target_phase = 3

        if self.model.backbone.config.progressive_phase != target_phase:
            print(f"\n[Progressive Fine-Tuning] Transitioning to Phase {target_phase} at Epoch {epoch + 1}")
            self.model.backbone.set_progressive_phase(target_phase)
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.optimizer.param_groups[0]["lr"],
                weight_decay=self.config.training.weight_decay,
            )

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Execute training phase for one epoch.

        Returns:
            Tuple of (train_loss, train_accuracy, avg_grad_norm).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_grad_norm = 0.0
        steps = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)
            else:
                autocast_ctx = torch.no_grad() if not torch.is_grad_enabled() else torch.enable_grad()

            with autocast_ctx:
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=self.config.training.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=self.config.training.max_grad_norm,
                )
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            total_grad_norm += float(norm.item()) if hasattr(norm, "item") else float(norm)
            steps += 1

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        avg_grad_norm = total_grad_norm / max(1, steps)
        return train_loss, train_acc, avg_grad_norm

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, Any], List[int], List[int], List[List[float]]]:
        """Execute validation phase over validation DataLoader.

        Returns:
            Tuple of (val_loss, metrics_dict, y_true, y_pred, y_prob).
        """
        self.model.eval()
        running_loss = 0.0
        total = 0

        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_y_prob: List[List[float]] = []

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

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

    def fit(self) -> Dict[str, Any]:
        """Execute complete training loop across all epochs."""
        print("=" * 65)
        print(f"STARTING CEPHALOMETRIC ANALYSIS TRAINING ON DEVICE: {self.device}")
        print("=" * 65)

        best_val_loss = float("inf")
        history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "balanced_acc": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "weighted_f1": [],
            "lr": [],
            "grad_norm": [],
        }

        # Initialize CSV header
        with open(self.csv_history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(history.keys()))

        for epoch in range(self.config.training.num_epochs):
            self._update_progressive_phase(epoch)

            train_loss, train_acc, grad_norm = self.train_epoch(epoch)
            val_loss, val_metrics, y_true, y_pred, y_prob = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Append metrics to history
            epoch_num = epoch + 1
            history["epoch"].append(epoch_num)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["balanced_acc"].append(val_metrics["balanced_accuracy"])
            history["macro_precision"].append(val_metrics["macro_precision"])
            history["macro_recall"].append(val_metrics["macro_recall"])
            history["macro_f1"].append(val_metrics["f1_macro"])
            history["weighted_f1"].append(val_metrics["f1_weighted"])
            history["lr"].append(current_lr)
            history["grad_norm"].append(grad_norm)

            # Append row to training_history.csv
            with open(self.csv_history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch_num,
                        round(train_loss, 4),
                        round(train_acc, 4),
                        round(val_loss, 4),
                        round(val_metrics["accuracy"], 4),
                        round(val_metrics["balanced_accuracy"], 4),
                        round(val_metrics["macro_precision"], 4),
                        round(val_metrics["macro_recall"], 4),
                        round(val_metrics["f1_macro"], 4),
                        round(val_metrics["f1_weighted"], 4),
                        round(current_lr, 6),
                        round(grad_norm, 4),
                    ]
                )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("F1_Macro/val", val_metrics["f1_macro"], epoch)
                self.writer.add_scalar("Balanced_Acc/val", val_metrics["balanced_accuracy"], epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)

            print(
                f"Epoch [{epoch_num:02d}/{self.config.training.num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"Macro F1: {val_metrics['f1_macro']:.4f} | Bal Acc: {val_metrics['balanced_accuracy']*100:.2f}%"
            )

            # Checkpoint management for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saved_path = self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    filename="best_model.pth",
                )
                print(f"  --> Best model checkpoint saved to: {saved_path}")

                # Save validation confusion matrix plot and CSV for best model
                Evaluator.save_confusion_matrix(
                    val_metrics["confusion_matrix"],
                    output_prefix=self.reports_dir / "confusion_matrix_val",
                )

            # EarlyStopping check
            if self.early_stopping(val_loss):
                print(f"\n[EarlyStopping] Validation loss did not improve for {self.early_stopping.patience} epochs. Stopping training early.")
                break

        print("\n" + "=" * 65)
        print(f"TRAINING COMPLETED! Best Validation Loss: {best_val_loss:.4f}")
        print("=" * 65)

        if self.writer is not None:
            self.writer.close()

        return {"best_val_loss": best_val_loss, "history": history}
