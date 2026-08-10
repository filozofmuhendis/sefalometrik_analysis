"""Checkpoint System Utility.

Manages saving and loading PyTorch model checkpoints, optimizer states, learning rate scheduler
states, epoch numbers, best validation metrics, class weights, config, and random seed.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manages saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: Union[str, Path] = "checkpoints") -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Path to directory where checkpoints are saved.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_git_commit_hash() -> str:
        """Get git commit hash string if repository is available."""
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "N/A (Git commit hash unretrievable)"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
        val_loss: float,
        filename: str = "best_model.pth",
        class_weights: Optional[torch.Tensor] = None,
        config: Optional[Any] = None,
        seed: int = 42,
        dataset_split_manifest: Optional[str] = None,
        class_mapping: Optional[Dict[str, Any]] = None,
        best_validation_macro_f1: Optional[float] = None,
        dataset_manifest_paths: Optional[Dict[str, str]] = None,
        best_metric_name: Optional[str] = None,
    ) -> Path:
        """Save model state, optimizer state, scheduler state, and complete training metadata."""
        filepath = self.checkpoint_dir / filename
        checkpoint_data: Dict[str, Any] = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_metric": best_validation_macro_f1 if best_validation_macro_f1 is not None else val_loss,
            "best_metric_name": best_metric_name or ("macro_f1" if best_validation_macro_f1 is not None else "val_loss"),
            "best_validation_macro_f1": best_validation_macro_f1,
            "seed": seed,
            "git_commit_hash": self._get_git_commit_hash(),
            "class_weights": class_weights.cpu() if class_weights is not None else None,
            "config": config,
            "dataset_split_manifest": dataset_split_manifest,
            "dataset_manifest_paths": dataset_manifest_paths,
            "class_mapping": class_mapping,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(checkpoint_data, filepath)
        return filepath

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        filename: str = "best_model.pth",
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Load model state, optimizer state, and scheduler state from a checkpoint file."""
        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint_data = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint_data and checkpoint_data["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint_data and checkpoint_data["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        return checkpoint_data
