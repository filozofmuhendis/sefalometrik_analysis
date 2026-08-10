"""EarlyStopping Callback Utility.

Monitors validation loss during model training and signals early termination
when loss fails to improve beyond a specified delta threshold after patience epochs.
"""

from typing import Optional


class EarlyStopping:
    """Early stopping callback to halt training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        """Initialize EarlyStopping callback.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum validation metric improvement threshold.
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if early stopping criteria is triggered.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            Boolean flag indicating if training should early stop.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
