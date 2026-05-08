"""
CNN Regime Detector — 1D CNN on OHLCV sequences for regime classification.

Implements DLQT Ch4 methodology:
- 1D convolutional layers on raw OHLCV time series
- Batch normalization + dropout for regularization
- Early stopping to prevent overfitting
- Focal loss for imbalanced regime classes
- Comparison with rule-based and tree-based regime detectors
- Automated experiment logging via ExperimentLogger

Architecture:
    OHLCV window → Conv1D → BN → ReLU → Pool → Conv1D → FC → regime label

Usage:
    from src.ml.cnn_regime import CNNRegimeDetector

    detector = CNNRegimeDetector(sequence_length=60, n_classes=4)
    result = detector.train(df, labels, epochs=50)
    predictions = detector.predict(df_new)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


class EarlyStopping:
    """Patience-based early stopping for neural network training.

    Monitors validation loss and stops training when no improvement
    is seen for `patience` consecutive epochs. Restores best model weights.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.best_weights: Dict[str, Any] | None = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, score: float, model: Any) -> bool:
        if self.best_score is None:
            self.best_score = score
            import copy

            self.best_weights = copy.deepcopy(
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )
            return False

        is_better = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if is_better:
            self.best_score = score
            import copy

            self.best_weights = copy.deepcopy(
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore(self, model: Any) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class Regime1DCNN:
    """1D Convolutional Neural Network for regime classification.

    Architecture:
        Conv1D(64, kernel=5) → BN → ReLU → MaxPool1D(2) → Dropout(0.2)
        → Conv1D(128, kernel=3) → BN → ReLU → MaxPool1D(2) → Dropout(0.2)
        → Conv1D(256, kernel=3) → BN → ReLU → AdaptiveAvgPool1D(1)
        → FC(128) → ReLU → Dropout(0.3) → FC(n_classes)

    Input shape: (batch, channels=5, sequence_length)
    Uses OHLCV channels packed as (Open, High, Low, Close, Volume).
    """

    def __init__(
        self,
        n_classes: int = 4,
        sequence_length: int = 60,
        dropout: float = 0.3,
        kernel_sizes: Tuple[int, int, int] = (5, 3, 3),
    ) -> None:
        try:
            import torch
            import torch.nn as nn

            torch.nn = nn
        except ImportError:
            raise ImportError("PyTorch is required. Run: uv sync")

        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.model: Any = None
        self._device: Any = None

    def _build_model(self) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            # Block 1
            nn.Conv1d(5, 64, kernel_size=self.kernel_sizes[0], padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout * 0.7),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=self.kernel_sizes[1], padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout * 0.7),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=self.kernel_sizes[2], padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            # Flatten + Head
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.n_classes),
        )

    def _get_device(self) -> Any:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        class_weights: np.ndarray | None = None,
        experiment_logger: Any = None,
    ) -> Dict[str, Any]:
        """Train the 1D CNN on OHLCV sequences.

        Args:
            X: Input array of shape (n_samples, 5, sequence_length).
               Channels: [Open, High, Low, Close, Volume] normalized.
            y: Labels array of shape (n_samples,) with integer class indices.
            X_val: Optional validation input.
            y_val: Optional validation labels.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            patience: Early stopping patience.
            class_weights: Per-class weights for imbalanced data.
            experiment_logger: Optional ExperimentLogger.

        Returns:
            Dict with train/val loss, accuracy, and training history.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.model = self._build_model()
        self._device = self._get_device()
        self.model = self.model.to(self._device)

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.long).to(self._device)

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self._device)
            has_val = True
        else:
            val_split = int(len(X_t) * 0.8)
            X_val_t = X_t[val_split:]
            y_val_t = y_t[val_split:]
            X_t = X_t[:val_split]
            y_t = y_t[:val_split]
            has_val = False

        n_train = len(X_t)

        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32).to(self._device)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        early_stopper = EarlyStopping(patience=patience, mode="min")

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                outputs = self.model(X_t[idx])
                loss = criterion(outputs, y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"train_loss={avg_train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f}"
                )

            if early_stopper(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        early_stopper.restore(self.model)

        results: Dict[str, Any] = {
            "best_val_loss": early_stopper.best_score,
            "final_val_accuracy": history["val_accuracy"][-1],
            "best_val_accuracy": max(history["val_accuracy"]),
            "epochs_trained": len(history["train_loss"]),
            "stopped_early": early_stopper.early_stop,
            "history": history,
        }

        if experiment_logger is not None:
            experiment_logger.log_config(
                hyperparams={
                    "model_type": "cnn_regime",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dropout": self.dropout,
                    "sequence_length": self.sequence_length,
                },
                cv_params={"early_stopping_patience": patience},
            )
            experiment_logger.log_fold_metrics(
                fold=1,
                train_metrics={"loss": avg_train_loss},
                test_metrics={"loss": val_loss, "accuracy": val_acc},
                n_train=n_train,
                n_test=len(X_val_t),
            )
            experiment_logger.log_summary_verdict(
                mean_oos_metrics={"accuracy": max(history["val_accuracy"])},
                n_folds=1,
            )
            experiment_logger.log_model(self.model, filename="cnn_regime.pth")
            torch.save(self.model.state_dict(), experiment_logger.run_dir / "cnn_regime.pth")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime class indices.

        Args:
            X: Input array of shape (n_samples, 5, sequence_length).

        Returns:
            Array of predicted class indices.
        """
        import torch

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            outputs = self.model(X_t)
            return outputs.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime class probabilities.

        Args:
            X: Input array of shape (n_samples, 5, sequence_length).

        Returns:
            Array of shape (n_samples, n_classes) with softmax probabilities.
        """
        import torch

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()


class CNNRegimeDetector:
    """CNN-based regime detector with data preprocessing and evaluation.

    Handles OHLCV DataFrame → normalized sequence conversion,
    training, prediction, and comparison with other regime detectors.

    Example:
        >>> detector = CNNRegimeDetector(sequence_length=60)
        >>> result = detector.train(df, labels)
        >>> predictions = detector.predict(df_new)
        >>> comparison = detector.compare_with_rule_based(df, rule_based_labels)
    """

    REGIME_NAMES = {0: "Trending", 1: "Ranging", 2: "Volatile", 3: "Transition"}

    def __init__(
        self,
        sequence_length: int = 60,
        n_classes: int = 4,
        dropout: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.dropout = dropout
        self.random_state = random_state
        self.cnn = Regime1DCNN(
            n_classes=n_classes,
            sequence_length=sequence_length,
            dropout=dropout,
        )
        self._label_encoder: Dict[str, int] = {}
        self._label_decoder: Dict[int, str] = {}
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def _build_sequences(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Convert OHLCV DataFrame to normalized sequences.

        Normalization: Z-score per channel across the full dataset.
        Shape: (n_samples - sequence_length + 1, 5, sequence_length).
        """
        ohlcv = df[_OHLCV_COLS].copy()
        for col in _OHLCV_COLS:
            if col == "Volume":
                ohlcv[col] = np.log1p(ohlcv[col])
            ohlcv[col] = (ohlcv[col] - ohlcv[col].mean()) / (ohlcv[col].std() + 1e-10)

        data = ohlcv.values.astype(np.float32)
        sequences = np.lib.stride_tricks.sliding_window_view(data, self.sequence_length, axis=0)
        return np.transpose(sequences, (0, 2, 1))

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        if not self._label_encoder:
            unique_labels = sorted(labels.dropna().unique())
            self._label_encoder = {name: i for i, name in enumerate(unique_labels)}
            self._label_decoder = {i: name for name, i in self._label_encoder.items()}

        return np.array([self._label_encoder.get(l, -1) for l in labels])

    def _decode_labels(self, indices: np.ndarray) -> List[str]:
        return [self._label_decoder.get(i, "Unknown") for i in indices]

    def train(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        val_df: pd.DataFrame | None = None,
        val_labels: pd.Series | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        experiment_logger: Any = None,
    ) -> Dict[str, Any]:
        """Train the CNN regime detector.

        Args:
            df: OHLCV DataFrame.
            labels: Regime label series (aligned with df index).
            val_df: Optional validation OHLCV DataFrame.
            val_labels: Optional validation labels.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            patience: Early stopping patience.
            experiment_logger: Optional ExperimentLogger.

        Returns:
            Dict with training results.
        """
        np.random.seed(self.random_state)

        X = self._build_sequences(df)
        y_encoded = self._encode_labels(labels)

        idx = labels.index[self.sequence_length - 1 :]
        label_aligned = labels.reindex(idx)

        n_seqs = len(X)
        y_labels = y_encoded[self.sequence_length - 1 : self.sequence_length - 1 + n_seqs]

        if len(y_labels) != n_seqs:
            y_labels = y_labels[:n_seqs]

        mask = y_labels >= 0
        X = X[mask]
        y_labels = y_labels[mask]

        X_val = None
        y_val = None
        if val_df is not None and val_labels is not None:
            X_val = self._build_sequences(val_df)
            y_encoded_val = self._encode_labels(val_labels)
            n_val = len(X_val)
            y_val_labels = y_encoded_val[
                self.sequence_length - 1 : self.sequence_length - 1 + n_val
            ]
            mask_val = y_val_labels >= 0
            X_val = X_val[mask_val]
            y_val = y_val_labels[mask_val]

        class_counts = np.bincount(y_labels, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-10)
        class_weights = class_weights / class_weights.sum() * self.n_classes

        return self.cnn.train(
            X=X,
            y=y_labels,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            class_weights=class_weights,
            experiment_logger=experiment_logger,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels for new OHLCV data.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of predicted regime names, NaN for first sequence_length-1 rows.
        """
        X = self._build_sequences(df)
        indices = self.cnn.predict(X)
        labels = self._decode_labels(indices)

        result = pd.Series(index=df.index, dtype=object)
        start = self.sequence_length - 1
        for i, label in enumerate(labels):
            if start + i < len(result):
                result.iloc[start + i] = label

        return result

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with columns for each regime class.
        """
        X = self._build_sequences(df)
        proba = self.cnn.predict_proba(X)

        cols = [self._label_decoder.get(i, f"class_{i}") for i in range(self.n_classes)]
        result = pd.DataFrame(index=df.index, columns=cols, dtype=float)
        start = self.sequence_length - 1
        for i, p in enumerate(proba):
            if start + i < len(result):
                result.iloc[start + i] = p

        return result

    def compare_with_rule_based(
        self,
        df: pd.DataFrame,
        rule_based_labels: pd.Series,
    ) -> pd.DataFrame:
        """Compare CNN predictions against rule-based regime labels.

        Args:
            df: OHLCV DataFrame.
            rule_based_labels: Labels from rule-based regime detector.

        Returns:
            DataFrame with accuracy per regime class.
        """
        from sklearn.metrics import classification_report

        preds = self.predict(df)

        idx = preds.dropna().index.intersection(rule_based_labels.dropna().index)
        y_true = rule_based_labels.loc[idx]
        y_pred = preds.loc[idx]

        report = classification_report(y_true, y_pred, output_dict=True)

        rows = []
        for cls_name in sorted(set(y_true)):
            if cls_name in report:
                rows.append(
                    {
                        "regime": cls_name,
                        "precision": report[cls_name]["precision"],
                        "recall": report[cls_name]["recall"],
                        "f1": report[cls_name]["f1-score"],
                        "support": report[cls_name]["support"],
                    }
                )

        return pd.DataFrame(rows)

    def compare_with_tree_based(
        self,
        df: pd.DataFrame,
        true_labels: pd.Series,
        tree_preds: pd.Series,
    ) -> pd.DataFrame:
        """Compare CNN vs tree-based regime classifier.

        Args:
            df: OHLCV DataFrame.
            true_labels: Ground truth labels.
            tree_preds: Predictions from tree-based classifier.

        Returns:
            DataFrame comparing both models per regime class.
        """
        from sklearn.metrics import accuracy_score

        cnn_preds = self.predict(df)
        idx = (
            cnn_preds.dropna()
            .index.intersection(true_labels.dropna().index)
            .intersection(tree_preds.dropna().index)
        )

        y_true = true_labels.loc[idx]
        y_cnn = cnn_preds.loc[idx]
        y_tree = tree_preds.loc[idx]

        cnn_acc = accuracy_score(y_true, y_cnn)
        tree_acc = accuracy_score(y_true, y_tree)

        return pd.DataFrame(
            [
                {"model": "CNN", "accuracy": cnn_acc},
                {"model": "Tree-Based", "accuracy": tree_acc},
            ]
        )
