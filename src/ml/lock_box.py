"""P25: Lock Box methodology — blind holdout for anti-overfitting.

The lock box is a set-aside test dataset accessed exactly ONCE after ALL
decisions (feature selection, model choice, hyperparameter tuning) are final.
This is the single most effective anti-overfitting measure for trading strategies.

Reference: "I Tried a Bunch of Things" (over-hyping paper), Bailey & López de Prado.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_ACCESSES = 1


class LockBoxSealedError(RuntimeError):
    """Raised when attempting to access the lock box more than once."""


class LockBox:
    """Blind holdout dataset with one-time access enforcement.

    Usage:
        box = LockBox(X, y, metadata={"symbol": "SPY", "date": "2026-05-22"})
        box.seal()  # lock after sealing
        # ... do all training, tuning, model selection ...
        X_test, y_test = box.unseal()  # one-time access
        # box.unseal() would raise LockBoxSealedError
    """

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._X = X
        self._y = y
        self._metadata = metadata or {}
        self._access_count = 0
        self._is_sealed = False
        self._content_hash: str = ""
        self._sealed_at: str = ""

    @property
    def n_samples(self) -> int:
        return len(self._X) if isinstance(self._X, pd.DataFrame) else self._X.shape[0]

    @property
    def is_sealed(self) -> bool:
        return self._is_sealed

    @property
    def is_accessed(self) -> bool:
        return self._access_count >= MAX_ACCESSES

    def seal(self) -> str:
        """Seal the lock box. Returns the content hash for pre-registration."""
        if isinstance(self._X, pd.DataFrame):
            raw = pd.concat([self._X, self._y.to_frame(name="target")], axis=1).to_numpy()
        else:
            raw = np.column_stack([self._X, self._y.reshape(-1, 1)])
        self._content_hash = hashlib.sha256(raw.tobytes()).hexdigest()[:16]
        self._sealed_at = datetime.now(timezone.utc).isoformat()
        self._is_sealed = True
        logger.info(
            "LockBox sealed: hash=%s n_samples=%d at=%s",
            self._content_hash,
            self.n_samples,
            self._sealed_at,
        )
        return self._content_hash

    def unseal(self) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
        """Unseal and return holdout data (one-time access only)."""
        if not self._is_sealed:
            raise LockBoxSealedError("LockBox must be sealed before unsealing")
        if self._access_count >= MAX_ACCESSES:
            raise LockBoxSealedError(
                f"LockBox already accessed {self._access_count} time(s). Max is {MAX_ACCESSES}."
            )
        self._access_count += 1
        logger.info(
            "LockBox accessed (%d/%d) at %s",
            self._access_count,
            MAX_ACCESSES,
            datetime.now(timezone.utc).isoformat(),
        )
        return self._X, self._y

    def to_state(self) -> dict[str, Any]:
        """Serializable state for checkpointing."""
        return {
            "n_samples": self.n_samples,
            "content_hash": self._content_hash,
            "sealed_at": self._sealed_at,
            "access_count": self._access_count,
            "is_sealed": self._is_sealed,
            "metadata": self._metadata,
        }


def create_lock_box(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.20,
    metadata: dict[str, Any] | None = None,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.Series, LockBox]:
    """Split data into train + lock box and return sealed LockBox.

    Returns (X_train, y_train, lock_box).
    """
    n = len(X)
    n_test = max(int(n * test_frac), 1)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    box = LockBox(X_test, y_test, metadata=metadata)
    box.seal()
    return X_train, y_train, box


def create_lock_box_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.20,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, LockBox]:
    """Split chronologically — holds out the LAST test_frac of data.

    Returns (X_train, y_train, lock_box).
    """
    n = len(X)
    n_test = max(int(n * test_frac), 1)
    split_idx = n - n_test

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    box = LockBox(X_test, y_test, metadata=metadata)
    box.seal()
    return X_train, y_train, box
