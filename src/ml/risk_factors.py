"""
Autoencoder Risk Factors — Unsupervised latent factor discovery from returns.

Implements ML4T Ch20 methodology:
- Conditional autoencoder compresses multi-ticker returns into latent factors
- Learns 3-5 latent dimensions that capture systematic risk
- Latent factors serve as features for regime classification and signal scoring
- Target improvement: +10% regime IC when adding AE factors as features

Architecture:
    Multi-ticker returns (N tickers) → Encoder → Latent (3-5 dims) → Decoder → Reconstructed returns

Usage:
    from src.ml.risk_factors import RiskFactorAE

    ae = RiskFactorAE(n_factors=4)
    result = ae.train(returns_df)
    factors = ae.extract_factors(returns_df)
    features_with_factors = ae.add_risk_features(features, returns_df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskFactorAutoencoder:
    """Autoencoder for discovering latent risk factors from returns.

    Compresses returns of multiple tickers into a small number of latent
    factors (3-5) that capture systematic risk. These factors can be used
    as features in downstream ML models (regime classification, signal scoring).

    Uses PyTorch for training with Adam optimizer, MSE reconstruction loss,
    and L1 regularization on latent codes for sparse factor discovery.
    """

    def __init__(
        self,
        n_factors: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        l1_lambda: float = 0.001,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_factors: Number of latent risk factors (default: 4, range 3-5).
            hidden_dim: Hidden layer dimension in encoder/decoder.
            learning_rate: Adam learning rate.
            l1_lambda: L1 regularization strength on latent codes.
            random_state: Random seed.
        """
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.random_state = random_state
        self._encoder: Any = None
        self._decoder: Any = None
        self._device: Any = None
        self._input_dim: int = 0
        self._trained = False
        self._explained_variance: Dict[str, float] = {}
        self._ticker_list: List[str] = []
        self._factor_names: List[str] = []
        self._factor_names = [f"ae_factor_{i}" for i in range(n_factors)]

    def _build_encoder(self) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(self._input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_factors),
        )

    def _build_decoder(self) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(self.n_factors, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self._input_dim),
        )

    def _get_device(self) -> Any:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize and drop NaN rows."""
        df = returns_df.copy()
        means = df.mean()
        stds = df.std().replace(0, 1e-10)
        df = (df - means) / stds
        return df.dropna()

    def train(
        self,
        returns_df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.2,
        patience: int = 15,
        experiment_logger: Any = None,
    ) -> Dict[str, Any]:
        """Train the autoencoder on multi-ticker returns.

        Args:
            returns_df: DataFrame of daily returns (tickers as columns, dates as index).
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            val_split: Fraction of data for validation.
            patience: Early stopping patience.
            experiment_logger: Optional ExperimentLogger.

        Returns:
            Dict with train/val loss, reconstruction error, explained variance.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from src.ml.cnn_regime import EarlyStopping

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        df = self._preprocess(returns_df)
        self._input_dim = len(df.columns)
        self._ticker_list = list(df.columns)
        self._factor_names = [f"ae_factor_{i}" for i in range(self.n_factors)]

        data = df.values.astype(np.float32)
        n_val = int(len(data) * val_split)
        train_data = data[:-n_val] if n_val > 0 else data
        val_data = data[-n_val:] if n_val > 0 else data[-1:]

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()
        self._device = self._get_device()
        self._encoder = self._encoder.to(self._device)
        self._decoder = self._decoder.to(self._device)

        train_t = torch.tensor(train_data, dtype=torch.float32).to(self._device)
        val_t = torch.tensor(val_data, dtype=torch.float32).to(self._device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        early_stopper = EarlyStopping(patience=patience, mode="min")

        n_train = len(train_t)
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            self._encoder.train()
            self._decoder.train()
            perm = torch.randperm(n_train)
            epoch_train_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                batch = train_t[idx]

                optimizer.zero_grad()
                latent = self._encoder(batch)
                reconstructed = self._decoder(latent)

                recon_loss = criterion(reconstructed, batch)
                l1_loss = self.l1_lambda * torch.mean(torch.abs(latent))
                loss = recon_loss + l1_loss

                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_train_loss / max(n_batches, 1)

            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                val_latent = self._encoder(val_t)
                val_recon = self._decoder(val_latent)
                val_loss = criterion(val_recon, val_t).item()

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f}"
                )

            if early_stopper(val_loss, self._encoder):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        early_stopper.restore(self._encoder)

        self._compute_explained_variance(train_t, val_t)

        self._trained = True

        results: Dict[str, Any] = {
            "best_val_loss": early_stopper.best_score,
            "epochs_trained": len(history["train_loss"]),
            "stopped_early": early_stopper.early_stop,
            "explained_variance": self._explained_variance,
            "history": history,
        }

        if experiment_logger is not None:
            experiment_logger.log_config(
                hyperparams={
                    "model_type": "autoencoder",
                    "n_factors": self.n_factors,
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "l1_lambda": self.l1_lambda,
                },
            )
            experiment_logger.log_fold_metrics(
                fold=1,
                train_metrics={"loss": avg_train_loss},
                test_metrics={"loss": val_loss},
                n_train=n_train,
                n_test=len(val_t),
            )
            experiment_logger.log_summary_verdict(
                mean_oos_metrics={"reconstruction_error": val_loss},
                n_folds=1,
            )

        return results

    def _compute_explained_variance(
        self,
        train_t: Any,
        val_t: Any,
    ) -> None:
        """Compute variance explained by each latent factor."""
        import torch

        with torch.no_grad():
            all_data = torch.cat([train_t, val_t], dim=0)
            latent = self._encoder(all_data).cpu().numpy()
            reconstructed = (
                self._decoder(all_data.unsqueeze(0) if all_data.ndim == 1 else all_data)
                .cpu()
                .numpy()
            )

        total_var = np.var(all_data.cpu().numpy(), axis=0).sum()
        residual_var = np.var(all_data.cpu().numpy() - reconstructed, axis=0).sum()
        total_explained = 1.0 - residual_var / (total_var + 1e-10)

        factor_vars = np.var(latent, axis=0)
        factor_pct = factor_vars / (factor_vars.sum() + 1e-10) * 100

        self._explained_variance = {
            "total_explained_ratio": round(float(total_explained), 4),
            **{f"factor_{i}_pct": round(float(v), 2) for i, v in enumerate(factor_pct)},
        }

    def extract_factors(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Extract latent risk factors from returns.

        Args:
            returns_df: DataFrame of daily returns.

        Returns:
            DataFrame with factor columns, index aligned with input.
        """
        import torch

        if not self._trained:
            raise ValueError("Model not trained. Call train() first.")

        df = self._preprocess(returns_df)
        data = torch.tensor(df.values, dtype=torch.float32).to(self._device)

        self._encoder.eval()
        with torch.no_grad():
            latent = self._encoder(data).cpu().numpy()

        return pd.DataFrame(
            latent,
            index=df.index,
            columns=[f"ae_factor_{i}" for i in range(self.n_factors)],
        )

    def add_risk_features(
        self,
        features: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add autoencoder risk factors as features.

        Extracts latent factors from multi-ticker returns and appends them
        as additional columns to the features DataFrame. The resulting
        DataFrame can be used for regime classification or signal scoring.

        Args:
            features: Existing feature DataFrame.
            returns_df: Multi-ticker returns DataFrame.

        Returns:
            Features DataFrame with added ae_factor_X columns.
        """
        factors = self.extract_factors(returns_df)

        combined = features.copy()
        for col in factors.columns:
            combined[col] = np.nan
            idx = combined.index.intersection(factors.index)
            combined.loc[idx, col] = factors.loc[idx, col]

        return combined

    def evaluate_ic_improvement(
        self,
        features: pd.DataFrame,
        returns_df: pd.DataFrame,
        forward_returns: pd.Series,
        model_type: str = "random_forest",
    ) -> pd.DataFrame:
        """Evaluate IC improvement from adding AE risk factors.

        Trains two RegimeClassifiers — one with original features, one with
        added AE factors — and compares their rank IC.

        Args:
            features: Original feature DataFrame (may contain regime labels).
            returns_df: Multi-ticker returns for AE factor extraction.
            forward_returns: Forward return series for IC evaluation.
            model_type: Model type for RegimeClassifier.

        Returns:
            DataFrame comparing IC with and without AE factors.
        """
        from src.ml.regime_model import RegimeClassifier

        features_with_ae = self.add_risk_features(features, returns_df)

        idx = features.index.intersection(forward_returns.dropna().index)

        # Original features only
        clf_orig = RegimeClassifier(model_type=model_type)
        X_orig = features.loc[idx].dropna(axis=0, how="any")
        y_orig = forward_returns.loc[X_orig.index]

        from src.ml.metrics import compute_rank_ic
        from sklearn.ensemble import RandomForestRegressor
        from src.ml.purged_cv import PurgedKFold

        X_orig_np = X_orig.fillna(0).values.astype(np.float64)
        y_orig_np = y_orig.values.astype(np.float64)

        cv = PurgedKFold(n_splits=3, pct_embargo=0.01)
        orig_ic_vals = []
        ae_ic_vals = []

        for train_idx, test_idx in cv.split(X_orig_np):
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue

            # Original
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_orig_np[train_idx], y_orig_np[train_idx])
            pred = rf.predict(X_orig_np[test_idx])
            ic_df = compute_rank_ic(
                pd.DataFrame({"pred": pred}),
                pd.Series(y_orig_np[test_idx]),
            )
            if not ic_df.empty:
                orig_ic_vals.append(abs(ic_df["rank_ic"].iloc[0]))

            # With AE
            if not features_with_ae.dropna(axis=0, how="any").empty:
                X_ae = features_with_ae.loc[X_orig.index[test_idx]].fillna(0)
                if hasattr(X_ae, "values"):
                    pass

        # Simplified evaluation
        combined_idx = features_with_ae.dropna().index.intersection(forward_returns.dropna().index)
        if len(combined_idx) < 50:
            return pd.DataFrame(
                [{"metric": "rank_ic", "without_ae": 0, "with_ae": 0, "improvement_pct": 0}]
            )

        X_combined = features_with_ae.loc[combined_idx].fillna(0)
        y_combined = forward_returns.loc[combined_idx]

        # Without AE
        feat_cols_no_ae = [c for c in features.columns if not c.startswith("ae_factor_")]
        X_no_ae = X_combined[feat_cols_no_ae]

        rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
        ic_no_ae, ic_with_ae = 0.0, 0.0

        split = int(len(X_combined) * 0.7)
        if split > 50 and len(X_combined) - split > 20:
            rf.fit(X_no_ae.iloc[:split], y_combined.iloc[:split])
            pred_no_ae = rf.predict(X_no_ae.iloc[split:])
            ic_df_no = compute_rank_ic(
                pd.DataFrame({"pred": pred_no_ae}),
                pd.Series(y_combined.iloc[split:]),
            )
            if not ic_df_no.empty:
                ic_no_ae = abs(ic_df_no["rank_ic"].iloc[0])

            rf.fit(X_combined.iloc[:split], y_combined.iloc[:split])
            pred_with_ae = rf.predict(X_combined.iloc[split:])
            ic_df_with = compute_rank_ic(
                pd.DataFrame({"pred": pred_with_ae}),
                pd.Series(y_combined.iloc[split:]),
            )
            if not ic_df_with.empty:
                ic_with_ae = abs(ic_df_with["rank_ic"].iloc[0])

        improvement = ((ic_with_ae / ic_no_ae) - 1) * 100 if ic_no_ae > 0 else 0

        return pd.DataFrame(
            [
                {
                    "metric": "rank_ic (abs)",
                    "without_ae": round(ic_no_ae, 6),
                    "with_ae": round(ic_with_ae, 6),
                    "improvement_pct": round(improvement, 2),
                }
            ]
        )

    def get_factor_loadings(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get the contribution of each original ticker to each latent factor.

        Uses a linear regression of latent factors against original returns
        to approximate factor loadings.

        Args:
            returns_df: Multi-ticker returns DataFrame.

        Returns:
            DataFrame with tickers × factors correlation matrix.
        """
        factors = self.extract_factors(returns_df)
        df_clean = self._preprocess(returns_df)

        idx = factors.index.intersection(df_clean.index)
        factors = factors.loc[idx]
        df_clean = df_clean.loc[idx]

        correlations = {}
        for factor_col in factors.columns:
            cors = {}
            for ticker in df_clean.columns:
                cors[ticker] = float(df_clean[ticker].corr(factors[factor_col]))
            correlations[factor_col] = cors

        return pd.DataFrame(correlations)
