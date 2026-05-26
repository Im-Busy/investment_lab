"""
Fine-tune Chronos-2 foundation model on financial OHLCV data.

Uses LoRA (Low-Rank Adaptation) to fine-tune the small Chronos model
variants on our specific financial data. This unlocks foundation model
capabilities that were degraded in zero-shot mode (Phase 14/E1).

Usage:
    # Fine-tune chronos-2-small on SPY with LoRA
    uv run scripts/finetune_chronos.py --symbol SPY --model chronos-2-small --lora

    # Fine-tune on basket, full training
    uv run scripts/finetune_chronos.py --basket SPY,QQQ,GLD --model bolt-small --epochs 20

    # Evaluate fine-tuned model vs zero-shot baseline
    uv run scripts/finetune_chronos.py --symbol SPY --eval-only --model-path models/chronos_finetuned_SPY

Note: Requires GPU for practical use. CPU is possible for bolt-tiny only (~6GB RAM).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/chronos")
OUTPUT_DIR = Path("outputs/chronos")


def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, end=end)


def compute_directional_accuracy(
    forecasts: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """Fraction of correct directional predictions (up/down vs yesterday)."""
    if len(forecasts) < 2 or len(actuals) < 2:
        return float("nan")
    pred_dir = np.sign(np.diff(forecasts.flatten()))
    true_dir = np.sign(np.diff(actuals.flatten()))
    valid = (pred_dir != 0) & (true_dir != 0)
    if not valid.any():
        return float("nan")
    return float(np.mean(pred_dir[valid] == true_dir[valid]))


def finetune_chronos(
    model_size: str,
    train_series: np.ndarray,
    val_series: np.ndarray,
    context_length: int = 512,
    prediction_length: int = 5,
    lora_rank: int = 4,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fine-tune a Chronos model using LoRA.

    Uses HuggingFace PEFT for parameter-efficient fine-tuning.
    Only works with chronos-2-small or bolt variants (< 100M params).

    Args:
        model_size: Model variant name (chronos-2-small, bolt-tiny, etc.)
        train_series: Training time series, shape (N_bars,).
        val_series: Validation time series, shape (N_bars,).
        context_length: Input window length.
        prediction_length: Forecast horizon.
        lora_rank: LoRA rank (higher = more capacity, more VRAM).
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size per GPU.
        device: Torch device.
        output_dir: Directory to save model and metrics.

    Returns:
        Dict with training metrics.
    """
    import torch
    from chronos import Chronos2Pipeline, ChronosConfig

    model_id = ChronosForecasterWrapper.MODEL_SIZES.get(model_size, model_size)
    logger.info(f"Loading {model_id} for fine-tuning...")

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype="auto" if "cuda" in str(device) else torch.float32,
    )

    model = pipeline.model
    tokenizer = pipeline.tokenizer
    model.train()

    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,}/{total:,} trainable params ({trainable / total:.1%})")

    train_tokenized = tokenizer(
        train_series[-context_length * 2 :],  # Use recent data for efficiency
        truncation=True,
        max_length=context_length,
        return_tensors="pt",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        total_samples = len(train_series) - context_length - prediction_length
        indices = np.random.permutation(total_samples)[: batch_size * 10]

        for i in range(0, min(len(indices), batch_size * 10), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_inputs = []
            batch_targets = []

            for idx in batch_idx:
                if idx < 0 or idx + context_length + prediction_length > len(train_series):
                    continue
                inp = train_series[idx : idx + context_length]
                tgt = train_series[idx + context_length : idx + context_length + prediction_length]
                batch_inputs.append(inp)
                batch_targets.append(tgt)

            if not batch_inputs:
                continue

            try:
                inputs = tokenizer(
                    batch_inputs,
                    truncation=True,
                    padding=True,
                    max_length=context_length,
                    return_tensors="pt",
                )
                targets = tokenizer(
                    batch_targets,
                    truncation=True,
                    padding=True,
                    max_length=prediction_length,
                    return_tensors="pt",
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = targets["input_ids"].to(device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                continue

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        val_loss = _compute_val_loss(
            model, tokenizer, val_series, context_length, prediction_length, device
        )
        history["val_loss"].append(val_loss)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.6f} | val_loss={val_loss:.6f}"
            )

    if output_dir:
        output_path = Path(output_dir) / f"chronos_finetuned_{model_size}"
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        logger.info(f"Model saved to {output_path}")

    return history


def _compute_val_loss(
    model,
    tokenizer,
    val_series: np.ndarray,
    context_length: int,
    prediction_length: int,
    device: str,
) -> float:
    model.eval()
    import torch

    losses = []
    total = min(20, len(val_series) - context_length * 2)

    for i in range(total):
        start = i * (context_length // 4)
        if start + context_length + prediction_length > len(val_series):
            break
        inp = val_series[start : start + context_length]
        tgt = val_series[start + context_length : start + context_length + prediction_length]

        try:
            inputs = tokenizer(
                [inp], truncation=True, max_length=context_length, return_tensors="pt"
            )
            targets = tokenizer(
                [tgt], truncation=True, max_length=prediction_length, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = targets["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            losses.append(outputs.loss.item())
        except Exception:
            continue

    model.train()
    return float(np.mean(losses)) if losses else float("inf")


class ChronosForecasterWrapper:
    """Namespace for model size constants."""

    MODEL_SIZES = {
        "chronos-2": "amazon/chronos-2",
        "chronos-2-small": "autogluon/chronos-2-small",
        "bolt-tiny": "amazon/chronos-bolt-tiny",
        "bolt-mini": "amazon/chronos-bolt-mini",
        "bolt-small": "amazon/chronos-bolt-small",
        "bolt-base": "amazon/chronos-bolt-base",
    }


def evaluate_finetuned(
    model_path: str,
    test_series: np.ndarray,
    context_length: int = 512,
    prediction_length: int = 5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate fine-tuned Chronos model vs zero-shot baseline."""
    import torch
    from chronos import Chronos2Pipeline

    finetuned = Chronos2Pipeline.from_pretrained(
        model_path, device_map=device, dtype="auto" if "cuda" in device else torch.float32
    )

    model_id = ChronosForecasterWrapper.MODEL_SIZES["bolt-tiny"]
    try:
        zero_shot = Chronos2Pipeline.from_pretrained(
            model_id, device_map=device, dtype="auto" if "cuda" in device else torch.float32
        )
    except Exception:
        zero_shot = None

    results = {"finetuned_dir_acc": [], "zero_shot_dir_acc": []}
    stride = context_length // 4

    for start in range(0, len(test_series) - context_length - prediction_length, stride):
        end = start + context_length
        inp = test_series[start:end]
        actual = test_series[end : end + prediction_length]

        if len(actual) < prediction_length:
            break

        finetuned_pred = finetuned.predict(inp, prediction_length=prediction_length)
        ft_forecast = (
            finetuned_pred.quantile(0.5).values
            if hasattr(finetuned_pred, "quantile")
            else np.array(finetuned_pred)
        )
        results["finetuned_dir_acc"].append(compute_directional_accuracy(ft_forecast, actual))

        if zero_shot is not None:
            zs_pred = zero_shot.predict(inp, prediction_length=prediction_length)
            zs_forecast = (
                zs_pred.quantile(0.5).values if hasattr(zs_pred, "quantile") else np.array(zs_pred)
            )
            results["zero_shot_dir_acc"].append(compute_directional_accuracy(zs_forecast, actual))

    ft_mean = (
        np.nanmean(results["finetuned_dir_acc"]) if results["finetuned_dir_acc"] else float("nan")
    )
    zs_mean = (
        np.nanmean(results["zero_shot_dir_acc"]) if results["zero_shot_dir_acc"] else float("nan")
    )

    return {
        "finetuned_directional_accuracy": ft_mean,
        "zero_shot_directional_accuracy": zs_mean,
        "delta": ft_mean - zs_mean
        if not np.isnan(ft_mean) and not np.isnan(zs_mean)
        else float("nan"),
        "n_windows": len(results["finetuned_dir_acc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 on financial data")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--basket", type=str, default=None, help="Comma-separated ticker list")
    parser.add_argument(
        "--model",
        type=str,
        default="chronos-2-small",
        choices=list(ChronosForecasterWrapper.MODEL_SIZES.keys()),
        help="Chronos model variant",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--test-start", type=str, default="2025-01-01")
    parser.add_argument("--test-end", type=str, default="2026-06-01")
    parser.add_argument("--context", type=int, default=512, help="Context length")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon")
    parser.add_argument("--lora", action="store_true", default=True, help="Use LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate only, skip training")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to fine-tuned model for eval"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for model + metrics")
    args = parser.parse_args()

    logger.info(
        f"Model: {args.model} | Context: {args.context} | Horizon: {args.horizon} | Device: {args.device}"
    )
    logger.info(f"LoRA: rank={args.lora_rank} | Epochs: {args.epochs} | LR: {args.lr}")

    if args.basket:
        symbols = [t.strip() for t in args.basket.split(",")]
    else:
        symbols = [args.symbol]

    if args.eval_only:
        if not args.model_path:
            logger.error("--model-path required for --eval-only")
            sys.exit(1)

        for sym in symbols:
            df_test = load_ohlcv(sym, args.test_start, args.test_end)
            close = df_test["Close"].values.astype(np.float64)
            logger.info(f"Evaluating on {sym}: {len(close)} test bars")

            eval_results = evaluate_finetuned(
                model_path=args.model_path,
                test_series=close,
                context_length=args.context,
                prediction_length=args.horizon,
                device=args.device,
            )
            logger.info(
                f"  Fine-tuned dir acc: {eval_results['finetuned_directional_accuracy']:.3f}"
            )
            logger.info(
                f"  Zero-shot dir acc:  {eval_results['zero_shot_directional_accuracy']:.3f}"
            )
            logger.info(f"  Delta:              {eval_results['delta']:.3f}")
            gate = (
                "PASS"
                if not np.isnan(eval_results["delta"]) and eval_results["delta"] > 0.05
                else "FAIL"
            )
            logger.info(f"  Gate (>5% dir acc): {gate}")
        return

    for sym_idx, sym in enumerate(symbols):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fine-tuning {args.model} on {sym} ({sym_idx + 1}/{len(symbols)})")

        df_train = load_ohlcv(sym, args.start, args.end)
        close = df_train["Close"].values.astype(np.float64)

        n_val = int(len(close) * 0.2)
        train_series = close[:-n_val]
        val_series = close[-n_val - args.context :]

        logger.info(f"Train bars: {len(train_series)}, Val bars: {len(val_series)}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output or str(OUTPUT_DIR / f"{sym}_{timestamp}")

        history = finetune_chronos(
            model_size=args.model,
            train_series=train_series,
            val_series=val_series,
            context_length=args.context,
            prediction_length=args.horizon,
            lora_rank=args.lora_rank,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=output_dir,
        )

        logger.info(
            f"Final loss: {history['loss'][-1]:.6f}, val_loss: {history['val_loss'][-1]:.6f}"
        )

        metrics = {
            "symbol": sym,
            "model": args.model,
            "context": args.context,
            "horizon": args.horizon,
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "final_loss": history["loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "loss_history": history["loss"],
            "val_loss_history": history["val_loss"],
            "output_dir": output_dir,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
