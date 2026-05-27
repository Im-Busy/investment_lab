#!/usr/bin/env python3
"""CLI for NLP sentiment pipeline — SVM+TF-IDF, BiLSTM+LR, and Ensemble.

Supports training on sample data (with distant supervision via emoticons)
and prediction on arbitrary text.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.sentiment_pipeline import (
    SVMTfidfSentiment,
    BiLSTMSentiment,
    SentimentEnsemble,
    apply_distant_supervision,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/sentiment")

SENTIMENT_LABEL = {1: "positive", -1: "negative", 0: "neutral"}

SAMPLE_TEXTS: list[str] = [
    "This stock is going to the moon! :)",
    "Great earnings report, very bullish :D",
    "Buying the dip, this is a steal ;)",
    "Strong fundamentals and growth outlook",
    "Market rally continues, new highs ahead",
    "Terrible quarter, selling everything :(",
    "Bearish outlook for the rest of the year :-(",
    "This company is going bankrupt :/",
    "Downgraded by analysts, avoid this stock :-\\",
    "Market crash incoming, get out now",
    "Earnings beat expectations by 15%",
    "Revenue growth accelerating quarter over quarter",
    "New product launch exceeded all forecasts",
    "Management increased dividend by 20%",
    "Analysts upgrade rating to outperform",
    "Missed earnings for the third straight quarter :(",
    "Layoffs announced, cost cutting measures",
    "Regulatory investigation into accounting practices",
    "Supply chain disruptions hurting margins :-(",
    "Debt downgrade from investment grade to junk :/",
    "Holding steady, waiting for next catalyst",
    "Sideways trading expected this week",
    "Nothing new to report, neutral outlook",
    "Consolidating after recent run-up",
    "In line with expectations, no surprises",
]


def _get_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def train_model(model_name: str) -> str:
    logger.info("Labeling %d sample texts via distant supervision", len(SAMPLE_TEXTS))
    labeled_texts, labels, ds_result = apply_distant_supervision(SAMPLE_TEXTS)
    logger.info(ds_result.summary())

    if len(labeled_texts) < 5:
        logger.warning(
            "Too few labeled samples (%d). Using all texts with synthetic labels.",
            len(labeled_texts),
        )
        labeled_texts = SAMPLE_TEXTS
        labels = [1] * 8 + [-1] * 8 + [0] * 9

    model_dir = _get_model_dir()

    if model_name == "svm":
        model = SVMTfidfSentiment()
        model.fit(labeled_texts, labels)
        save_path = model_dir / "svm_tfidf_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("SVM model saved to %s", save_path)

        result = model.predict("Bullish outlook for the quarter ahead")
        logger.info("Test prediction: %s (conf=%.3f)", result.label, result.confidence)

    elif model_name == "bilstm":
        model = BiLSTMSentiment()
        try:
            model.fit(labeled_texts, labels)
            save_path = model_dir / "bilstm_model.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("BiLSTM model saved to %s", save_path)
        except (ImportError, RuntimeError) as e:
            logger.error("BiLSTM training failed: %s", e)
            return f"BiLSTM training failed: {e}"

    elif model_name == "ensemble":
        model = SentimentEnsemble()
        model.fit(labeled_texts, labels)
        save_path = model_dir / "ensemble_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Ensemble model saved to %s", save_path)

        pred = model.predict("Bearish signals from technical analysis")
        logger.info("Test prediction: %d", pred)

    else:
        return f"Unknown model: {model_name}"

    return f"Model '{model_name}' trained and saved to {model_dir}"


def predict_text(model_name: str, text: str) -> str:
    model_dir = _get_model_dir()
    path_map = {
        "svm": model_dir / "svm_tfidf_model.pkl",
        "bilstm": model_dir / "bilstm_model.pkl",
        "ensemble": model_dir / "ensemble_model.pkl",
    }
    load_path = path_map.get(model_name)
    if load_path is None or not load_path.exists():
        return f"No saved model found for '{model_name}' at {load_path}. Train first with --train."

    with open(load_path, "rb") as f:
        model = pickle.load(f)

    if model_name == "svm":
        result = model.predict(text)
        return json.dumps(
            {
                "text": text,
                "model": model_name,
                "prediction": result.prediction,
                "confidence": round(result.confidence, 4),
                "label": result.label,
            },
            indent=2,
        )
    elif model_name == "bilstm":
        pred = int(model.predict(text))
        label = SENTIMENT_LABEL.get(pred, "unknown")
        return json.dumps(
            {"text": text, "model": model_name, "prediction": pred, "label": label}, indent=2
        )
    elif model_name == "ensemble":
        pred = int(model.predict(text))
        label = SENTIMENT_LABEL.get(pred, "unknown")
        return json.dumps(
            {"text": text, "model": model_name, "prediction": pred, "label": label}, indent=2
        )
    else:
        return f"Unknown model: {model_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP sentiment pipeline — train and predict")
    parser.add_argument(
        "--model",
        default="svm",
        choices=["svm", "bilstm", "ensemble"],
        help="Model type (default: svm)",
    )
    parser.add_argument("--train", action="store_true", help="Train model on sample data")
    parser.add_argument("--predict", type=str, metavar="TEXT", help="Predict sentiment on text")
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        sys.exit(1)

    if args.train:
        result = train_model(args.model)
        print(result)

    if args.predict:
        result = predict_text(args.model, args.predict)
        print(result)


if __name__ == "__main__":
    main()
