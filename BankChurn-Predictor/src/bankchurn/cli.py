"""Command-line interface for BankChurn Predictor.

This module provides the main CLI entry point with subcommands for:
- train: Train a new model
- evaluate: Evaluate a trained model
- predict: Make predictions on new data
- hyperopt: Hyperparameter optimization
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import BankChurnConfig
from .evaluation import ModelEvaluator
from .prediction import ChurnPredictor
from .training import ChurnTrainer

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "bankchurn.log", level: int = logging.INFO) -> None:
    """Configure logging for CLI.

    Parameters
    ----------
    log_file : str, default="bankchurn.log"
        Path to log file.
    level : int, default=logging.INFO
        Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def train_command(args: argparse.Namespace) -> int:
    """Execute train command.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        0 for success, non-zero for failure.
    """
    try:
        # Load config
        config = BankChurnConfig.from_yaml(args.config)

        # Initialize trainer
        trainer = ChurnTrainer(config, random_state=args.seed)

        # Load data
        data = trainer.load_data(args.input)

        # Prepare features
        X, y = trainer.prepare_features(data)

        # Train
        logger.info("Starting training...")
        model, metrics = trainer.train(X, y, use_cv=not args.no_cv)

        # Save model
        trainer.save_model(args.model, args.preprocessor)

        # Save metrics
        if args.metrics_output:
            import json

            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Metrics saved to {metrics_path}")

        logger.info("Training completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """Execute evaluate command.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        0 for success, non-zero for failure.
    """
    try:
        # Load model
        evaluator = ModelEvaluator.from_files(args.model, args.preprocessor)

        # Load data
        data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(data)} samples for evaluation")

        # Prepare data (assuming config for column names)
        config = BankChurnConfig.from_yaml(args.config)
        y = data[config.data.target_column]
        X = data.drop(columns=[config.data.target_column])

        # Evaluate
        evaluator.evaluate(X, y, output_path=args.output)

        # Fairness metrics if requested
        if args.fairness_features:
            fairness_features = args.fairness_features.split(",")
            fairness_metrics = evaluator.compute_fairness_metrics(X, y, fairness_features)

            # Save fairness metrics
            if args.output:
                import json

                fairness_path = Path(args.output).parent / "fairness_metrics.json"
                with open(fairness_path, "w") as f:
                    json.dump(fairness_metrics, f, indent=2)

                logger.info(f"Fairness metrics saved to {fairness_path}")

        logger.info("Evaluation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def predict_command(args: argparse.Namespace) -> int:
    """Execute predict command.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        0 for success, non-zero for failure.
    """
    try:
        # Load model
        predictor = ChurnPredictor.from_files(args.model, args.preprocessor)

        # Make predictions
        predictions = predictor.predict_batch(
            input_path=args.input,
            output_path=args.output,
            include_proba=not args.no_proba,
            threshold=args.threshold,
        )

        logger.info(f"Predictions saved: {len(predictions)} rows")
        logger.info("Prediction completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns
    -------
    parser : ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="BankChurn Predictor - Customer churn prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--config", required=True, help="Path to config YAML")
    train_parser.add_argument("--input", required=True, help="Path to input CSV")
    train_parser.add_argument("--model", default="models/best_model.pkl", help="Path to save model")
    train_parser.add_argument("--preprocessor", default="models/preprocessor.pkl", help="Path to save preprocessor")
    train_parser.add_argument("--metrics-output", help="Path to save metrics JSON")
    train_parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", required=True, help="Path to config YAML")
    eval_parser.add_argument("--input", required=True, help="Path to input CSV with labels")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--preprocessor", required=True, help="Path to preprocessor")
    eval_parser.add_argument("--output", help="Path to save evaluation results")
    eval_parser.add_argument("--fairness-features", help="Comma-separated list of sensitive features for fairness")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions on new data")
    predict_parser.add_argument("--input", required=True, help="Path to input CSV")
    predict_parser.add_argument("--output", required=True, help="Path to save predictions")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--preprocessor", required=True, help="Path to preprocessor")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    predict_parser.add_argument("--no-proba", action="store_true", help="Exclude probability scores")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    exit_code : int
        0 for success, non-zero for failure.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)

    # Set seed if provided
    if args.seed is not None:
        try:
            from common_utils.seed import set_seed

            set_seed(args.seed)
            logger.info(f"Random seed set to {args.seed}")
        except ImportError:
            logger.warning("common_utils.seed not available, skipping seed setting")

    # Execute command
    if args.command == "train":
        return train_command(args)
    elif args.command in ("evaluate", "eval"):
        return evaluate_command(args)
    elif args.command == "predict":
        return predict_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
