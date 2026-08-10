"""Main Entry Point for Modular Cephalometric Analysis Deep Learning System.

Usage:
    python main.py --mode demo
    python main.py --mode predict --image s1.jpg
    python main.py --mode train --epochs 10 --batch_size 4
"""

import argparse
from pathlib import Path
from typing import List

import torch

from configs.config import Config
from models.cephalometric_net import CephalometricAnalysisNet
from inference import CephalometricPredictor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cephalometric X-ray Analysis Deep Learning Architecture CLI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "predict", "train"],
        help="Execution mode: demo, predict, or train.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("s1.jpg"),
        help="Input image path for prediction mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained model checkpoint file (.pth).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training/inference.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    return parser.parse_args()


def run_demo() -> None:
    """Run full pipeline architecture demonstration."""
    print("=" * 65)
    print("CEPHALOMETRIC DEEP LEARNING PIPELINE DEMONSTRATION")
    print("=" * 65)

    config = Config()
    print("Initializing complete end-to-end CephalometricAnalysisNet...")
    model = CephalometricAnalysisNet(config=config)
    model.eval()

    sample_img_path = Path("s1.jpg")
    if sample_img_path.exists():
        print(f"Loading input sample image: {sample_img_path.resolve()}")
        image_input = sample_img_path
    else:
        print("Sample image 's1.jpg' not found. Using synthetic image array...")
        import numpy as np
        image_input = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    print("Running end-to-end inference (ImageEnhancement -> Backbone -> AFEM -> CFFM -> ACAM -> GRLM -> Head)...")
    predictor = CephalometricPredictor(config=config)
    results = predictor.predict(image_input, return_embedding=True)

    print("\nDEMO INFERENCE RESULTS:")
    print("-" * 65)
    print(f"Raw Logits            : {[round(l, 4) for l in results.logits]}")
    print(f"Class Probabilities   : { {k: round(v, 4) for k, v in results.class_probabilities.items()} }")
    print(f"Predicted Class Index : {results.predicted_class_index}")
    print(f"Predicted Class Name  : {results.predicted_class_name}")
    print(f"Prediction Confidence : {results.confidence*100:.2f}%")
    print(f"512-D Embedding Shape : {results.embedding.shape if results.embedding is not None else 'N/A'}")
    print("-" * 65)
    print("Pipeline execution completed successfully!")
    print("=" * 65)


def main() -> None:
    """Main execution function."""
    args = parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "predict":
        if not args.image.exists():
            raise FileNotFoundError(f"Input image file not found: {args.image}")
        print(f"Executing prediction on: {args.image.resolve()}")
        predictor = CephalometricPredictor(checkpoint_path=args.checkpoint)
        results = predictor.predict(args.image)
        print(f"Prediction Class Name: {results.predicted_class_name} | Index: {results.predicted_class_index} | Confidence: {results.confidence*100:.2f}%")
    elif args.mode == "train":
        print(f"Starting training mode for {args.epochs} epochs with batch size {args.batch_size}...")
        from train_full_model import main as run_train
        run_train()


if __name__ == "__main__":
    main()
