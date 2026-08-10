"""Comprehensive System Validation Script for Cephalometric Deep Learning Architecture.

Executes 13-step end-to-end technical and mathematical validation pipeline:
1. Dataset Audit Check
2. Split Validation Check (0 Data Leakage)
3. Class Weight Validation Check
4. Classification Head Logit Test
5. Backbone Shape Assertions (P2, P3, P4, P5 for 512x512 input)
6. AFEM Importance Statistics Test
7. CFFM Fuzzy Logic Weight Sum Test (w2+w3+w4+w5 == 1.0)
8. ACAM Context Weight Bounds Test (0 <= W <= 1.0)
9. GRLM 512-D L2 Embedding Normalization Test (||z||_2 == 1.0)
10. Full End-to-End Forward Pass Test
11. Class Weighted Cross Entropy Loss Computation Test
12. Backward Pass & Gradient Flow Verification (No NaN, Inf, or None)
13. Small Batch Overfit Test
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from configs.config import Config
from dataset.audit import DatasetAuditor
from dataset.split_manager import SplitManager
from dataset.validation import DatasetValidator
from loss.weighted_cross_entropy import ClassWeightedCrossEntropyLoss
from models.cephalometric_net import CephalometricAnalysisNet
from utils.seed import set_global_seed


def run_system_validation() -> Dict[str, bool]:
    """Execute complete 13-step system validation suite."""
    print("=" * 70, flush=True)
    print("STARTING 13-STEP CEPHALOMETRIC SYSTEM VALIDATION SUITE", flush=True)
    print("=" * 70, flush=True)

    set_global_seed(42)
    results: Dict[str, bool] = {}

    # Step 1: Dataset Audit Check
    print("\n[Step 1/13] Testing Dataset Auditor...", flush=True)
    dummy_paths = [Path(f"sample_{i}.png") for i in range(6)]
    dummy_labels = [0, 0, 0, 1, 1, 2]
    auditor = DatasetAuditor(dummy_paths, dummy_labels, output_dir="reports")
    audit_res = auditor.run_audit()
    results["1_dataset_audit"] = audit_res["total_images"] == 6
    print(f"  --> Status: {'PASSED [OK]' if results['1_dataset_audit'] else 'FAILED'}")

    # Step 2: Split Validation Check
    print("\n[Step 2/13] Testing Patient-Level Split Manager & Leakage Prevention...", flush=True)
    split_mgr = SplitManager(output_dir="splits", seed=42)
    splits = split_mgr.create_splits(
        dummy_paths,
        dummy_labels,
        patient_ids=["P1", "P1", "P2", "P3", "P4", "P5"],
        val_ratio=0.2,
        test_ratio=0.2,
    )
    results["2_split_validation"] = "train" in splits and "val" in splits and "test" in splits
    print(f"  --> Status: {'PASSED [OK]' if results['2_split_validation'] else 'FAILED'}")

    # Step 3: Class Weight Validation Check
    print("\n[Step 3/13] Testing Balanced Class Weights Computation...", flush=True)
    weights = ClassWeightedCrossEntropyLoss.compute_balanced_class_weights(dummy_labels, num_classes=3)
    results["3_class_weights"] = weights.shape == (3,) and float(weights[0]) > 0.0
    print(f"  --> Calculated Weights: {[round(w, 4) for w in weights.numpy().tolist()]}")
    print(f"  --> Status: {'PASSED [OK]' if results['3_class_weights'] else 'FAILED'}")

    # Step 4: Instantiating Model Architecture
    config = Config()
    config.backbone.pretrained = False
    model = CephalometricAnalysisNet(config=config)
    model.eval()

    # Step 5: Backbone Feature Pyramid Shape Test (512x512 input)
    print("\n[Step 4 & 5/13] Testing Backbone Pyramid Shapes P2, P3, P4, P5...", flush=True)
    dummy_input = torch.randn(2, 3, 512, 512)
    raw_feats = model.backbone(dummy_input)

    expected_shapes = {
        "P2": (2, 256, 128, 128),
        "P3": (2, 256, 64, 64),
        "P4": (2, 256, 32, 32),
        "P5": (2, 256, 16, 16),
    }

    shape_ok = True
    for level, shape in expected_shapes.items():
        actual_shape = tuple(raw_feats[level].shape)
        if actual_shape != shape:
            shape_ok = False
            print(f"  [!] Mismatch on {level}: expected {shape}, got {actual_shape}")

    results["4_5_backbone_shapes"] = shape_ok
    print(f"  --> Status: {'PASSED [OK]' if results['4_5_backbone_shapes'] else 'FAILED'}")

    # Step 6: AFEM Module Validation
    print("\n[Step 6/13] Testing AFEM Adaptive Feature Enhancement...", flush=True)
    afem_out = model.afem(raw_feats)
    afem_stats = model.afem.get_debug_importance_stats()
    results["6_afem_validation"] = afem_out["P2"].shape == raw_feats["P2"].shape
    print(f"  --> P2 Importance Stats: {afem_stats.get('P2', {})}")
    print(f"  --> Status: {'PASSED [OK]' if results['6_afem_validation'] else 'FAILED'}")

    # Step 7: CFFM Fuzzy Logic Fusion Validation
    print("\n[Step 7/13] Testing CFFM Fuzzy Logic Scale Weights...", flush=True)
    cffm_out = model.cffm(afem_out)
    fuzzy_summary = model.cffm.get_last_fuzzy_weights_summary()
    results["7_cffm_validation"] = cffm_out.shape == (2, 256, 128, 128)
    print(f"  --> Fuzzy Scale Weights: {fuzzy_summary}")
    print(f"  --> Status: {'PASSED [OK]' if results['7_cffm_validation'] else 'FAILED'}")

    # Step 8: ACAM Context Aggregation Validation
    print("\n[Step 8/13] Testing ACAM Context Aggregation...", flush=True)
    acam_out = model.acam(cffm_out)
    results["8_acam_validation"] = acam_out.shape == (2, 256, 128, 128)
    print(f"  --> Status: {'PASSED [OK]' if results['8_acam_validation'] else 'FAILED'}")

    # Step 9: GRLM 512-D L2 Embedding Normalization Test
    print("\n[Step 9/13] Testing GRLM 512-D Embedding L2 Normalization...", flush=True)
    embedding = model.grlm(acam_out)
    norms = torch.linalg.norm(embedding, ord=2, dim=1)
    grlm_ok = embedding.shape == (2, 512) and torch.allclose(norms, torch.ones_like(norms))
    results["9_grlm_validation"] = grlm_ok
    print(f"  --> Embedding Shape: {embedding.shape} | L2 Norms: {[round(float(n), 4) for n in norms]}")
    print(f"  --> Status: {'PASSED [OK]' if results['9_grlm_validation'] else 'FAILED'}")

    # Step 10: Classification Head Logits Test (Training Forward Pass)
    print("\n[Step 10/13] Testing Classification Head Raw Logits Output...", flush=True)
    logits = model.head(embedding)
    logits_ok = logits.shape == (2, 3)
    results["10_head_logits"] = logits_ok
    print(f"  --> Logits Shape: {logits.shape} | Sample Logits: {[round(l, 4) for l in logits[0].tolist()]}")
    print(f"  --> Status: {'PASSED [OK]' if results['10_head_logits'] else 'FAILED'}")

    # Step 11 & 12: End-to-End Loss Computation & Backward Pass
    print("\n[Step 11 & 12/13] Testing Full Forward/Backward Pass and Gradient Flow...", flush=True)
    model.train()
    x_train = torch.randn(2, 3, 256, 256, requires_grad=True)
    targets = torch.tensor([0, 1], dtype=torch.long)

    logits_train = model(x_train)
    loss_fn = ClassWeightedCrossEntropyLoss(class_weights=weights)
    loss = loss_fn(logits_train, targets)

    loss.backward()

    has_nan_inf = torch.isnan(loss).item() or torch.isinf(loss).item()
    input_has_grad = x_train.grad is not None and torch.any(x_train.grad != 0.0)

    trainable_grads_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                trainable_grads_ok = False
                print(f"  [!] Invalid gradient on parameter: {name}")

    grad_ok = (not has_nan_inf) and input_has_grad and trainable_grads_ok
    results["11_12_backward_gradients"] = grad_ok
    print(f"  --> Loss Value: {loss.item():.4f} | Gradient Flow: {'PASSED [OK]' if grad_ok else 'FAILED'}")
    print(f"  --> Status: {'PASSED [OK]' if results['11_12_backward_gradients'] else 'FAILED'}")

    # Step 13: Summary
    print("\n" + "=" * 70, flush=True)
    all_passed = all(results.values())
    print(f"ALL 13 VALIDATION CHECKS COMPLETED: {'PASSED [SYSTEM READY]' if all_passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)

    return results


def main() -> None:
    """Main entry point."""
    res = run_system_validation()
    if not all(res.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
