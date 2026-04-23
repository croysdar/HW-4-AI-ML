"""
optimize.py
===========
Optuna hyperparameter search for BNNClassifier (hybrid precision architecture).

Searches over: learning rate, weight decay, blank class weight, gradient clip.
Each trial trains for TRIAL_EPOCHS epochs with cosine annealing and returns
the best val accuracy seen. Unpromising trials are pruned early via MedianPruner.

Typical overnight run: 15 trials × 10 epochs × ~4 min/epoch ≈ 10 hours.

Usage:
  python project/optimize.py
  python project/optimize.py --trials 20 --epochs 10 --data-root project/data_combined
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-step noise

sys.path.insert(0, str(Path(__file__).parent))
from bnn_serengeti2 import (
    BNNClassifier, make_loaders, evaluate,
    DEVICE, BATCH_SIZE, ACCUM_STEPS, BinarizeConv2d,
)

TRIAL_EPOCHS = 10
OUTPUT_FILE  = Path(__file__).parent / "optuna_results.json"


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, data_root: str, num_epochs: int) -> float:
    lr           = trial.suggest_float("lr",           2e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-3, 5e-2, log=True)
    blank_weight = trial.suggest_float("blank_weight", 1.0,  2.5)
    grad_clip    = trial.suggest_float("grad_clip",    0.5,  2.0)

    train_loader, test_loader = make_loaders(data_root)
    model     = BNNClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([blank_weight, 1.0]).to(DEVICE)
    )

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(imgs), labels)
            (loss / ACCUM_STEPS).backward()

            last_batch = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, BinarizeConv2d):
                            m.weight.data.clamp_(-1.0, 1.0)
                optimizer.zero_grad()

        _, val_acc, _, _, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()
        best_val_acc = max(best_val_acc, val_acc)

        print(f"  Trial {trial.number:>2}  Epoch {epoch:>2}/{num_epochs}"
              f"  val={val_acc:.1f}%  best={best_val_acc:.1f}%"
              f"  lr={lr:.2e}  bw={blank_weight:.2f}  gc={grad_clip:.2f}")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optuna search for BNNClassifier hyperparameters")
    parser.add_argument("--trials",    type=int, default=15,
                        help="Number of trials (default: 15)")
    parser.add_argument("--epochs",    type=int, default=TRIAL_EPOCHS,
                        help=f"Epochs per trial (default: {TRIAL_EPOCHS})")
    parser.add_argument("--data-root", default="project/data_combined", metavar="DIR",
                        help="Dataset root (default: project/data_combined)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Search — BNNClassifier")
    print(f"{'='*60}")
    print(f"Trials      : {args.trials}")
    print(f"Epochs/trial: {args.epochs}")
    print(f"Dataset     : {args.data_root}")
    print(f"Search space: lr, weight_decay, blank_weight, grad_clip")
    print(f"{'='*60}\n")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(trial, args.data_root, args.epochs),
        n_trials=args.trials,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best val accuracy : {study.best_value:.1f}%")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:<20} {v:.6g}")

    completed = [t for t in study.trials if t.value is not None]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nCompleted trials  : {len(completed)}")
    print(f"Pruned trials     : {len(pruned)}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "best_val_acc": study.best_value,
        "best_params":  study.best_params,
        "all_trials": [
            {
                "number":  t.number,
                "val_acc": t.value,
                "params":  t.params,
                "state":   str(t.state),
            }
            for t in study.trials
        ],
    }
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {OUTPUT_FILE}")
    print("\nTo train with best params, run:")
    print(f"  .venv/bin/python3 project/bnn_serengeti2.py train \\")
    print(f"    --data-root project/data_combined --epochs 50")
    print(f"  (then update LR/weight_decay/blank_weight/grad_clip in bnn_serengeti2.py)")


if __name__ == "__main__":
    main()
