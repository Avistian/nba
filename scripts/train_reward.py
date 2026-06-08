"""Train the reward model q(x, a) from logged feedback and persist it.

Usage:
    uv run python scripts/train_reward.py --logs data/logs.parquet --out artifacts/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba.config import get_settings
from nba.data.features import featurize
from nba.data.simulator import frame_to_events
from nba.reward.model import RewardModel


def _reliability(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Bucket predictions and report mean predicted vs realized reward per bucket."""
    order = np.argsort(pred)
    pred, actual = pred[order], actual[order]
    splits = np.array_split(np.arange(len(pred)), bins)
    curve: list[dict[str, float]] = []
    for idx in splits:
        if len(idx) == 0:
            continue
        curve.append(
            {
                "n": float(len(idx)),
                "mean_pred": float(pred[idx].mean()),
                "mean_actual": float(actual[idx].mean()),
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reward model from logged feedback.")
    parser.add_argument("--logs", type=Path, default=Path("data/logs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = get_settings()
    settings = settings.model_copy(update={"seed": args.seed})

    frame = pd.read_parquet(args.logs)
    events = frame_to_events(frame)
    model = RewardModel.fit(events, settings=settings, val_frac=args.val_frac)

    # Held-out evaluation on a fresh seeded split mirroring fit's split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(events))
    n_val = max(1, int(len(events) * args.val_frac))
    val_events = [events[i] for i in perm[:n_val]]
    x_val = np.vstack([featurize(e.context, e.action) for e in val_events])
    y_val = np.array([e.reward for e in val_events], dtype=np.float64)

    raw = np.asarray(model.booster.predict(x_val), dtype=np.float64)
    cal = model._predict(x_val)
    metrics = {
        "n_events": len(events),
        "n_val": len(val_events),
        "mse_raw": float(np.mean((raw - y_val) ** 2)),
        "mse_calibrated": float(np.mean((cal - y_val) ** 2)),
        "mae_calibrated": float(np.mean(np.abs(cal - y_val))),
        "mean_pred_calibrated": float(cal.mean()),
        "mean_actual": float(y_val.mean()),
        "reliability_calibrated": _reliability(cal, y_val),
    }

    model.save(args.out)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"trained on {metrics['n_events']} events, validated on {metrics['n_val']}")
    print(f"MSE raw={metrics['mse_raw']:.5f}  calibrated={metrics['mse_calibrated']:.5f}")
    print(
        f"mean pred (cal)={metrics['mean_pred_calibrated']:.4f}  "
        f"actual={metrics['mean_actual']:.4f}"
    )
    print(f"saved model + metrics -> {args.out}")


if __name__ == "__main__":
    main()
