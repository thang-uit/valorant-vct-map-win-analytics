"""Train baselines and GRU, then export the small public web-demo payload."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    ROUND_FEATURES,
    build_prefix_samples,
    chronological_split,
    load_prepared_data,
)
from src.models import (  # noqa: E402
    PrefixDataset,
    WinProbabilityGRU,
    build_baseline_features,
    build_sequence_vocabulary,
    collate_prefix_batch,
    evaluate_probabilities,
)


SEED = 42
SEASONS = (2023, 2024)
MIN_ROUND = 3
BATCH_SIZE = 256
MAX_EPOCHS = 18
EARLY_STOPPING_PATIENCE = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_data_root() -> Path:
    return REPO_ROOT.parent / "RawData" / "Valorant-Champion-Tour-2021-2026-Data"


def make_loader(rows: list[dict], vocabulary, shuffle: bool) -> DataLoader:
    dataset = PrefixDataset(rows, vocabulary)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_prefix_batch,
    )


@torch.no_grad()
def sequence_probabilities(
    model: WinProbabilityGRU, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for histories, lengths, maps, agents_a, agents_b, _ in loader:
        logits = model(
            histories.to(device),
            lengths,
            maps.to(device),
            agents_a.to(device),
            agents_b.to(device),
        )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values)


def fit_gru(
    train_rows: list[dict],
    validation_rows: list[dict],
    vocabulary,
    use_agent_attention: bool,
    device: torch.device,
) -> tuple[WinProbabilityGRU, list[dict]]:
    model = WinProbabilityGRU(
        map_count=len(vocabulary.maps),
        agent_count=len(vocabulary.agents),
        round_feature_count=len(ROUND_FEATURES),
        use_agent_attention=use_agent_attention,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.BCEWithLogitsLoss()
    train_loader = make_loader(train_rows, vocabulary, shuffle=True)
    validation_loader = make_loader(validation_rows, vocabulary, shuffle=False)
    validation_targets = np.asarray(
        [row["target"] for row in validation_rows], dtype=np.int64
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for histories, lengths, maps, agents_a, agents_b, targets in train_loader:
            optimizer.zero_grad()
            logits = model(
                histories.to(device),
                lengths,
                maps.to(device),
                agents_a.to(device),
                agents_b.to(device),
            )
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_probabilities = sequence_probabilities(
            model, validation_loader, device
        )
        validation_metrics = evaluate_probabilities(
            validation_targets, validation_probabilities
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_log_loss": validation_metrics["log_loss"],
            }
        )
        print(
            f"  epoch={epoch:02d} train_loss={np.mean(losses):.4f} "
            f"validation_log_loss={validation_metrics['log_loss']:.4f}"
        )

        current_loss = validation_metrics["log_loss"]
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("GRU không tạo được trạng thái tốt nhất")
    model.load_state_dict(best_state)
    return model, history


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main() -> None:
    seed_everything(SEED)
    data_root = Path(os.environ.get("VALORANT_DATA_ROOT", default_data_root()))
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại {data_root}. "
            "Hãy đặt VALORANT_DATA_ROOT tới thư mục chứa vct_2023 và vct_2024."
        )

    print(f"Data root: {data_root}")
    print(f"Seasons: {SEASONS}")
    frame = load_prepared_data(data_root, SEASONS)
    samples = build_prefix_samples(frame, min_round=MIN_ROUND)
    split = chronological_split(samples, validation_fraction=0.2, seed=SEED)
    print(f"Round rows: {len(frame):,}")
    print(f"Prefix samples: {len(samples):,}")
    print(f"Maps: {split.map_counts}")
    print(
        "Samples:",
        {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
    )

    y_train = np.asarray([row["target"] for row in split.train], dtype=np.int64)
    y_validation = np.asarray(
        [row["target"] for row in split.validation], dtype=np.int64
    )
    y_test = np.asarray([row["target"] for row in split.test], dtype=np.int64)
    baseline_features = build_baseline_features(
        split.train, split.validation, split.test
    )

    baseline_models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            # ``approx`` is stable with the Homebrew XGBoost build on Apple
            # Silicon for this feature matrix; ``hist`` can crash natively.
            tree_method="approx",
            n_jobs=1,
            random_state=SEED,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=180,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            thread_count=1,
            random_seed=SEED,
            allow_writing_files=False,
        ),
    }

    results: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}
    for name, model in baseline_models.items():
        print(f"\nTraining {name}")
        model.fit(baseline_features.train, y_train)
        validation_probabilities = model.predict_proba(baseline_features.validation)[:, 1]
        test_probabilities = model.predict_proba(baseline_features.test)[:, 1]
        test_predictions[name] = test_probabilities
        validation_metrics = evaluate_probabilities(
            y_validation, validation_probabilities
        )
        test_metrics = evaluate_probabilities(y_test, test_probabilities)
        results.append(
            {
                "model": name,
                "validation": validation_metrics,
                "test": test_metrics,
            }
        )
        print(
            f"  validation Log Loss={validation_metrics['log_loss']:.4f}; "
            f"test Log Loss={test_metrics['log_loss']:.4f}"
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTraining GRU on {device}")
    vocabulary = build_sequence_vocabulary(split.train)
    test_loader = make_loader(split.test, vocabulary, shuffle=False)
    for use_attention, name in ((False, "GRU + mean"), (True, "GRU + agent attention")):
        print(f"\nTraining {name}")
        model, history = fit_gru(
            split.train,
            split.validation,
            vocabulary,
            use_agent_attention=use_attention,
            device=device,
        )
        validation_loader = make_loader(split.validation, vocabulary, shuffle=False)
        validation_probabilities = sequence_probabilities(
            model, validation_loader, device
        )
        test_probabilities = sequence_probabilities(model, test_loader, device)
        test_predictions[name] = test_probabilities
        validation_metrics = evaluate_probabilities(
            y_validation, validation_probabilities
        )
        test_metrics = evaluate_probabilities(y_test, test_probabilities)
        results.append(
            {
                "model": name,
                "validation": validation_metrics,
                "test": test_metrics,
                "training_history": history,
            }
        )
        print(
            f"  validation Log Loss={validation_metrics['log_loss']:.4f}; "
            f"test Log Loss={test_metrics['log_loss']:.4f}"
        )

        if not use_attention:
            main_model = model
            main_probabilities = test_probabilities

    checkpoint_metrics = []
    for checkpoint in (6, 12, 18):
        mask = np.asarray([row["round"] == checkpoint for row in split.test])
        if not mask.any():
            continue
        checkpoint_metrics.append(
            {
                "round": checkpoint,
                "samples": int(mask.sum()),
                "models": {
                    name: evaluate_probabilities(y_test[mask], probabilities[mask])
                    for name, probabilities in test_predictions.items()
                },
            }
        )

    calibration_curves = {}
    for name in ("Logistic Regression", "GRU + mean", "GRU + agent attention"):
        expected, observed = calibration_curve(
            y_test,
            test_predictions[name],
            n_bins=10,
            strategy="quantile",
        )
        calibration_curves[name] = {
            "predicted_probability": expected.tolist(),
            "observed_frequency": observed.tolist(),
        }

    # Keep the web payload small while retaining enough trajectories to explore.
    test_by_map: dict[str, list[tuple[dict, float]]] = {}
    for row, probability in zip(split.test, main_probabilities):
        test_by_map.setdefault(row["map_id"], []).append((row, float(probability)))
    selected_map_ids = [
        map_id
        for map_id, _ in sorted(
            test_by_map.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:24]
    ]
    demo_maps = []
    for map_id in selected_map_ids:
        values = sorted(test_by_map[map_id], key=lambda item: item[0]["round"])
        first = values[0][0]
        trajectory = []
        previous = None
        for row, probability in values:
            trajectory.append(
                {
                    "round": row["round"],
                    "score_A": row["score_A"],
                    "score_B": row["score_B"],
                    "probability": probability,
                    "probability_change": None
                    if previous is None
                    else probability - previous,
                    **row["economy"],
                }
            )
            previous = probability
        demo_maps.append(
            {
                "map_id": map_id,
                "map": first["map"],
                "team_A": first["team_A"],
                "team_B": first["team_B"],
                "agents_A": first["agents_A"],
                "agents_B": first["agents_B"],
                "final_score_A": first["final_score_A"],
                "final_score_B": first["final_score_B"],
                "target": first["target"],
                "trajectory": trajectory,
            }
        )

    payload = {
        "project_title": "Post-Match VALORANT Analytics",
        "subtitle": "Map-win probability after each round",
        "data_note": (
            "Interactive post-match simulation based on VCT 2023-2024 data. "
            "It is not a live match prediction system."
        ),
        "source": {
            "name": "Valorant Champion Tour 2021-2023 Data",
            "url": "https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data",
            "seasons_used": list(SEASONS),
        },
        "data_summary": {
            "round_rows": len(frame),
            "prefix_samples": len(samples),
            "maps": split.map_counts,
            "features": ROUND_FEATURES,
            "minimum_round": MIN_ROUND,
        },
        "models": results,
        "checkpoint_metrics": checkpoint_metrics,
        "calibration_curves": calibration_curves,
        "main_model": "GRU + mean",
        "maps": demo_maps,
    }

    output_path = REPO_ROOT / "web" / "data" / "demo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
