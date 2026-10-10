"""Feature construction and models used by the reproducible experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BaselineFeatures:
    """Tabular matrices and names, built with training-set vocabularies."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_names: list[str]


def _fit_vocab(samples: Iterable[dict], field: str) -> dict[str, int]:
    values = sorted({str(sample[field]) for sample in samples})
    return {value: index + 1 for index, value in enumerate(values)}


def _fit_agent_vocab(samples: Iterable[dict]) -> dict[str, int]:
    values = {
        str(agent)
        for sample in samples
        for side in ("agents_A", "agents_B")
        for agent in sample[side]
    }
    return {value: index + 1 for index, value in enumerate(sorted(values))}


def _one_hot(value: str, vocabulary: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    index = vocabulary.get(value)
    if index is not None:
        vector[index - 1] = 1.0
    return vector


def _multi_hot(values: Iterable[str], vocabulary: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for value in values:
        index = vocabulary.get(str(value))
        if index is not None:
            vector[index - 1] = 1.0
    return vector


def _baseline_row(
    sample: dict, map_vocab: dict[str, int], agent_vocab: dict[str, int]
) -> np.ndarray:
    history = np.asarray(sample["history"], dtype=np.float32)
    # The tabular baselines see the current state, the average state so far,
    # and the change from the first observed round. They do not see order.
    numeric = np.concatenate(
        [history[-1], history.mean(axis=0), history[-1] - history[0]]
    )
    return np.concatenate(
        [
            numeric,
            _one_hot(str(sample["map"]), map_vocab),
            _multi_hot(sample["agents_A"], agent_vocab),
            _multi_hot(sample["agents_B"], agent_vocab),
        ]
    ).astype(np.float32)


def build_baseline_features(
    train: list[dict], validation: list[dict], test: list[dict]
) -> BaselineFeatures:
    """Build leakage-safe tabular features using only train vocabularies."""

    map_vocab = _fit_vocab(train, "map")
    agent_vocab = _fit_agent_vocab(train)
    names = [
        f"current_{index}"
        for index in range(11)
    ] + [
        f"mean_{index}"
        for index in range(11)
    ] + [
        f"change_{index}"
        for index in range(11)
    ]
    names += [f"map={value}" for value in sorted(map_vocab)]
    names += [f"agents_A={value}" for value in sorted(agent_vocab)]
    names += [f"agents_B={value}" for value in sorted(agent_vocab)]

    def matrix(rows: list[dict]) -> np.ndarray:
        return np.vstack([_baseline_row(row, map_vocab, agent_vocab) for row in rows])

    return BaselineFeatures(
        train=matrix(train),
        validation=matrix(validation),
        test=matrix(test),
        feature_names=names,
    )


def evaluate_probabilities(y_true: Iterable[int], probabilities: Iterable[float]) -> dict:
    """Return probability metrics; lower Log Loss/Brier are better."""

    y = np.asarray(list(y_true), dtype=np.int64)
    p = np.clip(np.asarray(list(probabilities), dtype=np.float64), 1e-6, 1 - 1e-6)
    result = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    )
    return result


@dataclass(frozen=True)
class SequenceVocabulary:
    maps: dict[str, int]
    agents: dict[str, int]


def build_sequence_vocabulary(train: list[dict]) -> SequenceVocabulary:
    """Fit categorical vocabularies on train only; index 0 means unknown."""

    return SequenceVocabulary(
        maps=_fit_vocab(train, "map"),
        agents=_fit_agent_vocab(train),
    )


class PrefixDataset(Dataset):
    """PyTorch dataset for variable-length round histories."""

    def __init__(self, rows: list[dict], vocabulary: SequenceVocabulary):
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        agents_a = [
            self.vocabulary.agents.get(str(value), 0)
            for value in sorted(row["agents_A"])
        ]
        agents_b = [
            self.vocabulary.agents.get(str(value), 0)
            for value in sorted(row["agents_B"])
        ]
        return {
            "history": torch.tensor(row["history"], dtype=torch.float32),
            "length": len(row["history"]),
            "map_id": self.vocabulary.maps.get(str(row["map"]), 0),
            "agents_A": torch.tensor(agents_a, dtype=torch.long),
            "agents_B": torch.tensor(agents_b, dtype=torch.long),
            "target": float(row["target"]),
        }


def collate_prefix_batch(rows: list[dict]) -> tuple[torch.Tensor, ...]:
    histories = pad_sequence(
        [row["history"] for row in rows], batch_first=True, padding_value=0.0
    )
    lengths = torch.tensor([row["length"] for row in rows], dtype=torch.long)
    maps = torch.tensor([row["map_id"] for row in rows], dtype=torch.long)
    agents_a = torch.stack([row["agents_A"] for row in rows])
    agents_b = torch.stack([row["agents_B"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    return histories, lengths, maps, agents_a, agents_b, targets


class WinProbabilityGRU(nn.Module):
    """GRU over round history plus map and agent-composition embeddings."""

    def __init__(
        self,
        map_count: int,
        agent_count: int,
        round_feature_count: int,
        use_agent_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_agent_attention = use_agent_attention
        self.map_embedding = nn.Embedding(map_count + 1, 8)
        self.agent_embedding = nn.Embedding(agent_count + 1, 12)
        self.agent_score = nn.Linear(12, 1)
        self.gru = nn.GRU(
            input_size=round_feature_count,
            hidden_size=48,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(48 + 8 + 12 + 12, 48),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 1),
        )

    def _team_vector(self, agents: torch.Tensor) -> torch.Tensor:
        vectors = self.agent_embedding(agents)
        if not self.use_agent_attention:
            return vectors.mean(dim=1)
        weights = torch.softmax(self.agent_score(vectors).squeeze(-1), dim=1)
        return (vectors * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        maps: torch.Tensor,
        agents_a: torch.Tensor,
        agents_b: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            histories,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        joined = torch.cat(
            [
                hidden[-1],
                self.map_embedding(maps),
                self._team_vector(agents_a),
                self._team_vector(agents_b),
            ],
            dim=1,
        )
        return self.head(joined).squeeze(1)
