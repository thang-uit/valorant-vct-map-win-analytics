"""Tạo đặc trưng và định nghĩa các mô hình của thí nghiệm."""

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
    """Ma trận đầu vào cho baseline, với từ điển chỉ học từ tập train."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_names: list[str]


def _fit_vocab(samples: Iterable[dict], field: str) -> dict[str, int]:
    # Lấy các giá trị phân loại đã thấy trong train và sắp xếp để mã hóa ổn định.
    values = sorted({str(sample[field]) for sample in samples})
    # Chừa index 0 cho giá trị chưa từng thấy ở train.
    return {value: index + 1 for index, value in enumerate(values)}


def _fit_agent_vocab(samples: Iterable[dict]) -> dict[str, int]:
    # Gom mọi Agent của cả hai đội trong tập train thành một vocabulary chung.
    values = {
        str(agent)
        for sample in samples
        for side in ("agents_A", "agents_B")
        for agent in sample[side]
    }
    # Mã bắt đầu từ 1 vì 0 đại diện cho unknown trong Embedding.
    return {value: index + 1 for index, value in enumerate(sorted(values))}


def _one_hot(value: str, vocabulary: dict[str, int]) -> np.ndarray:
    # One-hot tạo vector chỉ có một vị trí bằng 1 cho tên map.
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    index = vocabulary.get(value)
    # Nếu map không có trong vocabulary thì giữ toàn bộ vector bằng 0.
    if index is not None:
        vector[index - 1] = 1.0
    return vector


def _multi_hot(values: Iterable[str], vocabulary: dict[str, int]) -> np.ndarray:
    # Multi-hot cho phép năm Agent cùng bật nhiều vị trí trong một vector.
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for value in values:
        index = vocabulary.get(str(value))
        if index is not None:
            vector[index - 1] = 1.0
    return vector


def _baseline_row(
    sample: dict, map_vocab: dict[str, int], agent_vocab: dict[str, int]
) -> np.ndarray:
    # Đổi lịch sử round thành ma trận số có dạng số round đã quan sát x 11.
    history = np.asarray(sample["history"], dtype=np.float32)
    # Baseline chỉ thấy trạng thái cuối, trung bình lịch sử và độ thay đổi từ đầu.
    # Vì đã tổng hợp như vậy nên nó không còn biết thứ tự từng round.
    numeric = np.concatenate(
        [history[-1], history.mean(axis=0), history[-1] - history[0]]
    )
    # Nối numeric với map one-hot và Agent multi-hot thành một vector phẳng.
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
    """Tạo đặc trưng bảng mà không lấy vocabulary từ validation/test."""

    # Fit vocabulary từ train để giá trị ở test chưa từng thấy không làm lộ dữ liệu.
    map_vocab = _fit_vocab(train, "map")
    agent_vocab = _fit_agent_vocab(train)
    # Tạo tên cột để khi cần có thể truy nguyên từng phần của vector đầu vào.
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

    # Dùng cùng vocabulary cho cả ba tập để số chiều ma trận luôn giống nhau.
    def matrix(rows: list[dict]) -> np.ndarray:
        return np.vstack([_baseline_row(row, map_vocab, agent_vocab) for row in rows])

    return BaselineFeatures(
        train=matrix(train),
        validation=matrix(validation),
        test=matrix(test),
        feature_names=names,
    )


def evaluate_probabilities(y_true: Iterable[int], probabilities: Iterable[float]) -> dict:
    """Tính metric cho xác suất; Log Loss và Brier thấp hơn là tốt hơn."""

    # Đưa iterator về mảng số nguyên để sklearn xử lý thống nhất.
    y = np.asarray(list(y_true), dtype=np.int64)
    # Chặn xác suất quá sát 0 hoặc 1 để log không bị vô hạn.
    p = np.clip(np.asarray(list(probabilities), dtype=np.float64), 1e-6, 1 - 1e-6)
    # Accuracy và F1 dùng ngưỡng 0.5; đây là metric phụ vì mục tiêu chính là xác suất.
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
    """Fit vocabulary phân loại từ train; index 0 nghĩa là giá trị chưa biết."""

    return SequenceVocabulary(
        maps=_fit_vocab(train, "map"),
        agents=_fit_agent_vocab(train),
    )


class PrefixDataset(Dataset):
    """Dataset PyTorch chuyển prefix có độ dài khác nhau thành tensor."""

    def __init__(self, rows: list[dict], vocabulary: SequenceVocabulary):
        # rows là các prefix đã tạo từ data.py; vocabulary là mã đã fit từ train.
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        # Lấy một prefix theo vị trí mà DataLoader yêu cầu.
        row = self.rows[index]
        # Agent được sắp xếp để thứ tự trong CSV không ảnh hưởng đến vector đội hình.
        agents_a = [
            self.vocabulary.agents.get(str(value), 0)
            for value in sorted(row["agents_A"])
        ]
        agents_b = [
            self.vocabulary.agents.get(str(value), 0)
            for value in sorted(row["agents_B"])
        ]
        # Trả về tensor số và các trường phụ để model biết độ dài history thật.
        return {
            "history": torch.tensor(row["history"], dtype=torch.float32),
            "length": len(row["history"]),
            "map_id": self.vocabulary.maps.get(str(row["map"]), 0),
            "agents_A": torch.tensor(agents_a, dtype=torch.long),
            "agents_B": torch.tensor(agents_b, dtype=torch.long),
            "target": float(row["target"]),
        }


def collate_prefix_batch(rows: list[dict]) -> tuple[torch.Tensor, ...]:
    # Pad các history ngắn thành cùng chiều dài trong một batch.
    histories = pad_sequence(
        [row["history"] for row in rows], batch_first=True, padding_value=0.0
    )
    # lengths giữ số bước thật để GRU bỏ qua phần padding.
    lengths = torch.tensor([row["length"] for row in rows], dtype=torch.long)
    # Map và Agent đã là index số, chỉ cần gom thành tensor batch.
    maps = torch.tensor([row["map_id"] for row in rows], dtype=torch.long)
    agents_a = torch.stack([row["agents_A"] for row in rows])
    agents_b = torch.stack([row["agents_B"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    return histories, lengths, maps, agents_a, agents_b, targets


class WinProbabilityGRU(nn.Module):
    """GRU đọc chuỗi round rồi ghép embedding map và đội hình hai đội."""

    def __init__(
        self,
        map_count: int,
        agent_count: int,
        round_feature_count: int,
        use_agent_attention: bool = True,
    ) -> None:
        # Khởi tạo lớp cha để PyTorch đăng ký các tham số của module con.
        super().__init__()
        # Chọn nhánh mean hay attention khi gộp năm Agent.
        self.use_agent_attention = use_agent_attention
        # +1 dành cho index 0 của giá trị unknown.
        self.map_embedding = nn.Embedding(map_count + 1, 8)
        self.agent_embedding = nn.Embedding(agent_count + 1, 12)
        # Lớp này chấm điểm từng Agent nếu dùng attention.
        self.agent_score = nn.Linear(12, 1)
        # GRU đọc mỗi bước gồm 11 đặc trưng; hidden state có 48 chiều.
        self.gru = nn.GRU(
            input_size=round_feature_count,
            hidden_size=48,
            batch_first=True,
        )
        # Prediction head biến vector ghép thành một logit duy nhất.
        self.head = nn.Sequential(
            nn.Linear(48 + 8 + 12 + 12, 48),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 1),
        )

    def _team_vector(self, agents: torch.Tensor) -> torch.Tensor:
        # Lookup embedding cho năm index Agent của một đội.
        vectors = self.agent_embedding(agents)
        if not self.use_agent_attention:
            # Baseline mean coi năm Agent có vai trò như nhau.
            return vectors.mean(dim=1)
        # Chấm điểm rồi chuẩn hóa trọng số về tổng bằng 1.
        weights = torch.softmax(self.agent_score(vectors).squeeze(-1), dim=1)
        # Tổng có trọng số là vector đại diện cho toàn đội.
        return (vectors * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        maps: torch.Tensor,
        agents_a: torch.Tensor,
        agents_b: torch.Tensor,
    ) -> torch.Tensor:
        # Pack giúp GRU không học từ các số 0 được thêm vào khi padding.
        packed = pack_padded_sequence(
            histories,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # hidden[-1] là trạng thái cuối của lớp GRU duy nhất.
        _, hidden = self.gru(packed)
        # Ghép lịch sử, map và hai vector đội hình theo chiều feature.
        joined = torch.cat(
            [
                hidden[-1],
                self.map_embedding(maps),
                self._team_vector(agents_a),
                self._team_vector(agents_b),
            ],
            dim=1,
        )
        # Bỏ chiều 1 thừa để trả về một logit cho mỗi mẫu trong batch.
        return self.head(joined).squeeze(1)
