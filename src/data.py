"""Đọc, kiểm tra và biến đổi dữ liệu VCT thành chuỗi round cho mô hình.

Repository công khai chỉ chứa code, không chứa các CSV gốc. Khi chạy ở máy
cá nhân, đường dẫn dữ liệu được lấy từ biến môi trường ``VALORANT_DATA_ROOT``
hoặc từ thư mục ``RawData`` mặc định cạnh repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Năm cột này cùng nhau xác định một map cụ thể trong một loạt trận.
KEY_COLUMNS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]

# Thí nghiệm chính dùng 2023 để train/validation và 2024 làm test cuối.
SEASONS = (2023, 2024)

# Thứ tự cố định của 11 giá trị được đưa vào mỗi bước thời gian của GRU.
ROUND_FEATURES = [
    "rounds_seen",
    "score_A",
    "score_B",
    "round_win_A",
    "loadout_A",
    "credits_A",
    "buy_level_A",
    "round_win_B",
    "loadout_B",
    "credits_B",
    "buy_level_B",
]
# Chuyển nhãn kinh tế dạng chữ thành số có thứ tự để mạng đọc được.
BUY_LEVELS = {
    "Eco: 0-5k": 0,
    "Semi-eco: 5-10k": 1,
    "Semi-buy: 10-20k": 2,
    "Full buy: 20k+": 3,
}


@dataclass(frozen=True)
class DatasetSplit:
    """Ba tập dữ liệu, trong đó không tách một loạt BO3/BO5 sang hai tập."""

    train: list[dict]
    validation: list[dict]
    test: list[dict]
    map_counts: dict[str, int]


def parse_credits(value: object) -> float:
    """Đổi chuỗi tiền như ``3.6k`` thành số 3600 để mô hình xử lý."""

    # Giá trị thiếu được giữ là NaN để bước kiểm tra chất lượng xử lý sau.
    if pd.isna(value):
        return np.nan
    # Chuẩn hóa khoảng trắng, chữ hoa và dấu phẩy trước khi đổi kiểu số.
    text = str(value).strip().lower().replace(",", "")
    try:
        # Hậu tố k nghĩa là nghìn; nếu không có hậu tố thì đọc như số thường.
        return float(text[:-1]) * 1000 if text.endswith("k") else float(text)
    except ValueError:
        # Chuỗi không hợp lệ không được đoán giá trị thay thế.
        return np.nan


def _read_season(
    data_root: Path, season: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Mỗi mùa có ba bảng cần thiết nằm trong thư mục matches.
    season_root = data_root / f"vct_{season}" / "matches"
    required = [
        season_root / "maps_scores.csv",
        season_root / "eco_rounds.csv",
        season_root / "overview.csv",
    ]
    # Báo lỗi sớm để người chạy biết chính xác file nào đang thiếu.
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy các file dữ liệu cần thiết:\n" + "\n".join(missing)
        )
    # low_memory=False giúp pandas suy luận kiểu cột ổn định hơn với CSV lớn.
    return tuple(
        pd.read_csv(path, low_memory=False) for path in required
    )  # type: ignore[return-value]


def _valid_agent_groups(overview: pd.DataFrame) -> pd.DataFrame:
    """Lấy đúng một đội hình đủ năm Agent cho mỗi đội và mỗi map."""

    # Chỉ giữ các cột cần để gom đội hình, tránh mang thống kê sau trận đi xa hơn.
    columns = KEY_COLUMNS + ["Team", "Player", "Agents"]
    # Side=both là tổng hợp cả map; All Maps không đại diện cho một map cụ thể.
    rows = overview.loc[
        overview["Side"].eq("both") & overview["Map"].ne("All Maps"), columns
    ].dropna(subset=["Agents"])
    # Loại bản ghi trùng của cùng người chơi trong cùng map.
    rows = rows.drop_duplicates(KEY_COLUMNS + ["Team", "Player"])
    # Gom danh sách Agent của năm người chơi thành một danh sách đội hình.
    groups = rows.groupby(KEY_COLUMNS + ["Team"], dropna=False)["Agents"].agg(list)
    groups = groups.rename("squad").reset_index()
    groups["player_count"] = groups["squad"].map(len)
    # Chỉ nhận đội hình đủ 5 người; không tự thêm Agent còn thiếu.
    return groups.loc[groups["player_count"].eq(5)].drop(columns="player_count")


def _prepare_season(data_root: Path, season: int) -> pd.DataFrame:
    # Đọc ba nguồn gốc: điểm cuối map, kinh tế từng round và đội hình.
    scores, economy, overview = _read_season(data_root, season)

    # Nếu khóa map bị trùng, merge many-to-many có thể nhân bản dữ liệu âm thầm.
    if scores.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Mùa {season} có khóa map bị trùng trong maps_scores.csv")
    # Map hòa không tạo được nhãn thắng/thua nên bị loại khỏi thí nghiệm.
    scores = scores.loc[scores["Team A Score"].ne(scores["Team B Score"])].copy()
    # target=1 khi đội A thắng cả map, target=0 khi đội B thắng.
    scores["target"] = (scores["Team A Score"] > scores["Team B Score"]).astype("int64")

    # Chỉ mang tên đội, điểm cuối và target sang bảng economy.
    score_columns = KEY_COLUMNS + [
        "Team A",
        "Team B",
        "Team A Score",
        "Team B Score",
        "target",
    ]
    # many_to_one yêu cầu mỗi khóa map bên scores chỉ có một dòng.
    economy = economy.merge(
        scores[score_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    # Gắn từng dòng economy vào Team A hoặc Team B bằng tên đội.
    economy["side"] = np.select(
        [
            economy["Team"].eq(economy["Team A"]),
            economy["Team"].eq(economy["Team B"]),
        ],
        ["A", "B"],
        default="unknown",
    )
    # Giữ lại đúng hai kết quả round hợp lệ là Win và Loss.
    economy = economy.loc[
        economy["side"].ne("unknown") & economy["Outcome"].isin(["Win", "Loss"])
    ].copy()
    # Đổi hai cột tiền từ chuỗi có k sang số thực.
    economy["loadout"] = economy["Loadout Value"].map(parse_credits)
    economy["credits"] = economy["Remaining Credits"].map(parse_credits)
    # Mã hóa mức mua theo bốn nhãn mà nguồn CSV cung cấp.
    economy["buy_level"] = economy["Type"].map(BUY_LEVELS)
    # Win=1 và Loss=0 để cộng dồn thành tỷ số hiện tại.
    economy["round_win"] = economy["Outcome"].eq("Win").astype("int64")

    # Một round hợp lệ phải có đúng hai dòng và hai side khác nhau.
    round_key = KEY_COLUMNS + ["Round Number"]
    quality = economy.groupby(round_key).agg(
        row_count=("side", "size"), side_count=("side", "nunique")
    )
    good_rounds = quality.index[
        quality["row_count"].eq(2) & quality["side_count"].eq(2)
    ]
    economy = economy.set_index(round_key).loc[good_rounds].reset_index()

    # Pivot đưa hai dòng A/B của một round lên cùng một dòng dữ liệu.
    wide = economy.pivot(
        index=round_key,
        columns="side",
        values=["loadout", "credits", "buy_level", "round_win", "Type"],
    )
    wide.columns = [f"{field}_{side}" for field, side in wide.columns]
    wide = wide.reset_index()
    # Gắn lại điểm cuối và target sau khi pivot.
    wide = wide.merge(
        scores[
            KEY_COLUMNS
            + [
                "Team A",
                "Team B",
                "Team A Score",
                "Team B Score",
                "target",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )

    # Ghép đội hình cố định của map vào hai phía A và B.
    squads = _valid_agent_groups(overview)
    for side in ("A", "B"):
        side_squads = squads.rename(
            columns={"Team": f"Team {side}", "squad": f"squad_{side}"}
        )
        wide = wide.merge(
            side_squads[KEY_COLUMNS + [f"Team {side}", f"squad_{side}"]],
            on=KEY_COLUMNS + [f"Team {side}"],
            how="inner",
            validate="many_to_one",
        )

    # Các cột này phải là số trước khi tính tổng cộng dồn và đưa vào model.
    numeric_columns = [
        "round_win_A",
        "round_win_B",
        "buy_level_A",
        "buy_level_B",
        "loadout_A",
        "loadout_B",
        "credits_A",
        "credits_B",
    ]
    for column in numeric_columns:
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    # Sắp xếp để thứ tự thời gian của chuỗi không bị đảo.
    wide = wide.sort_values(KEY_COLUMNS + ["Round Number"]).copy()
    grouped = wide.groupby(KEY_COLUMNS, sort=False)
    # Tính tỷ số tại thời điểm sau từng round, không dùng điểm cuối làm input.
    wide["score_A"] = grouped["round_win_A"].cumsum()
    wide["score_B"] = grouped["round_win_B"].cumsum()
    wide["rounds_seen"] = grouped.cumcount() + 1

    # Chỉ giữ map có round liên tục từ 1..N và tổng thắng khớp điểm cuối.
    # Điều kiện này loại dữ liệu thiếu hoặc ghép nhầm nhưng trông vẫn hợp lệ.
    summary = wide.groupby(KEY_COLUMNS).agg(
        row_count=("Round Number", "size"),
        last_round=("Round Number", "max"),
        score_A=("score_A", "max"),
        score_B=("score_B", "max"),
        final_A=("Team A Score", "first"),
        final_B=("Team B Score", "first"),
    )
    valid_keys = summary.index[
        summary["row_count"].eq(summary["last_round"])
        & summary["score_A"].eq(summary["final_A"])
        & summary["score_B"].eq(summary["final_B"])
    ]
    wide = wide.set_index(KEY_COLUMNS).loc[valid_keys].reset_index()
    wide = wide.loc[wide[numeric_columns].notna().all(axis=1)].copy()
    wide["season"] = season
    # map_id có thêm season để tránh trùng tên khóa giữa các mùa.
    wide["map_id"] = (
        wide[["season"] + KEY_COLUMNS]
        .astype(str)
        .agg(lambda row: "|".join(row.tolist()), axis=1)
    )
    # match_group bỏ tên Map, dùng để giữ các map cùng một loạt trận trong một tập.
    wide["match_group"] = (
        wide[["season", "Tournament", "Stage", "Match Type", "Match Name"]]
        .astype(str)
        .agg(lambda row: "|".join(row.tolist()), axis=1)
    )
    wide["round_win_A"] = wide["round_win_A"].astype("int64")
    wide["round_win_B"] = wide["round_win_B"].astype("int64")
    return wide.reset_index(drop=True)


def load_prepared_data(
    data_root: Path, seasons: Iterable[int] = SEASONS
) -> pd.DataFrame:
    """Đọc các mùa được chọn và nối thành một bảng một dòng mỗi round."""

    # tuple giúp cố định danh sách mùa và phát hiện trường hợp người dùng truyền rỗng.
    selected_seasons = tuple(int(season) for season in seasons)
    if not selected_seasons:
        raise ValueError("Cần chọn ít nhất một mùa giải")
    # Mỗi mùa được làm sạch độc lập trước khi nối để dễ kiểm tra lỗi nguồn.
    frames = [_prepare_season(data_root, season) for season in selected_seasons]
    return pd.concat(frames, ignore_index=True)


def _scaled_round_features(frame: pd.DataFrame) -> np.ndarray:
    """Scale riêng input mạng; bảng dữ liệu gốc vẫn giữ giá trị dễ đọc."""

    # copy để không sửa trực tiếp các cột raw trong DataFrame.
    values = frame[ROUND_FEATURES].to_numpy(dtype="float32").copy()
    # rounds_seen và score tối đa quanh 20 nên chia 20.
    values[:, [0, 1, 2]] /= 20.0
    # loadout và credits có thang nghìn nên chia 10.000.
    values[:, [4, 5, 8, 9]] /= 10000.0
    # buy_level chỉ nằm trong 0..3 nên chia 3.
    values[:, [6, 10]] /= 3.0
    return values


def build_prefix_samples(frame: pd.DataFrame, min_round: int = 3) -> list[dict]:
    """Tạo một mẫu sau mỗi round, bắt đầu từ ``min_round`` và bỏ round cuối."""

    # Không dự đoán quá sớm khi mới có một round; giá trị này là một giả định thí nghiệm.
    if min_round < 1:
        raise ValueError("min_round phải lớn hơn hoặc bằng 1")

    # Mỗi phần tử của list là một prefix độc lập về mặt input, nhưng cùng map có thể liên quan.
    samples: list[dict] = []
    for map_id, map_frame in frame.groupby("map_id", sort=False):
        # Đảm bảo history luôn đi từ round 1 đến round đang xét.
        map_frame = map_frame.sort_values("Round Number").reset_index(drop=True)
        states = _scaled_round_features(map_frame)
        first = map_frame.iloc[0]
        # Bỏ round cuối vì sau round đó map đã kết thúc và target đã lộ hoàn toàn.
        for end in range(min_round, len(map_frame)):
            current = map_frame.iloc[end - 1]
            samples.append(
                {
                    "map_id": str(map_id),
                    "match_group": str(first["match_group"]),
                    "season": int(first["season"]),
                    "round": int(current["Round Number"]),
                    "history": states[:end].tolist(),
                    "agents_A": list(first["squad_A"]),
                    "agents_B": list(first["squad_B"]),
                    "map": str(first["Map"]),
                    "team_A": str(first["Team A"]),
                    "team_B": str(first["Team B"]),
                    "score_A": int(current["score_A"]),
                    "score_B": int(current["score_B"]),
                    "target": int(first["target"]),
                    "final_score_A": int(first["Team A Score"]),
                    "final_score_B": int(first["Team B Score"]),
                    "economy": {
                        "loadout_A": float(current["loadout_A"]),
                        "loadout_B": float(current["loadout_B"]),
                        "credits_A": float(current["credits_A"]),
                        "credits_B": float(current["credits_B"]),
                        "buy_A": str(current["Type_A"]),
                        "buy_B": str(current["Type_B"]),
                    },
                }
            )
    return samples


def chronological_split(
    samples: list[dict], validation_fraction: float = 0.2, seed: int = 42
) -> DatasetSplit:
    """Tách 2023 thành train/validation và giữ nguyên 2024 làm test cuối.

    Đơn vị chia là loạt trận, nên các map trong cùng BO3/BO5 không bị rải sang
    hai tập. Năm 2024 không được dùng trong quá trình chọn mô hình.
    """

    # Tỷ lệ validation phải nằm trong khoảng mở (0, 1).
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction phải nằm giữa 0 và 1")

    # Gom prefix theo match_group để mọi map của một series đi cùng nhau.
    by_group: dict[str, list[dict]] = {}
    for sample in samples:
        by_group.setdefault(str(sample["match_group"]), []).append(sample)

    # Random generator có seed giúp lần chia có thể tái lập.
    rng = np.random.default_rng(seed)
    groups_2023 = sorted(
        group for group, rows in by_group.items() if int(rows[0]["season"]) == 2023
    )
    # Chỉ xáo trộn các series 2023, không xáo trộn map trong series.
    rng.shuffle(groups_2023)
    validation_count = max(1, round(len(groups_2023) * validation_fraction))
    validation_groups = set(groups_2023[:validation_count])
    train_groups = set(groups_2023[validation_count:])

    # Tạo ba danh sách prefix sau khi đã xác định nhóm map.
    train = [row for row in samples if row["match_group"] in train_groups]
    validation = [row for row in samples if row["match_group"] in validation_groups]
    test = [row for row in samples if int(row["season"]) == 2024]
    map_counts = {
        "train": len({row["map_id"] for row in train}),
        "validation": len({row["map_id"] for row in validation}),
        "test": len({row["map_id"] for row in test}),
    }
    return DatasetSplit(train, validation, test, map_counts)
