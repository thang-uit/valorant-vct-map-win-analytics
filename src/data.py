"""Prepare round-prefix samples from the local VCT CSV files.

The public repository contains the preparation code, not the downloaded CSV
files. The CSV files are located through ``VALORANT_DATA_ROOT`` or through
the default ``RawData`` folder documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]
SEASONS = (2023, 2024)
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
BUY_LEVELS = {
    "Eco: 0-5k": 0,
    "Semi-eco: 5-10k": 1,
    "Semi-buy: 10-20k": 2,
    "Full buy: 20k+": 3,
}


@dataclass(frozen=True)
class DatasetSplit:
    """Samples split without putting one match series in two partitions."""

    train: list[dict]
    validation: list[dict]
    test: list[dict]
    map_counts: dict[str, int]


def parse_credits(value: object) -> float:
    """Convert values such as ``3.6k`` into numeric credits."""

    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    try:
        return float(text[:-1]) * 1000 if text.endswith("k") else float(text)
    except ValueError:
        return np.nan


def _read_season(
    data_root: Path, season: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    season_root = data_root / f"vct_{season}" / "matches"
    required = [
        season_root / "maps_scores.csv",
        season_root / "eco_rounds.csv",
        season_root / "overview.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy các file dữ liệu cần thiết:\n" + "\n".join(missing)
        )
    return tuple(
        pd.read_csv(path, low_memory=False) for path in required
    )  # type: ignore[return-value]


def _valid_agent_groups(overview: pd.DataFrame) -> pd.DataFrame:
    """Return one complete five-agent squad per team and map."""

    columns = KEY_COLUMNS + ["Team", "Player", "Agents"]
    rows = overview.loc[
        overview["Side"].eq("both") & overview["Map"].ne("All Maps"), columns
    ].dropna(subset=["Agents"])
    rows = rows.drop_duplicates(KEY_COLUMNS + ["Team", "Player"])
    groups = rows.groupby(KEY_COLUMNS + ["Team"], dropna=False)["Agents"].agg(list)
    groups = groups.rename("squad").reset_index()
    groups["player_count"] = groups["squad"].map(len)
    return groups.loc[groups["player_count"].eq(5)].drop(columns="player_count")


def _prepare_season(data_root: Path, season: int) -> pd.DataFrame:
    scores, economy, overview = _read_season(data_root, season)

    # Prevent a silent many-to-many merge if the source changes later.
    if scores.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Mùa {season} có khóa map bị trùng trong maps_scores.csv")
    scores = scores.loc[
        scores["Team A Score"].ne(scores["Team B Score"])
    ].copy()
    scores["target"] = (
        scores["Team A Score"] > scores["Team B Score"]
    ).astype("int64")

    score_columns = KEY_COLUMNS + [
        "Team A",
        "Team B",
        "Team A Score",
        "Team B Score",
        "target",
    ]
    economy = economy.merge(
        scores[score_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    economy["side"] = np.select(
        [
            economy["Team"].eq(economy["Team A"]),
            economy["Team"].eq(economy["Team B"]),
        ],
        ["A", "B"],
        default="unknown",
    )
    economy = economy.loc[
        economy["side"].ne("unknown") & economy["Outcome"].isin(["Win", "Loss"])
    ].copy()
    economy["loadout"] = economy["Loadout Value"].map(parse_credits)
    economy["credits"] = economy["Remaining Credits"].map(parse_credits)
    economy["buy_level"] = economy["Type"].map(BUY_LEVELS)
    economy["round_win"] = economy["Outcome"].eq("Win").astype("int64")

    round_key = KEY_COLUMNS + ["Round Number"]
    quality = economy.groupby(round_key).agg(
        row_count=("side", "size"), side_count=("side", "nunique")
    )
    good_rounds = quality.index[
        quality["row_count"].eq(2) & quality["side_count"].eq(2)
    ]
    economy = economy.set_index(round_key).loc[good_rounds].reset_index()

    # Put the two team rows of a round on one row.
    wide = economy.pivot(
        index=round_key,
        columns="side",
        values=["loadout", "credits", "buy_level", "round_win", "Type"],
    )
    wide.columns = [f"{field}_{side}" for field, side in wide.columns]
    wide = wide.reset_index()
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

    wide = wide.sort_values(KEY_COLUMNS + ["Round Number"]).copy()
    grouped = wide.groupby(KEY_COLUMNS, sort=False)
    wide["score_A"] = grouped["round_win_A"].cumsum()
    wide["score_B"] = grouped["round_win_B"].cumsum()
    wide["rounds_seen"] = grouped.cumcount() + 1

    # Keep only maps whose round numbers are 1..N and whose wins equal the
    # final score. This rejects partial or incorrectly joined maps.
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
    wide["map_id"] = wide[["season"] + KEY_COLUMNS].astype(str).agg(
        lambda row: "|".join(row.tolist()), axis=1
    )
    wide["match_group"] = wide[
        ["season", "Tournament", "Stage", "Match Type", "Match Name"]
    ].astype(str).agg(lambda row: "|".join(row.tolist()), axis=1)
    wide["round_win_A"] = wide["round_win_A"].astype("int64")
    wide["round_win_B"] = wide["round_win_B"].astype("int64")
    return wide.reset_index(drop=True)


def load_prepared_data(
    data_root: Path, seasons: Iterable[int] = SEASONS
) -> pd.DataFrame:
    """Load and clean the selected seasons into one round-level table."""

    selected_seasons = tuple(int(season) for season in seasons)
    if not selected_seasons:
        raise ValueError("Cần chọn ít nhất một mùa giải")
    frames = [_prepare_season(data_root, season) for season in selected_seasons]
    return pd.concat(frames, ignore_index=True)


def _scaled_round_features(frame: pd.DataFrame) -> np.ndarray:
    """Scale only for neural-network input; raw values remain in the table."""

    values = frame[ROUND_FEATURES].to_numpy(dtype="float32").copy()
    values[:, [0, 1, 2]] /= 20.0
    values[:, [4, 5, 8, 9]] /= 10000.0
    values[:, [6, 10]] /= 3.0
    return values


def build_prefix_samples(frame: pd.DataFrame, min_round: int = 3) -> list[dict]:
    """Create one sample after each non-final round of a map."""

    if min_round < 1:
        raise ValueError("min_round phải lớn hơn hoặc bằng 1")

    samples: list[dict] = []
    for map_id, map_frame in frame.groupby("map_id", sort=False):
        map_frame = map_frame.sort_values("Round Number").reset_index(drop=True)
        states = _scaled_round_features(map_frame)
        first = map_frame.iloc[0]
        # The final round is excluded because it ends the map.
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
    """Split 2023 into train/validation and reserve all 2024 for testing.

    Match series are the grouping unit, so maps from one BO3/BO5 cannot be
    scattered across train and validation. The 2024 season is untouched until
    the final test step.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction phải nằm giữa 0 và 1")

    by_group: dict[str, list[dict]] = {}
    for sample in samples:
        by_group.setdefault(str(sample["match_group"]), []).append(sample)

    rng = np.random.default_rng(seed)
    groups_2023 = sorted(
        group for group, rows in by_group.items() if int(rows[0]["season"]) == 2023
    )
    rng.shuffle(groups_2023)
    validation_count = max(1, round(len(groups_2023) * validation_fraction))
    validation_groups = set(groups_2023[:validation_count])
    train_groups = set(groups_2023[validation_count:])

    train = [row for row in samples if row["match_group"] in train_groups]
    validation = [row for row in samples if row["match_group"] in validation_groups]
    test = [row for row in samples if int(row["season"]) == 2024]
    map_counts = {
        "train": len({row["map_id"] for row in train}),
        "validation": len({row["map_id"] for row in validation}),
        "test": len({row["map_id"] for row in test}),
    }
    return DatasetSplit(train, validation, test, map_counts)
