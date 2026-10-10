# VALORANT VCT Map-Win Analytics

An esports analytics project for post-match VALORANT decision support. It uses VCT 2023-2024 match, round economy, and agent composition data to estimate map-win probability after each round with GRU-based deep learning models, benchmarked against machine learning baselines and presented through an interactive web demo.

## Project question

After a round has ended, can the available score, economy, and agent-composition information estimate the probability that Team A will win the whole map?

The output is a probability for post-match analysis. It is not a live betting tool, a tactical recommendation engine, or a causal estimate of an agent's effect.

## Repository layout

```text
src/                 Data preparation and model code
scripts/             Reproducible training and demo-data generation
notebooks/           Guided analysis notebook
web/                 Static GitHub Pages demo
web/data/            Small derived demo data only
```

Raw CSV files are intentionally excluded from this public repository. The Kaggle dataset should be downloaded separately and stored locally.

## Data location

By default, the training script looks for:

```text
../RawData/Valorant-Champion-Tour-2021-2026-Data
```

relative to this repository. Override it with `VALORANT_DATA_ROOT` when necessary:

```bash
export VALORANT_DATA_ROOT="/absolute/path/to/Valorant-Champion-Tour-2021-2026-Data"
```

Only seasons 2023 and 2024 are used in the main experiment:

- 2023: training and validation data
- 2024: final test data

The split keeps the complete 2024 season untouched until final testing. Within 2023, complete match series are divided into training and validation groups. The chronological design avoids using future-season maps for model fitting. A map with incomplete economy or agent data is excluded rather than filled with made-up values.

The local download contains more seasons. They are not included in the main result because the 2025 tournament labels do not consistently match the economy table, which would make a large but unreliable merge. Later seasons can be added after tournament-name normalization.

## Run the analysis

Use the Homebrew Python installation configured for the project:

```bash
/opt/homebrew/opt/python@3.13/libexec/bin/python3 scripts/train.py
```

The script writes derived demo data to `web/data/demo.json`. It does not copy raw data into this repository.

## View the web demo

The demo is a static site and can be served locally with:

```bash
python3 -m http.server 8000 --directory web
```

Open <http://localhost:8000>. The same `web/` directory can be published with GitHub Pages.

## Data source

Ryan Luong, *Valorant Champion Tour 2021-2023 Data*, Kaggle. The local download used by this project also contains the 2024 folder; the report documents the exact seasons and cleaning rules used in the experiment.

