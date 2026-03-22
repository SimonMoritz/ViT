# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `sar/`. Model definitions are in `sar/models/`, dataset loaders in `sar/data/`, augmentation utilities in `sar/augmentation.py`, and training entry points in `sar/train/`. Top-level scripts are `main.py` for dataset splitting and `app.py` for the Streamlit viewer. Raw images and labels are stored in `Airport_Dataset_v0_images/` and `Airport_Dataset_v0_labels/`; generated splits are written under `dataset/train/`, `dataset/val/`, and `dataset/test/`.

## Build, Test, and Development Commands

Use `uv` for local setup and execution:

- `uv sync` installs dependencies from `pyproject.toml` and `uv.lock`.
- `uv pip install -e .` installs the package in editable mode.
- `uv run python -m main` creates the train/val/test split.
- `uv run python -m sar.train.train_mae --img_dir Airport_Dataset_v0_images --output_dir checkpoints/mae` starts MAE pretraining.
- `uv run python -m sar.train.train_simclr ...` and `uv run python -m sar.train.train_detection ...` run later training stages.
- `uv run streamlit run app.py` launches the annotation viewer.
- `uv run tensorboard --logdir checkpoints` opens training logs.
- `uv run ruff check .` runs lint checks before review.
- `uv run ty check` runs static type checking.

## Coding Style & Naming Conventions

Target Python 3.13+ and follow PEP 8 with 4-space indentation. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as model components. Keep training scripts thin and put reusable logic under `sar/`. Add type hints to new public functions, dataset interfaces, and nontrivial model helpers so `ty` can check them effectively. Run `uv run ruff check .` and `uv run ty check` before submitting changes, and avoid introducing new warnings without a clear reason.

## Testing Guidelines

There is no dedicated `tests/` suite yet. For changes to data loading, augmentation, or training code, validate with a focused `uv run python -m ...` command on a small sample before larger runs, then run `uv run ruff check .` and `uv run ty check`. When adding tests, place them under `tests/`, prefer `test_*.py` naming, and cover dataset parsing, tensor shapes, and checkpoint loading paths.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit messages such as `clean docs` and `Fix: Remove per-patch normalization for SAR images`. Keep commits focused on one change. Pull requests should summarize the problem, describe the approach, list validation steps, and include screenshots when `app.py` UI behavior changes. Note any dataset or checkpoint assumptions so reviewers can reproduce results.
