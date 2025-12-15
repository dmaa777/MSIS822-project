# Detection of AI-Generated Arabic Text (MSIS-822)

This repository implements the MSIS-822 graduation project: **Detection of AI-Generated Arabic Text: A Data Mining Approach** using the Hugging Face dataset **KFUPM-JRCAI/arabic-generated-abstracts**.

## Repository layout (per course requirements)

- `data/`
  - `raw/` raw dataset exports (not committed)
  - `processed/` cleaned + feature tables (not committed unless small)
  - `external/` external resources (e.g., lexicons)
- `notebooks/` exploratory and main workflow notebooks
- `src/` reusable Python modules (function bodies copied verbatim from the main notebook)
- `scripts/` runnable entry points (download/build/train/evaluate)
- `models/` saved models
- `reports/` figures and presentations
- `docs/` supporting documentation

## Quick start

1. Create environment and install dependencies:
   - `pip install -r requirements.txt`

2. Download dataset:
   - `python scripts/download_data.py`

3. Build features (scaffold; the canonical pipeline is in notebooks):
   - `python scripts/build_features.py`

4. Train baseline model (scaffold):
   - `python scripts/train_models.py`

## Notebooks

- `01_Phase1_Data_Acquisition.ipynb`
- `02_Phase2_Preprocessing_EDA.ipynb`
- `04_Phase3_4_Modeling_Features.ipynb` (main implementation)
- `_archive/phase_3_and_4_updated_original.ipynb` (unaltered backup)

## Notes

- The module functions in `src/` are copied from `phase 3 and 4 updated.ipynb` without editing the function bodies.
- XGBoost is optional; if not installed, use scikit-learn alternatives.
