# Tab-Distillation (ICAIF 2024)

Implementation of **Tab-Distillation: Impacts of Dataset Distillation on Tabular Data for Outlier Detection** (ICAIF ’24).  
The code applies **Dataset Condensation with Distribution Matching (DM)** to **tabular outlier-detection** datasets, producing a **small, class-balanced synthetic dataset** (e.g., 10/50/100 samples per class) that can be used to train a classifier.

> Paper: *Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection* (ICAIF ’24).

---

## 1) Setup

### Requirements
- Python 3.9+ recommended
- PyTorch, NumPy, Pandas, scikit-learn

Install:
```bash
pip install -r requirements.txt
```

---

## 2) Data layout

This repo expects **pre-split CSVs** per dataset and run:

```
data/
  <DATASET_NAME>/
    run1/
      <DATASET_NAME>_train_set.csv
      <DATASET_NAME>_test_set.csv
```

Example:
```
data/Credit_Default/run1/Credit_Default_train_set.csv
data/Credit_Default/run1/Credit_Default_test_set.csv
```

The CSVs must contain the columns used in the original experiments (see `utils_Tab_DM.get_tabular_dataset`).

---

## 3) Run distillation

The main script is `main_Tab_DM.py`.

### Example (Credit Default)
```bash
python main_Tab_DM.py  --dataset Credit_Default  --model MLP  --init real  --num_exp 1  --num_eval 1
```

### Key arguments
- `--dataset`: one of `{Credit_Default, Credit_Fraud, Census_Income, Adult_Data, Bank_Marketing, KDD_Cup, IEEE_Fraud, Covertype}`
- `--model`: distillation model, e.g. `--spc 10 50 100 500 1000`
- `--init`: `real` (default) or `noise`
- `--num_exp`: the number of experiments
- `--num_eval`: the number of evaluating randomly initialized models

---

## 4) Outputs

All outputs go under `--save_path` (default: `results/`):

- `results/synthetic/`
  - distilled synthetic sets saved as CSV
- `results/tables/`
  - per-repeat results CSV
  - mean/std aggregate across repeats

Synthetic dataset CSV format:
- feature columns: `Col_0 ... Col_{D-1}`
- label column: `label`

---

## 5) Method sketch

For each class, DM matches the **mean embedding** of real samples and synthetic samples (embedding = penultimate layer of the MLP).  
Synthetic samples are optimized directly with SGD/Adam, producing a small, class-balanced set. fileciteturn1file0

---

## 6) Citation

If you use this code, please cite:

```
@inproceedings{herurkar2024tabdistillation,
  title     = {Tab-Distillation: Impacts of Dataset Distillation on Tabular Data for Outlier Detection},
  author    = {Herurkar, Dayananda and Raue, Federico and Dengel, Andreas},
  booktitle = {5th ACM International Conference on AI in Finance (ICAIF '24)},
  year      = {2024},
  doi       = {10.1145/3677052.3698660}
}
```

---

## 7) Notes / known limitations

- This repo assumes you already have the CSV splits prepared. It does **not** download datasets.
- Binary outlier detection assumes label `1` is the outlier class when computing PR-AUC / ROC-AUC. fileciteturn1file16
- Determinism is best-effort (`--seed`), but exact reproducibility can still vary across CUDA/cuDNN versions.

---


