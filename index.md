
---
layout: default
title: "Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection"
---

# Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection

**Authors**: Dayananda Herurkar, Federico Raue, Andreas Dengel  
**Affiliations**: German Research Center for Artificial Intelligence (DFKI), RPTU Kaiserslautern-Landau

---

## Abstract
Dataset distillation aims to replace large training sets with significantly smaller synthetic sets while preserving essential information. In this paper, we apply Distribution Matching (DM) to tabular datasets for outlier detection. This approach enhances outlier detection performance, reduces class imbalance, and ensures higher correlation among features, making it effective for industrial applications relying on tabular data.

---

## Introduction
Tabular data, commonly used in finance, healthcare, and manufacturing, faces unique challenges due to its mixed types of features and frequent class imbalances. Traditional methods for dataset reduction, such as coreset selection, often struggle to retain essential data characteristics. Our study introduces a novel approach, utilizing **Dataset Condensation with Distribution Matching (DM)**, to distill tabular datasets for outlier detection. This method effectively addresses class imbalance, improves model performance, and reduces computational demands.

### Contributions
- **Enhanced Outlier Representation**: Synthetic datasets show improved representation of outliers, leading to more robust outlier detection.
- **Resilience Against Feature Pruning**: Distilled datasets demonstrate robustness even with feature removal, maintaining high classification performance.
- **Generalization**: Our methodology generalizes well across multiple model types, showcasing its adaptability.

---

## Methodology
Our approach leverages **Distribution Matching (DM)** for dataset condensation, where a synthetic dataset is created by matching the feature distributions of the real dataset. This process balances the class distribution, captures essential information from the original data, and allows for a smaller, efficient dataset for training. DM uses a neural network to minimize the difference between real and synthetic data distributions, yielding a highly representative, smaller dataset.

---

## Results

### Performance Comparison
Models trained on distilled datasets outperformed traditional coreset methods, demonstrating significant improvement in **Mean Accuracy** and **F1-Score**. The distilled synthetic data achieved high outlier detection rates while using only a fraction of the original data.

![Performance Comparison Table](#) <!-- Placeholder link for performance table -->

### Class Separation Between Inliers and Outliers
The synthetic dataset exhibits clearer class separation between inliers and outliers, resulting in superior **True Positive Rate (TPR)** compared to full datasets. This improvement is visualized through density plots and decision boundaries.

![Class Separation Figure](#) <!-- Placeholder link for figure -->

### Pruning Resiliency
The distilled synthetic datasets showed strong resiliency against feature pruning. Even with up to 75% of features pruned, the models maintained high **F1-Scores**, while models trained on full datasets showed drastic drops in performance with minimal pruning.

![Pruning Resiliency Figure](#) <!-- Placeholder link for figure -->

### Cross-Model Generalization
Our distilled dataset generalizes effectively across different models, including **Random Forest**, **Decision Tree**, and **Logistic Regression**, maintaining robust outlier detection performance regardless of the model type.

![Cross-Model Generalization Table](#) <!-- Placeholder link for table -->

---

## Code
You can find the code and related resources for this project on [GitHub](https://github.com/username/repository) <!-- Replace with actual link -->

## Citation
If you find this work helpful in your research, please consider citing it as:

```bibtex
@inproceedings{herurkar2024tabdistillation,
  title={Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection},
  author={Herurkar, Dayananda and Raue, Federico and Dengel, Andreas},
  booktitle={Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF '24)},
  year={2024},
  doi={10.1145/3677052.3698660}
}
```

## Download
[Download the full paper (PDF)](./icaif24-66.pdf)

---
