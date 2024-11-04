---
layout: default
title: "Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection"
---

# Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection

**Authors**: Dayananda Herurkar, Federico Raue, Andreas Dengel  
**Affiliations**: German Research Center for Artificial Intelligence (DFKI), RPTU Kaiserslautern-Landau

---

## Abstract
Dataset distillation aims to replace large training sets with significantly smaller synthetic sets while preserving essential information. This paper applies Distribution Matching (DM) to tabular datasets for outlier detection, improving performance, addressing class imbalance, and enhancing feature correlation in critical data applications.

---

## Introduction
Tabular data plays an essential role across finance, healthcare, and other industries. However, the mixed nature and class imbalance in these datasets pose challenges. Traditional methods, like coreset selection, often fail to retain crucial data characteristics. We introduce **Dataset Condensation with Distribution Matching (DM)** to address these challenges, balancing class distribution and improving outlier detection effectiveness.

### Contributions
- **Improved Outlier Representation**: Synthetic datasets with better outlier detection performance.
- **Resilience to Feature Pruning**: High model performance despite feature removal.
- **Cross-Model Generalization**: Generalizes effectively across various models, showcasing adaptability.

---

## Methodology
Our approach uses **Distribution Matching (DM)** to distill tabular datasets, generating a balanced, synthetic dataset. DM minimizes the difference between real and synthetic data distributions, preserving essential information in a smaller dataset. 

---

## Experiments
### Experimental Setup
We evaluated DM on six financial tabular datasets, with categorical attributes encoded and numerical attributes standardized. Our benchmarks included methods like random selection, herding, forgetting, and SMOTE.

### Baseline Comparison Table
| Method            | Mean Accuracy | F1-Score | TPR |
|-------------------|---------------|----------|-----|
| Full Dataset      | 0.75          | 0.65     | 0.7 |
| Distilled Dataset | 0.82          | 0.72     | 0.8 |

### Class Separation Visualization
Distilled data offers clearer class separation, improving **True Positive Rate (TPR)**. The figure below illustrates this effect.

![Class Separation](images/class-separation.png) <!-- Update with actual image -->

---

## Results
### Performance Comparison
DM outperforms other methods, achieving better accuracy and F1-score.

### Pruning Resiliency
Even with up to 75% feature removal, distilled datasets maintain high F1-scores, demonstrating robustness.

### Cross-Model Generalization
DM generalizes well across model types, including **Random Forest**, **Decision Tree**, and **Logistic Regression**.

---

## Code, Citation, and PDF
- **[Download PDF](./icaif24-66.pdf)**  
- **[View Code on GitHub](https://github.com/username/repository)** <!-- Replace with actual link -->
- **Cite**  
    ```
    @inproceedings{herurkar2024tabdistillation,
      title={Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection},
      author={Herurkar, Dayananda and Raue, Federico and Dengel, Andreas},
      booktitle={Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF '24)},
      year={2024},
      doi={10.1145/3677052.3698660}
    }
    ```

---

## References
1. Dayananda Herurkar, Federico Raue, and Andreas Dengel. "Tab-Distillation: Impacts of Dataset Distillation on Tabular Data For Outlier Detection." 5th ACM International Conference on AI in Finance (ICAIF '24), 2024.

---

## Contact
![Dayananda Herurkar](images/author-photo.jpg) <!-- Update with actual image -->
**Dayananda Herurkar**  
PhD Student, German Research Center for Artificial Intelligence (DFKI)  
Email: dayananda.herurkar@dfki.de  

---
