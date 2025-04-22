# 💓 Heart Stroke Prediction using Machine Learning
*A Hybrid Feature Selection Approach*

**Author:** Sri Harsha Vardhan Chikkala  
**Affiliation:** SRM University - AP, India

## 📘 Abstract
This project proposes a machine learning-based approach to predict the risk of heart stroke using a hybrid feature selection technique. By integrating **low variance filtering** and **Principal Component Analysis (PCA)**, we refine the dataset to its most critical features. This improves model performance, reduces computational complexity, and provides valuable insights into key health indicators. We experiment with five classifiers — **Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost** — and demonstrate superior results, especially with **Random Forest** achieving up to **97.52% accuracy**.

## 🧠 Problem Statement
Heart stroke is a major cause of mortality worldwide. Early detection through predictive modeling can significantly reduce fatality rates. However, models suffer when trained on imbalanced and high-dimensional datasets. This project addresses:
* **Class imbalance** using **SMOTE**
* **Feature redundancy** using a **hybrid selection approach**
* **Optimal model performance** with comparative evaluations

## 📊 Dataset
* **Source**: Kaggle - Stroke Prediction Dataset
* **Records**: ~43,000 samples post-processing
* **Features**: 11 input features + 1 target (stroke)

Key Features:
* `age`, `gender`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
* `ever_married`, `work_type`, `Residence_type`, `smoking_status`

## 🛠️ Methodology
1. **Data Preprocessing**
   * Handled nulls using **Decision Tree-based imputation**
   * **Z-score normalization**
   * Encoded categorical variables

2. **Balancing Data**
   * Applied **SMOTE** to handle heavy imbalance (~700 stroke vs 43,000 non-stroke samples)

3. **Hybrid Feature Selection**
   * **Low Variance Filter** (threshold: σ² < 0.01)
   * **Principal Component Analysis (PCA)** (tested with 5, 7, 9 components)

4. **Model Training**
   * Models: `KNN`, `Decision Tree`, `Logistic Regression`, `Random Forest`, `XGBoost`
   * Evaluated across four feature sets: full set, PCA-reduced sets (5, 7, 9 components)

## 📈 Results

| Model | Accuracy (Full) | Accuracy (PCA-9) | Best Model |
|-------|----------------|-----------------|------------|
| KNN | 92.36% | 92.36% | |
| Decision Tree | 95.76% | 93.96% | |
| Logistic Reg. | 83.43% | 83.17% | |
| Random Forest | **97.52%** | 96.68% | ✅ Best Overall |
| XGBoost | 96.02% | 94.33% | |

* **Random Forest** consistently achieved the highest performance across all PCA components.
* **PCA** improved KNN and LR performance at higher dimensions (e.g., 9 components).
* Ensemble methods were robust to feature reduction.

## 🧪 Evaluation Metrics
* **Accuracy**, **Precision**, **Recall**, **F1-Score**
* Analysis was conducted before and after feature selection to assess model stability.

## ⏱️ Time Complexity Summary

| Phase | Complexity (Before) | Complexity (After) |
|-------|-------------------|-------------------|
| SMOTE | O(n²) | O(n²) |
| Model Training | O(nmlog(n)) | O(nplog(n)) |
| Feature Selection | – | O(np² + p³) (PCA) |

## 🚀 Getting Started

### 🔧 Prerequisites
Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### ▶️ Run the Notebook
1. Clone the repository:

```bash
git clone https://github.com/Harsha-chikkala/Heart-Stroke-Prediction-using-Machine-Learning-A-hybrid-feature-selection-approach.git
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook Heart_Stroke_Prediction_using_Machine_Learning_A_hybrid_feature_selection_Approach.ipynb
```

3. Run all cells to see preprocessing, training, evaluation, and results.

## 📌 Key Takeaways
* Hybrid feature selection significantly enhances performance in models sensitive to irrelevant features.
* Ensemble models (e.g., RF, XGBoost) show resilience to dimensional reduction.
* SMOTE proves essential in addressing imbalanced classification tasks in healthcare datasets.

