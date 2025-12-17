
致队友：
1. gitignore文件设置了：
   不上传原始数据
   不上传冻结后的 .npy
   冻结后的 .npy自己运行 data_prep.py 生成
2. 请先 git clone 仓库，Branch → New Branch
   不要修改 data/ 目录
   只在 src/ 下开发各自负责的文件
   修改前请先 git pull
   提交请写清楚 commit message

3. 数据接口运用示例：X_train = np.load("X_train.npy")


# Cost-Sensitive Ensemble Learning for Imbalanced Bank Marketing

This project is a course final project on **algorithm improvement for imbalanced classification**.
We study how **cost-sensitive learning** and **density-aware sample weighting** can improve
ensemble classifiers on the **UCI Bank Marketing** dataset.

The project is intentionally designed to be **simple, modular, and explainable**, focusing on
algorithmic ideas rather than heavy engineering or deep learning models.

---

## 1. Project Objective

- Address **class imbalance** in real-world classification problems
- Improve **Recall / F1 / G-mean** instead of Accuracy
- Extend classic models taught in class:
  - Logistic Regression
  - Decision Tree / Random Forest
  - Ensemble Methods
- Demonstrate **algorithm-level improvements** using Python (scikit-learn)

---

## 2. Dataset

- **Name**: Bank Marketing
- **Source**: UCI Machine Learning Repository  
- **Link**: https://archive.ics.uci.edu/ml/datasets/bank+marketing
- **Task**: Binary classification (term deposit subscription)
- **Positive class ratio**: ~11%
- **Data size**: 45,211 samples (optionally downsampled to 10,000)

---

## 3. Methods Overview

### 3.1 Baseline
- Random Forest classifier without any imbalance handling

### 3.2 Cost-Sensitive Learning
- Assign higher misclassification cost to the minority (positive) class
- Implemented via `class_weight` or `sample_weight`

### 3.3 Density-Aware Sample Weighting
- Use K-Nearest Neighbors (KNN) to estimate local neighborhood difficulty
- Hard positive samples (surrounded by negatives) receive higher weights
- Combined with cost-sensitive training

---

## 4. Project Structure

```

project/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  └─ bank-full.csv
│  └─ processed/
│     ├─ X_train.npy
│     ├─ X_val.npy
│     ├─ X_test.npy
│     ├─ y_train.npy
│     ├─ y_val.npy
│     └─ y_test.npy
├─ src/
│  ├─ data_prep.py
│  ├─ baseline.py
│  ├─ effect.py
│  ├─ efficiency.py
│  └─ metrics.py
├─ outputs/
│  ├─ models/
│  └─ results_effect.csv
└─ run.py

````

---

## 5. Workflow

1. **Data preparation**
   ```bash
   python src/data_prep.py
````

2. **Run baseline models**

   ```bash
   python src/baseline.py
   ```

3. **Run effect improvement experiments**

   ```bash
   python src/effect.py
   ```

4. *(Optional)* **Run efficiency optimization**

   ```bash
   python src/efficiency.py
   ```

---

## 6. Evaluation Metrics

Because the dataset is imbalanced, we focus on:

* **Recall (Positive class)**
* **F1-score**
* **G-mean**
* Accuracy is reported only as a reference

---

## 7. Key Contributions

* Demonstrated the limitations of standard ensemble models on imbalanced data
* Proposed a **cost-sensitive + density-aware** training strategy
* Achieved consistent improvements in Recall and F1-score
* Maintained model simplicity and interpretability

---

## 8. Team Collaboration Design

* Data interface is **frozen** using preprocessed `.npy` files
* Effect improvement and efficiency optimization are **fully decoupled**
* Team members can work in parallel without blocking each other

---

## 9. Notes

* All experiments use a fixed random seed (`random_state = 42`)
* Only classic machine learning models are used (no deep learning)
* The project prioritizes **clarity and explainability** over complexity

---

## 10. References

* Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning*
* Breiman, L. (2001). *Random Forests*
* He, H., & Garcia, E. (2009). *Learning from Imbalanced Data*

```