# 效率优化模块 (Efficiency Module) - 使用指南

## 概述

本文档详细说明如何运行效率优化模块，完成"三层对比"实验，展示模型在**变得更小、更快、更稳**方面的优化效果。

---

## 前置条件

### 1. 环境准备
确保已完成以下步骤：

```bash
# 在项目根目录执行
cd /Users/apple/Desktop/数据挖掘2/bank-marketing-imbalanced

# 安装依赖包（如果尚未安装）
pip install -r requirements.txt
```

### 2. 数据准备
确保已生成预处理数据：

```bash
python src/data_prep.py
```

这将在 `data/processed/` 下生成以下文件：
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`

### 3. Effect优化完成
确保已训练过effect优化的模型。建议先运行：

```bash
python src/effect.py
```

这会生成 `outputs/results_effect.csv`，作为参考基线。

---

## 快速开始

### 方法1：直接运行（推荐）

在项目根目录执行：

```bash
python -m src.efficiency.run_efficiency
```

或者：

```bash
cd src/efficiency
python run_efficiency.py
```

### 方法2：从Python脚本调用

```python
from src.efficiency.run_efficiency import main

main()
```

---

## 运行流程详解

### 执行步骤

运行脚本后，会依次执行以下三层对比实验：

#### **Layer 1: 基线模型 (Baseline)**
- **目的**：建立性能基准
- **方法**：标准随机森林，不做任何优化
- **操作**：
  1. 训练200棵树的RF模型
  2. 在验证集上调优阈值（以F1为目标）
  3. 在测试集上评估性能
- **输出指标**：Recall、F1、G-mean、AUC、预测时间

#### **Layer 2: Effect优化**
- **目的**：通过算法改进提升性能（不考虑模型大小）
- **方法**：成本敏感学习 + 密度感知样本加权
- **操作**：
  1. 计算样本权重（基于KNN密度估计）
  2. 使用 `class_weight={0:1, 1:4}` 和 `sample_weight` 训练RF
  3. 在验证集上调优阈值
  4. 在测试集上评估性能
- **输出指标**：同Layer 1，并计算相对Baseline的改进率

#### **Layer 3: Effect + Efficiency优化**
- **目的**：在保持性能的前提下，实现模型压缩和加速
- **方法**：K-Means树聚类 + 加权投票集成
- **核心流程**：

  ```
  1. 从Layer 2的200棵树中提取预测矩阵
     ↓
  2. K-Means聚类（默认10个簇）
     ↓
  3. 从每个簇选出F1最高的树
     ↓
  4. 基于F1分数计算树权重 (∑F1_i / ∑F1)
     ↓
  5. 加权投票推理: ∑(Tree_i(x) × Weight_i)
  ```

- **输出指标**：同上，并计算：
  - 模型大小减少比例
  - 预测速度提升倍数

---

## 输出文件

运行完成后，会在 `outputs/` 目录生成以下文件：

### 1. **results_efficiency_comparison.csv**

完整的三层对比结果表，包含：

| 列名 | 说明 |
|------|------|
| method | 方法名称 (Baseline / Effect Optimized / Effect + Efficiency) |
| trees_count | 模型使用的树数量 |
| train_time_s | 训练耗时（秒） |
| pred_time_s | 预测耗时（秒） |
| threshold | 调优后的决策阈值 |
| recall | 正类召回率 |
| f1 | F1-Score |
| gmean | G-mean评分 |
| auc | ROC-AUC评分 |

**示例输出：**
```
method,trees_count,train_time_s,pred_time_s,threshold,recall,f1,gmean,auc
Baseline,200,45.32,0.0234,0.48,0.7850,0.6234,0.6543,0.8765
Effect Optimized,200,47.89,0.0231,0.42,0.8123,0.6521,0.6789,0.8912
Effect + Efficiency,10,47.89,0.0041,0.45,0.8034,0.6412,0.6732,0.8876
```

### 2. **efficiency_comparison.png**

四子图对比可视化：

```
┌─────────────────────────────────────────────────┐
│  三层对比: Baseline vs Effect vs Effect+Efficiency│
├──────────────────┬──────────────────────────────┤
│  ROC Curves      │  Recall & F1-Score Comparison│
│                  │                              │
│  (AUC对比)       │  (性能指标对比)              │
├──────────────────┼──────────────────────────────┤
│ Prediction Time  │ Model Size vs Performance   │
│ Comparison (ms)  │ Trade-off                   │
│                  │                              │
│  (速度对比)      │  (大小与性能权衡)            │
└──────────────────┴──────────────────────────────┘
```

四个子图详解：
1. **左上 ROC Curves**：展示三种方法的ROC曲线，反映分类性能
2. **右上 Recall & F1-Score**：直观对比性能指标
3. **左下 Prediction Time**：展示预测时间减少情况
4. **右下 Model Size vs Performance**：显示压缩率与性能的权衡

---

## 关键参数说明

在 `EfficiencyConfig` 中可调整：

```python
@dataclass
class EfficiencyConfig:
    # 数据路径
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    
    # 基础配置
    random_state: int = 42              # 随机种子
    n_estimators: int = 200             # RF树数量
    max_depth: int | None = None        # 树最大深度
    
    # Effect优化参数
    pos_class_weight: float = 4.0       # 正类权重
    knn_k: int = 5                      # 密度估计K值
    density_alpha: float = 5.0          # 密度加权强度
    
    # Efficiency优化参数
    n_clusters: int = 10                # K-Means聚类数 ⭐
    use_threshold_tuning: bool = True   # 是否调优阈值
    threshold_objective: str = "f1"     # 调优目标 ("f1" 或 "recall")
    t_min: float = 0.05                 # 阈值搜索范围下界
    t_max: float = 0.95                 # 阈值搜索范围上界
    t_step: float = 0.01                # 阈值搜索步长
```

### 调整建议

- **n_clusters**：默认10，可根据树数量调整
  - 树数200 → 推荐 n_clusters = 8-15
  - 树数100 → 推荐 n_clusters = 5-10
  - 目标：保留30%-50%的树，保持性能

- **pos_class_weight**：默认4.0
  - 越大 → 更强调正类 → Recall更高，可能降低准确率
  - 建议范围：2.0-6.0

- **density_alpha**：默认5.0
  - 越大 → 密度加权效果更强
  - 建议范围：2.5-7.5

---

## 常见问题

### Q1: 运行报错 "ModuleNotFoundError"
**解决**：确保在项目根目录运行，且已执行 `pip install -r requirements.txt`

### Q2: 预测时间没有显著下降
**解决**：
- 增加 `n_clusters` 值（会保留更少的树）
- 检查选中树的数量：`len(selected_trees)` 应远小于 `n_estimators`

### Q3: Effect+Efficiency的F1突然大幅下降
**原因**：聚类选择的树可能不均衡
**解决**：
- 减小 `n_clusters`（增加每个簇的树数）
- 尝试不同的 `random_state` 值

### Q4: 如何提高聚类后模型的稳定性？
**建议**：
1. 增加 `n_clusters` 中的 `n_init` 参数（默认10）
2. 在多个随机种子下运行，取平均结果
3. 使用更小的 `n_clusters` 保留更多树

---

## 性能指标解释

| 指标 | 公式 | 含义 |
|------|------|------|
| **Recall** | TP/(TP+FN) | 正类覆盖率，越高越好 |
| **F1-Score** | 2×P×R/(P+R) | 精确率与召回率的调和均值 |
| **G-mean** | √(TPR×TNR) | 两类的平衡性评分 |
| **AUC** | 不同阈值下TPR的积分 | 分类性能综合评估 |
| **Speedup** | t_baseline / t_pruned | 预测加速倍数 |

---

## 实验对比解释

### 为什么要三层对比？

1. **Baseline** → **Effect Optimized**：验证算法改进的效果
2. **Effect Optimized** → **Effect + Efficiency**：展示在保持性能的前提下的压缩效果
3. **Overall**：完整展现从基线到最优解的进步

### 预期结果

在imbalanced数据上的典型表现：

```
Baseline:
  - Recall: ~0.70-0.75
  - F1: ~0.55-0.60
  - Pred time: ~23ms

Effect Optimized:
  - Recall: ~0.78-0.82 (+10-15%)
  - F1: ~0.60-0.65 (+8-12%)
  - Pred time: ~23ms (无明显变化)

Effect + Efficiency:
  - Recall: ~0.77-0.81 (可能略低，但在可接受范围)
  - F1: ~0.59-0.64 (略低，但仍优于Baseline)
  - Pred time: ~4-5ms (加速5-6倍)
  - Trees: 10-15 (保留5-8%)
```

---

## 进阶用法

### 自定义TreePruner

```python
from src.efficiency.models import TreePruner, predict_ensemble

# 创建并拟合剪枝器
pruner = TreePruner(n_clusters=15, random_state=42)
pruner.fit(rf_model, X_val, y_val)

# 获取选中的树和权重
selected_trees = pruner.get_selected_trees()
tree_weights = pruner.get_tree_weights()

# 进行预测
y_pred_proba = predict_ensemble(
    rf_model, 
    X_test, 
    selected_trees, 
    tree_weights, 
    use_proba=True
)

# 使用阈值生成二分类预测
y_pred = (y_pred_proba >= 0.5).astype(int)
```

### 批量测试不同的n_clusters

```python
for n_clusters in [5, 10, 15, 20]:
    cfg = EfficiencyConfig(n_clusters=n_clusters)
    # ... 运行实验并记录结果
```

---

## 输出到Excel进行进一步分析

```python
import pandas as pd

# 读取结果
df = pd.read_csv('outputs/results_efficiency_comparison.csv')

# 导出到Excel
df.to_excel('efficiency_results.xlsx', index=False)

# 计算改进率
for col in ['recall', 'f1', 'gmean', 'auc']:
    baseline_val = df.loc[df['method'] == 'Baseline', col].values[0]
    df[f'{col}_improvement_%'] = (df[col] - baseline_val) / baseline_val * 100
```

---

## 完整示例脚本

```bash
#!/bin/bash
# run_efficiency_pipeline.sh

cd /Users/apple/Desktop/数据挖掘2/bank-marketing-imbalanced

echo "=== 步骤1: 准备数据 ==="
python src/data_prep.py

echo "=== 步骤2: 运行baseline和effect优化 ==="
python src/effect.py

echo "=== 步骤3: 运行efficiency优化 (三层对比) ==="
python -m src.efficiency.run_efficiency

echo "=== 步骤4: 生成报告 ==="
python << 'PYTHON_EOF'
import pandas as pd
df = pd.read_csv('outputs/results_efficiency_comparison.csv')
print("\n" + "="*60)
print("最终结果总结")
print("="*60)
print(df.to_string(index=False))
print("\n可视化已保存至: outputs/efficiency_comparison.png")
PYTHON_EOF

echo "完成!"
```

---

## 下一步

完成此模块后，建议：

1. ✅ 查看生成的 `efficiency_comparison.png` 可视化
2. ✅ 分析 `results_efficiency_comparison.csv` 中的数据
3. ✅ 参考 `PROJECT_SUMMARY.md` 了解整体项目设计和结果分析
4. ✅ 根据需要调整参数并重新运行实验

---

## 技术支持

如遇问题，请检查：
1. 是否正确加载了数据（检查 `data/processed/` 下的文件）
2. 是否安装了所有依赖包
3. Python版本是否 >= 3.8
4. 是否有足够的内存（建议8GB+）

