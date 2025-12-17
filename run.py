from src.data_prep import load_data
from src.baseline import run_baseline
# from src.effect import ImprovedRF
# from src.efficiency import EnsembleOptimizer
from src.metrics import evaluate

# 1. Load Data
X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir="data/processed/")

# 2. Run Baseline (Member C)
base_model, base_pred = run_baseline(X_train, y_train, X_test)
evaluate(y_test, base_pred, name="Baseline")

# # 3. Run Effect Improvement (Member A)
# improver = ImprovedRF()
# full_forest = improver.fit(X_train, y_train)

# # 4. Run Efficiency Improvement (Member B)
# optimizer = EnsembleOptimizer(full_forest)
# optimizer.prune(X_val, y_val)
# final_pred = optimizer.predict(X_test)

# # 5. Final Eval
# evaluate(y_test, final_pred, name="Final Method")