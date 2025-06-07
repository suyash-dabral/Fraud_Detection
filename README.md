# 🛡️ Credit Card Fraud Detection using XGBoost & Bayesian Optimization

This project builds a machine learning pipeline to detect fraudulent credit card transactions. It uses:

- 🧠 XGBoost for classification  
- 🔍 Bayesian hyperparameter optimization (Hyperopt)  
- 🧪 Custom evaluation metric (recall at a minimum precision)  
- ⚖️ Class imbalance handling  
- ⏱️ Early stopping and threshold tuning  

---

## 📊 Dataset

Dataset used: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- Total Transactions: 284,807  
- Fraudulent: 492 (≈0.17%)  
- Features: 30 anonymized PCA features, `Time`, `Amount`, and `Class` (target)

---

## ⚙️ Features

✅ Uses `XGBClassifier` from XGBoost  
✅ Hyperparameter tuning via Bayesian optimization with `hyperopt`  
✅ Custom scoring metric: **maximum recall where precision ≥ 0.05**  
✅ Threshold tuning for real-world fraud detection constraints  
✅ Handles class imbalance using `scale_pos_weight`

---

## 🧠 How It Works
- Load and split the dataset (80% training, 20% testing)

- Further split training for early stopping

- Tune XGBoost hyperparameters using hyperopt.fmin()

- Train XGBoost with the best hyperparameters

- Find the optimal classification threshold where precision ≥ 0.05

- Predict frauds using the thresholded probabilities

- Report precision and recall

