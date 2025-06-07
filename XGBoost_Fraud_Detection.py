import pandas as pd
import numpy as np
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, precision_recall_curve, recall_score, precision_score
from functools import partial
import hyperopt

# Suppress DeprecationWarnings (XGBoost + SciPy issue)
warnings.filterwarnings('ignore', category=DeprecationWarning)

MIN_PRECISION = 0.05  # minimum precision requirement

def conditional_recall_score(y_true, pred_proba, min_prec=MIN_PRECISION):
    pr, rc, _ = precision_recall_curve(y_true, pred_proba[:, 1])
    return np.max(rc[pr >= min_prec]) if np.any(pr >= min_prec) else 0

def objective(params, X, y, X_early_stop, y_early_stop, scorer, n_folds=10):
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    imbalance_ratio = neg_count / pos_count

    xgb_clf = XGBClassifier(**params, scale_pos_weight=imbalance_ratio,
                            n_estimators=2000, n_jobs=1, use_label_encoder=False)

    xgb_fit_params = {
        'early_stopping_rounds': 50,
        'eval_metric': ['logloss'],
        'eval_set': [(X_early_stop, y_early_stop)],
        'verbose': False
    }

    cv_score = np.mean(cross_val_score(xgb_clf, X, y, cv=n_folds,
                                       fit_params=xgb_fit_params, n_jobs=-1,
                                       scoring=scorer))

    return {'loss': -cv_score, 'status': hyperopt.STATUS_OK, 'params': params}

def tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter):
    scorer = make_scorer(conditional_recall_score, needs_proba=True)

    obj = partial(objective, scorer=scorer, X=X_train, y=y_train,
                  X_early_stop=X_early_stop, y_early_stop=y_early_stop)

    trials = hyperopt.Trials()

    hyperopt.fmin(fn=obj, space=param_space, algo=hyperopt.tpe.suggest,
                  max_evals=n_iter, trials=trials)

    return trials.best_trial['result']['params']

def optimal_threshold(estimator, X, y, n_folds=10, min_prec=MIN_PRECISION, fit_params=None):
    cv_pred_prob = cross_val_predict(estimator, X, y, method='predict_proba',
                                     cv=n_folds, fit_params=fit_params, n_jobs=-1)[:, 1]
    pr, _, threshold = precision_recall_curve(y, cv_pred_prob)
    pr = pr[:-1]  # drop the last element
    threshold = threshold[pr >= min_prec]
    return min(threshold) if len(threshold) > 0 else 0.5

def thresholded_predict(X, estimator, threshold):
    return np.array([1 if p >= threshold else 0 for p in estimator.predict_proba(X)[:, 1]])

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1)

    # Early stopping split (10% of total data = 1/8 of 80%)
    X_train, X_early_stop, y_train, y_early_stop = train_test_split(
        X_train, y_train, stratify=y_train, test_size=1/8, random_state=1)

    # Define hyperparameter search space
    param_space = {
        'max_depth': hyperopt.hp.choice('max_depth', np.arange(3, 10, dtype=int)),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.2),
        'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hyperopt.hp.uniform('gamma', 0, 5),
        'reg_alpha': hyperopt.hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0, 1)
    }

    # Run tuning
    best_params = tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter=25)

    # Train final model
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    final_model = XGBClassifier(**best_params, scale_pos_weight=imbalance_ratio,
                                n_estimators=2000, n_jobs=-1, use_label_encoder=False)

    fit_params = {
        'early_stopping_rounds': 50,
        'eval_metric': 'logloss',
        'eval_set': [(X_early_stop, y_early_stop)],
        'verbose': False
    }

    final_model.fit(X_train, y_train, **fit_params)

    # Find optimal threshold
    best_threshold = optimal_threshold(final_model, X_train, y_train, fit_params=fit_params)

    # Predict on test set
    y_pred = thresholded_predict(X_test, final_model, best_threshold)

    # Evaluate
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
