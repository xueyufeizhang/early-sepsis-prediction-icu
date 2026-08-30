"""Classic ML on static features (brief Section 6; Module 3).

Day 6-7. LR, SVM, RandomForest, XGBoost, MLP. Each with grid/random-search
hyperparameter tuning over patient-grouped CV folds; persist the best config.

Planned functions:
  - get_search_space(model_name)   tuning grid per model
  - build_estimator(model_name)    sklearn/xgboost estimator factory
  - tune_and_fit(model_name, ...)   grouped-CV search -> best fitted model
"""
