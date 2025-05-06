import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

def train_models(df, top_features):
    X = df[top_features]
    y = df['Fraud label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring='f1', n_jobs=-1)
    rf_search.fit(X_train_smote, y_train_smote)
    best_rf = rf_search.best_estimator_

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_iter=5, scoring='f1', cv=3, n_jobs=-1)
    xgb_search.fit(X_train_smote, y_train_smote)
    best_xgb = xgb_search.best_estimator_

    return best_rf, best_xgb, X_test, y_test
