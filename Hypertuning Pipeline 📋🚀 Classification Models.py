from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd

def dataset(path):
    df = pd.read_csv(path)
    if 'liver_cancer' not in df.columns:
        raise ValueError(f"target column 'liver_cancer' not found. available columns: {df.columns.tolist()}")
    y = df['liver_cancer'].copy()
    X = df.drop(columns=['liver_cancer'])
    return X, y

def categorical_map(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'Male': 0, 'Female': 1})
    if 'alcohol_consumption' in X.columns:
        X['alcohol_consumption'] = X['alcohol_consumption'].map({'Never': 0, 'Occasional': 1, 'Regular': 2})
    if 'smoking_status' in X.columns:
        X['smoking_status'] = X['smoking_status'].map({'Never': 0, 'Former': 1, 'Current': 2})
    if 'physical_activity_level' in X.columns:
        X['physical_activity_level'] = X['physical_activity_level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    return X

def data_check(X: pd.DataFrame):
    print("Columns:", X.columns.tolist())
    print(X.describe(include='all').transpose())

def data_segmentation(X: pd.DataFrame, y: pd.Series, test_size=0.3, random_state=87):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def remove_outliers_LOF(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    preds = lof.fit_predict(X_train)
    mask = preds != -1  
    X_train_filtered = X_train.loc[mask].reset_index(drop=True)
    y_train_filtered = pd.Series(y_train).loc[mask].reset_index(drop=True)
    return X_train_filtered, y_train_filtered

def create_column_transformer(feature_names):
    desired_power = ['bmi', 'alcohol_consumption', 'liver_function_score', 'alpha_fetoprotein_level', 'physical_activity_level']
    desired_standard = ['age']

    power_cols = [c for c in desired_power if c in feature_names]
    standard_cols = [c for c in desired_standard if c in feature_names]

    transformers = []
    if power_cols:
        transformers.append(('power', PowerTransformer(), power_cols))
    if standard_cols:
        transformers.append(('standard', StandardScaler(), standard_cols))

    if not transformers:
        return ColumnTransformer([('passthrough_all', 'passthrough', list(feature_names))])

    return ColumnTransformer(transformers=transformers, remainder='passthrough')

models = {
    'RandomForest': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(eval_metric='logloss'),
    'LGBMClassifier': LGBMClassifier(verbose=-1 ,enable_default_to_drop=False, feature_name=None),
    'SVC': SVC()}

param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__bootstrap': [True],
        'classifier__max_features': ['sqrt', 'log2', None],},

    'XGBClassifier': {
        'classifier__n_estimators': [100, 300, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [1, 1.0, 1.5],},

    'LGBMClassifier': {
        'classifier__n_estimators': [100, 300, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [31, 64, 127],
        'classifier__max_depth': [-1, 5, 10],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
        'classifier__min_child_samples': [5, 10, 20]},

    'SVC': {
        'classifier__C': [0.1, 0.2, 0.3, 0.8, 1.0, 10.0],
        'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7, base=10)),
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}}

def tune_model(X_train: pd.DataFrame, y_train: pd.Series):
    best_params = {}
    for model_name, model in models.items():
        preprocessor = create_column_transformer(X_train.columns)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        param_grid = param_grids[model_name]

        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_iter=10, cv=5,
            random_state=87, n_jobs=-1, error_score='raise')

        print(f"\nTuning {model_name} ...")
        try:
            random_search.fit(X_train, y_train)
            best_params[model_name] = random_search.best_params_
            print(f"Best params for {model_name}: {best_params[model_name]}")
        except Exception as e:
            print(f"Error while tuning {model_name}: {e}")
    return best_params

if __name__ == "__main__":
    PATH = r'C:\Users\User\Documents\Programming\Dataset\synthetic_liver_cancer_dataset.csv'
    X, y = dataset(PATH)
    X = categorical_map(X)
    data_check(X)

    X_train, X_test, y_train, y_test = data_segmentation(X, y)
    print("Before LOF:", X_train.shape)
    X_train_lof, y_train_lof = remove_outliers_LOF(X_train, y_train)
    print("After LOF:", X_train_lof.shape)

    best_model_params = tune_model(X_train_lof, y_train_lof)
    print("\nAll best params:", best_model_params)

# Email :arshia.khodadad.ir@gmail.compile
# telegram :@arshiacodes