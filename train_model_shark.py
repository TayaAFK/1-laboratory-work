from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
from datetime import datetime


def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['Age']), df['Age']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("./df_clear.csv")
    X, Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ["l1", "l2", "elasticnet"],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True],
    }
    
    mlflow.set_experiment("linear model sharks")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv = 3, n_jobs = 4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_age_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
        y_val_real = power_trans.inverse_transform(y_val)
        (rmse, mae, r2) = eval_metrics(y_val_real, y_age_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        penalty = best.penalty
        eta0 = best.eta0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("eta0", eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        with open("lr_sharks.pkl", "wb") as file:
            joblib.dump(best, file)
        with open("/home/taisiya/airflow/metrics.txt", "w") as f:
            f.write("Metrics for Shark Attack Model\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"R2 Score: {r2}\n")
            f.write(f"Date: {datetime.now()}\n")
            
