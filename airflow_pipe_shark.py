import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import shutil
import kagglehub
from train_model_shark import train

def download_data():
    path = kagglehub.dataset_download("teajay/global-shark-attacks")
    csv_source = os.path.join(path, "attacks.csv")
    local_target = "attacks.csv"
    shutil.copyfile(csv_source, local_target)
    
    df = pd.read_csv(local_target, encoding='latin-1')
    print("df shape: ", df.shape)
    return local_target

def clear_data():
    df = pd.read_csv("attacks.csv", encoding='latin-1')
    cat_columns = ['Type', 'Country', 'Sex ', 'Activity']
    target_column = 'Age'
    df = df[cat_columns + [target_column]].dropna()
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df = df[(df.Age > 0) & (df.Age < 100)]
    df = df.reset_index(drop=True)
    
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_columns] = ordinal.fit_transform(df[cat_columns].astype(str))
    
    df.to_csv('df_clear.csv', index=False)
    return True

dag_sharks = DAG(
    dag_id="shark_attack_train_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=None,
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data, 
    task_id="download_sharks", 
    dag=dag_sharks
)

clear_task = PythonOperator(
    python_callable=clear_data, 
    task_id="clear_sharks", 
    dag=dag_sharks
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_sharks", 
    dag=dag_sharks
)
download_task >> clear_task >> train_task
