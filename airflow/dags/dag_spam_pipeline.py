from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "alaa",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

def check_silver_data():
    path = "/opt/airflow/data/silver/messages_clean.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier Silver introuvable : {path}")
    print("✅ Contrôle qualité OK : les données Silver sont prêtes.")

with DAG(
    dag_id="il15_spam_batch_orchestration",
    default_args=default_args,
    description="Pipeline quotidien : Contrôle Silver -> Agrégats Gold -> Réentraînement ML",
    schedule_interval="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["bigdata", "il15", "spam"],
) as dag:

    t1_check_silver = PythonOperator(
        task_id="check_silver_quality",
        python_callable=check_silver_data,
    )

    t2_compute_gold_and_ml = BashOperator(
        task_id="run_gold_aggregations_and_ml",
        bash_command="python /opt/airflow/spark/batch_gold_and_ml.py",
    )

    # Ordre d'exécution strict
    t1_check_silver >> t2_compute_gold_and_ml