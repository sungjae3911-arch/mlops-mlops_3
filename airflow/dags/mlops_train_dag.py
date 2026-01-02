from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "mlops",
}

with DAG(
    dag_id="mlops_train_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="""
        echo "=== PREPROCESS START ==="
        python /opt/mlops/data-prepare/main.py
        echo "=== PREPROCESS END ==="
        """
    )

    train = BashOperator(
        task_id="train",
        bash_command="""
        echo "=== TRAIN START ==="
        docker run --rm \
          -e WANDB_API_KEY=${WANDB_API_KEY} \
          -v /home/ubuntu/mlops-mlops_3:/opt/mlops \
          mlops-train:latest \
          train movie_predictor --num_epochs 3
        echo "=== TRAIN END ==="
        """
    )

    preprocess >> train

















