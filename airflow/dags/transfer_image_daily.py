from datetime import datetime, timedelta
from textwrap import dedent
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pendulum

from utils import connect_sftp

KST = pendulum.timezone("Asia/Seoul")

def send_dir():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    date = str(str(yesterday).split(" ")[0])
    path_local = f'/opt/data/raw/{date}'
    print(f"date: {date}")
    sftp = connect_sftp.connSftp()
    sftp.upload(path_server='/cctv/data/raw', path_local=path_local, date=date)

default_args = {
    'owner': 'bgyoon',
    'depends_on_past': False,
    'email': ['byungguan.yoon@telstar-hommel.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='upload_dir',
    default_args=default_args,
    description='Daily transfer image',
    # schedule_interval='@hourly',
    schedule_interval='0 0 * * *',
    start_date=datetime(2022, 1, 4, 0, 0, tzinfo=KST),
    catchup=False,
    tags=['v0.0.1']
) as dag:
    t1 = PythonOperator(
        task_id='transfer',
        python_callable=send_dir
    )


