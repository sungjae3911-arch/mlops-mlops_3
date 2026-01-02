import os

# 강제로 환경 변수 설정 (.env로 가져오지 못함)
os.environ['DB_HOST'] = '172.17.0.1'
os.environ['DB_PORT'] = '3307'
os.environ['DB_USER'] = 'user'
os.environ['DB_PASSWORD'] = 'password'

import pandas as pd
from sqlalchemy import create_engine, text

def write_db(data: pd.DataFrame, db_name, table_name):   
    try:
        engine = create_engine(url=(
            f"mysql+mysqldb://"
            f"{os.environ.get('DB_USER')}:"
            f"{os.environ.get('DB_PASSWORD')}@"
            f"{os.environ.get('DB_HOST')}:"
            f"{os.environ.get('DB_PORT')}/"
            f"{db_name}"))

        connect = engine.connect()
        data.to_sql(table_name, connect, if_exists="append")

    except Error as e:
        print(f"MySQL 에러 발생 (코드 {e.errno}): {e.msg}")
    finally:          
        if 'connection' in locals() and connect.is_connected():
            connect.close()


def get_engine(db_name):
    engine = create_engine(url=(
        f"mysql+mysqldb://"
        f"{os.environ.get('DB_USER')}:"
        f"{os.environ.get('DB_PASSWORD')}@"
        f"{os.environ.get('DB_HOST')}:"
        f"{os.environ.get('DB_PORT')}/"
        f"{db_name}"))
    return engine

def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    connect = engine.connect()
    result = connect.execute(
        statement=text(
            f"select recommend_content_id from {table_name} "
            f"order by `index` desc limit :k"
        ),
        parameters={"table_name": table_name, "k": k}
    )
    connect.close()
    contents = [data[0] for data in result]
    return contents