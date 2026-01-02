import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
import wandb
from dotenv import load_dotenv
import numpy as np

from src.dataset.watch_log import get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.model.movie_predictor import MoviePredictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.utils.enums import ModelTypes
from src.dataset.preprocessing import TMDBPreProcessor
from src.dataset.crawler import TMDBCrawler
from src.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from src.postprocess.postprocess import write_db


load_dotenv()
init_seed()


def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")


def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"  # movie-predictor-000

    return runs[0].name


def run_train(model_name, num_epochs=10, batch_size=64):
    ModelTypes.validation(model_name)

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)
    
    project_name = model_name.replace("_", "-")
    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based movie recommend model",
        tags=["content-based", "movie", "recommend"],
        config=locals(),
    )
    
    # 데이터셋 및 DataLoader 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, batch_size=64, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, batch_size=64, shuffle=False)
    test_loader = SimpleDataLoader(test_dataset.features, test_dataset.labels, batch_size=64, shuffle=False)

    # 모델 초기화
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_dim": 64
    }
    model_class = ModelTypes[model_name.upper()].value
    model = MoviePredictor(**model_params)

    # 학습 루프
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader)
        val_loss, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val-Train Loss : {val_loss-train_loss:.4f}")
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/Valid": val_loss})

    wandb.finish()
   
   # 테스트
    test_loss, predictions = evaluate(model, test_loader)
    print(f"Test Loss : {test_loss:.4f}")
    print([train_dataset.decode_content_id(idx) for idx in predictions])
   
    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        loss=train_loss,
        scaler=train_dataset.scaler,
        label_encoder=train_dataset.label_encoder,
    ) 

def check_api_key():
    tmdb_api_key = os.getenv('TMDB_API_KEY')
    #print("tmdb_api_key ->", tmdb_api_key)

    if not tmdb_api_key:
        raise ValueError(" API_KEY not found! Please set it in .env file")
    print("API_KEY loaded successfully")

def make_result_folder_crawler():
    # 현재 실행 중인 파이썬 파일의 절대 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '..')
    parent_dir = os.path.abspath(parent_dir)  # 절대 경로로 정규화

    result_path = os.path.join(parent_dir, "data-prepare/result")
    # print(f"*존재 여부: {os.path.exists(result_path)}")

    os.makedirs(result_path, exist_ok=True)
    # print(f"parent_dir  :{parent_dir}")
    # print(f"result_path  :{result_path}")

def run_popular_movie_crawler():
    print("Starting Crawling...")
    tmdb_crawler = TMDBCrawler()
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=1)
    tmdb_crawler.save_movies_to_json_file(result, "./data-prepare/result", "popular")

    print("Starting Preprocessing...")
    tmdb_preprocessor = TMDBPreProcessor(result)
    tmdb_preprocessor.run()
    tmdb_preprocessor.save("watch_log")
    print(f"Results saved to watch_log.csv")

def run_preprocess():

    check_api_key()
    make_result_folder_crawler()
    run_popular_movie_crawler()

def run_inference(data=None, batch_size=64):
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)

    if data is None:
        data = []

    data = np.array(data)

    recommend = inference(model, scaler, label_encoder, data, batch_size)
    print(recommend)

    write_db(recommend_to_df(recommend), "moviedb", "recommend")
    print(f"=== mysql database write를 종료합니다.===")


if __name__ == '__main__':
    fire.Fire({
        "preprocess": run_preprocess,
        "train":run_train,
        "inference": run_inference,
    })
