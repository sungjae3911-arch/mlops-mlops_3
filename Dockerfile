# 베이스 이미지 (파이썬 3.11 사용)
FROM python:3.11-bookworm

# 메타 데이터
LABEL maintainer="MLOps_Group_3 Project"
LABEL version="0.1.0"
LABEL description="Set the same environment for MLOps projects"

# 작업 디렉토리로 설정
WORKDIR /opt/mlops

# 1단계 - 의존성 설치 (캐싱 활용)
#현재 디렉토리(/opt/mlops)에 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2단계 - 환경설정 및 코드 복사
COPY .env.template .

# 데이터 전처리 스크립트
COPY data-prepare/ ./data-prepare/

# 소스코드
COPY src/ ./src/

# 모델 저장 디렉토리 생성
RUN mkdir -p models

# PYTHONPATH 설정 (명시적으로, 필요시)
ENV PYTHONPATH=/opt/mlops:$PYTHONPATH

# ENTRYPOINT와 CMD 분리로 유연성 확보
ENTRYPOINT ["python", "src/main.py"]
CMD ["train", "--model_name=movie_predictor", "--num_epochs=20"]
