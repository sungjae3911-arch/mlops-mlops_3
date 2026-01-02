#!/bin/sh
set -e  # 에러 발생 시 스크립트 중단

echo '=== MLOps Pipeline ==='

case "$1" in
    ""|bash)
        echo "docker container 접속..."
        exec "$@"
        ;;

    preprocess)
        echo "Starting preprocessing..."
        python src/main.py "$@"
        ;;

    train)    
        echo "Starting train..."
        python src/main.py "$@"
        ;;

    inference)
        echo "Starting inference..."
        python src/main.py "$@"
        ;;       
    
    *)
        echo "Usage: {""|bash|preprocess|train|inference}"
        echo ' ""  : docker container 접속 '
        echo " bash :  docker container 접속 "
        echo " preprocess : 데이터 수집 및 전처리 진행 "
        echo " train : 수집된 데이터로 모델 학습 및 평가 "
        echo " iference : 학습 모델 추론 "
        ;;
esac