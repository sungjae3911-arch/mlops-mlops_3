FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

LABEL maintainer="MLOps Pipeline"
LABEL version="1.0.0"
LABEL description="MLOps Pipeline"

WORKDIR /opt/mlops

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# COPY data-prepare/ ./data-prepare/
# COPY src/ ./src/

COPY .env.template .

# ENV PYTHONPATH=/opt/mlops

# entrypoint.sh & 실행권한
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
 # default: Script
CMD [""]
