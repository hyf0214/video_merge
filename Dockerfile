FROM docker-regsitry.tencentcloudcr.com/wuwang/python:3.10-slim

USER root
WORKDIR /app
COPY . .
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt \
    && mkdir /app/temp
RUN curl -fL "https://g-ldyi2063-generic.pkg.coding.net/dev/test/footage.mp4?version=latest" -o footage.mp4

EXPOSE 8080/tcp
HEALTHCHECK CMD curl --fail http://localhost:8080/health
CMD ["python", "record_video_merge.py"]