# 在仓库根目录构建：docker build -f demo/Dockerfile -t eldercare-demo .
# 运行：docker run -p 8000:8000 -e DEMO_ACCESS_CODE=xxx -e DEEPSEEK_API_KEY=xxx eldercare-demo
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt /tmp/core-requirements.txt
COPY demo/requirements.txt /tmp/demo-requirements.txt
RUN pip install --no-cache-dir -r /tmp/core-requirements.txt -r /tmp/demo-requirements.txt

COPY app ./app
COPY seed ./seed
COPY demo ./demo

# 非 root 运行；演示数据全在内存里，不需要可写目录
RUN useradd --no-create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn demo.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
