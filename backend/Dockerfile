FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY . .

# 讓 Cloud Run 傳進來的 PORT 環境變數生效
ENV PORT=8080

# 用 $PORT 啟動服務
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT