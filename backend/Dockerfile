FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY . .

# 用 $PORT 啟動服務
CMD ["sh","-c","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]