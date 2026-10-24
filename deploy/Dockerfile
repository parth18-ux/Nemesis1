# One image for the API and the scoring workers.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt redis
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
