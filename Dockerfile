FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl make build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock Makefile ./
RUN pip install --no-cache-dir -r requirements.lock && python -m spacy download en_core_web_sm
COPY . .
ENV HOST=0.0.0.0
EXPOSE 8501
CMD ["make", "app"]
