FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8090
EXPOSE 8090

CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8090"]
