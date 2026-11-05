FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY murmur ./murmur
RUN pip install --no-cache-dir ".[kokoro]"

# Books, finished audiobooks and voice models live in volumes so they survive upgrades.
ENV MURMUR_HOME=/data
VOLUME ["/data"]
EXPOSE 8000

# First run: `docker compose run --rm murmur murmur models kokoro --small`
CMD ["murmur", "serve", "--host", "0.0.0.0", "--port", "8000"]
