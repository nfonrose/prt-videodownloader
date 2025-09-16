# prt-videodownloader-api Dockerfile
# Using slim base and installing ffmpeg for yt-dlp post-processing/merging
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create expected directories inside the image
RUN mkdir -p /opt/prt/prt-videodownloader/db \
    && mkdir -p /opt/prt/prt-videodownloader/data

# Copy application code
COPY app.py entrypoint.py openapi.json ./

EXPOSE 9080

# Default environment (can be overridden by docker-compose)
ENV PORT=9080 \
    PRT_VIDEODOWNLOADER_VIDEODATAPATH=/opt/prt/prt-videodownloader/data \
    PRT_VIDEODOWNLOADER_SQLITEFILEPATH=/opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite \
    PRT_VIDEODOWNLOADER_HTTPSBASEURL=http://localhost:8080/videodownloader

CMD ["python", "entrypoint.py"]
