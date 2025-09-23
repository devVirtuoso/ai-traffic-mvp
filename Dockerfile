# Dockerfile for ai-traffic-mvp
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Streamlit dashboard port
EXPOSE 8501

# Default command: run dashboard
CMD ["streamlit", "run", "src/dashboard/app.py"]
