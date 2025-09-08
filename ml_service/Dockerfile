# # ml_service/Dockerfile
# FROM python:3.11-slim

# WORKDIR /app

# # Install system dependencies for scientific packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY app/ /app/
# RUN mkdir -p /app/storage /app/config

# # Create empty __init__.py files to make directories packages
# RUN touch /app/__init__.py /app/core/__init__.py /app/models/__init__.py /app/storage/__init__.py /app/config/__init__.py

# # Set PYTHONPATH to recognize the app directory
# ENV PYTHONPATH=/

# # Run the application
# CMD ["python", "app/main.py"]

# ml_service/Dockerfile.fixed - Fixed Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create config directory
RUN mkdir -p /app/config

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FIXED: Use the correct path to main_emqx.py
CMD ["python", "-m", "app.main"]