FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system deps (for pandas/numpy compile deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload and cleaned directories
RUN mkdir -p uploads cleaned

# Expose port (Render sets PORT env variable automatically)
EXPOSE 10000

# Run with gunicorn — production-ready
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --timeout 120 --worker-class sync app:app
