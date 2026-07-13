FROM python:3.11-slim

# Install system dependencies for building python packages and database client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Gunicorn port
EXPOSE 8765

# Start Gunicorn server
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8765", "notes_server:app"]
