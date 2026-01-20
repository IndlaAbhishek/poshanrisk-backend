FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install NumPy FIRST (critical)
RUN pip install numpy==1.23.5

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
