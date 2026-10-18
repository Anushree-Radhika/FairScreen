FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core build dependencies and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cargo \
    rustc \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    zlib1g-dev \
    libjpeg-dev \
    libssl-dev \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Ensure pyarrow is removed from requirements
RUN sed -i '/pyarrow/d' requirements.txt

# Upgrade pip build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies while telling pip NOT to attempt building pyarrow
RUN pip install --no-cache-dir --no-deps-check -r requirements.txt || \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]