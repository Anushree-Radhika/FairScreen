FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all compilers, build tools, and dev libraries required for s390x
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    libutf8proc-dev \
    libbrotli-dev \
    libbz2-dev \
    libzstd-dev \
    liblz4-dev \
    libsnappy-dev \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Upgrade pip build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Tell pyarrow build system to bundle its C++ components
ENV PYARROW_WITH_DATASET=1
ENV PYARROW_WITH_PARQUET=1
ENV PYARROW_BUILD_TYPE=release

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]