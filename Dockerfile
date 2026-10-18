FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install compilers, git, and required system development libraries for s390x
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

# Remove any indirect pyarrow requirement if present
RUN sed -i '/pyarrow/d' requirements.txt

# Upgrade pip and pre-install build backend dependencies (fixes meson-python error)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel meson-python meson ninja

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]