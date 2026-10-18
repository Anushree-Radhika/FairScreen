FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install compilers, git, and required C/Fortran/Rust system libraries for s390x
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
    patchelf \
    libopenblas-dev \
    liblapack-dev \
    zlib1g-dev \
    libjpeg-dev \
    libssl-dev \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Ensure pyarrow is removed from requirements before pip install
RUN sed -i '/pyarrow/d' requirements.txt

# Upgrade core pip build tools and install build backends required by pandas/scipy
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir meson-python cython

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]