FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install compilers, math/dev libraries, and the pre-compiled system pyarrow package
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cargo \
    rustc \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    zlib1g-dev \
    libjpeg-dev \
    libssl-dev \
    python3-pyarrow \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Remove or ignore pyarrow from requirements.txt to avoid pip trying to re-compile it
RUN sed -i '/pyarrow/d' requirements.txt

# Upgrade pip build tools and install remaining dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]