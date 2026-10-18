FROM python:3.11-slim

WORKDIR /app

# Suppress interactive debconf warnings
ENV DEBIAN_FRONTEND=noninteractive

# Install C/C++, Fortran, and Rust compilers required for s390x wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cargo \
    rustc \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# Upgrade pip and setuptools before installing requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]