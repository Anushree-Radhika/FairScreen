FROM condaforge/miniforge3:latest

WORKDIR /app

COPY requirements.txt .

# Install dependencies using conda-forge which has pre-built s390x binaries
RUN mamba install -y -c conda-forge \
    python=3.11 \
    pandas \
    numpy \
    scikit-learn \
    scipy \
    joblib \
    streamlit \
    pypdf \
    pdfplumber \
    pyarrow \
    && mamba clean --all -f -y

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]
