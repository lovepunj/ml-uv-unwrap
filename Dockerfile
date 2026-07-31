FROM python:3.9-slim

WORKDIR /app

# System deps: build tools for libigl/xatlas, GL libs for trimesh scene graph
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (default wheel pulls CUDA ~2GB+).
# PyPI must stay the primary index: the pytorch CPU channel does not mirror
# torch's deps (e.g. symsolve, typing_extensions), so --index-url alone fails.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# PartField checkpoint is excluded from git; unwrap partuv modes fall back
# to geometric features when it is absent.
RUN echo "PartField checkpoint not bundled (optional)"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8080}"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8080') + '/api/health')" || exit 1
