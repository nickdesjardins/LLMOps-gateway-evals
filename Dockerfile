# Production image for the public showcase (Render free web service: 512 MB RAM, no disk).
#
# The embedding model (BAAI/bge-small-en-v1.5, quantized ONNX, about 67 MB) is downloaded at
# BUILD time into the image, because a free host starts every instance with an empty disk:
# without this, each cold start would fetch it from Hugging Face before answering a question.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_DIR=/srv/models \
    HF_HOME=/srv/models/.hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # glibc keeps one malloc arena per thread by default, which inflates resident memory.
    MALLOC_ARENA_MAX=2

# The service runs as an unprivileged user. Only the model cache belongs to it; the code stays
# owned by root, so the process cannot rewrite it.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /srv/models \
    && chown app:app /srv/models

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

USER app

# Bake the weights into the image. The name must match FASTEMBED_MODEL (render.yaml sets it).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/srv/models')"

# From here on the model is read from the image only: no call to Hugging Face at run time.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY corpus ./corpus

EXPOSE 8000

# /livez answers without touching Redis or a model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/livez').status == 200 else 1)"

# Render injects PORT. One worker on purpose: the budget, the in-memory metrics and the
# embedding model live in this process, and a second worker would load a second model.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
