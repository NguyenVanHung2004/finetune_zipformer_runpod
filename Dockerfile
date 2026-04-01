# ─────────────────────────────────────────────────────────────────
# Dockerfile – RunPod Fine-Tune: Zipformer Adapter (Vietnamese ASR)
# ─────────────────────────────────────────────────────────────────
# Base image: PyTorch 2.4.0 + Python 3.11 + CUDA 12.4 (devel)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ICEFALL_DIR=/icefall
ENV HF_HOME=/workspace/hf_cache

# ── 1. System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg wget curl sox \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Upgrade pip ──────────────────────────────────────────────
RUN pip install --upgrade pip --quiet

# ── 3. Install k2  (must match: torch 2.4.0 + cuda 12.4 + py 3.11)
#    Try two common manylinux tags; first success wins.
RUN pip install --no-cache-dir --no-deps \
        "https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250715+cuda12.4.torch2.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
    || pip install --no-cache-dir --no-deps \
        "https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250715+cuda12.4.torch2.4.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"

# ── 4. Clone icefall (shallow) + install its deps ───────────────
RUN git clone --depth=1 https://github.com/k2-fsa/icefall /icefall

RUN pip install --no-cache-dir -q -r /icefall/requirements.txt \
    && pip install --no-cache-dir -q -e /icefall

# ── 5. Install app-level Python deps ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 6. Copy app code ────────────────────────────────────────────
COPY . /app/

# ── 7. Run builder: patches icefall scripts at image-build time ─
RUN python builder.py

# ── 8. Entry-point ──────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
