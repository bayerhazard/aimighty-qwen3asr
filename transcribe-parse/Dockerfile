# transcribe-parse sidecar for AIM Qwen3 1.7B ASR.
# Adds ffmpeg so the proxy can decode + chunk audio for response_format=verbose_json
# (vLLM cannot emit verbose_json for qwen3-asr). The proxy script is also provided
# by the chart's ConfigMap (mounted at /opt/transcribe-parse.py) — this baked copy
# only serves as a working default.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY transcribe-parse.py /opt/transcribe-parse.py
RUN chmod 0755 /opt/transcribe-parse.py

USER 1000
ENTRYPOINT ["python3", "/opt/transcribe-parse.py"]
