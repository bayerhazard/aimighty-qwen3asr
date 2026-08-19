#!/usr/bin/env python3
# ASR proxy with verbose_json support (sidecar container).
#
# Qwen3-ASR / vLLM cannot emit verbose_json itself (supports_segment_timestamp
# is False, base vLLM rejects response_format=verbose_json with HTTP 400). The
# proxy therefore:
#   - response_format != verbose_json  -> passthrough to vLLM, strip the
#     "language <X><asr_text>" metadata so clients get pure text (unchanged).
#   - response_format == verbose_json  -> decode the audio with ffmpeg (16 kHz
#     mono PCM WAV), transcribe fixed time chunks, return one OpenAI-compatible
#     segment per chunk with real start/end timestamps. Chunk granularity only —
#     Qwen3-ASR never emits word-level timestamps.
# If chunking fails, fall back to a plain json transcription so clients still
# get text (Insilo then builds a single segment) instead of a hard 400.
import email
import email.policy
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

BACKEND = os.environ.get("AM_BACKEND", "http://127.0.0.1:8000").rstrip("/")
PORT = int(os.environ.get("AM_PORT", "8010"))
CHUNK_SECONDS = int(os.environ.get("VERBOSE_CHUNK_SECONDS", "30"))
MIN_TAIL_SECONDS = 0.5
REQUEST_TIMEOUT = int(os.environ.get("AM_TIMEOUT", "900"))
TARGET_RATE = 16000
TARGET_CHANNELS = 1
TARGET_WIDTH = 2  # 16-bit PCM

_TAG_RE = re.compile(r"<\|[^|]*\|>")


def clean_text(raw):
    if not raw:
        return ""
    s = str(raw).strip()
    s = _TAG_RE.sub("", s)
    if "<asr_text>" in s:
        s = s.split("<asr_text>", 1)[1]
    else:
        m = re.match(r"^language\s+\S+\s*\n", s)
        if m:
            s = s[m.end():]
    return s.strip()


def parse_raw(raw):
    """Return (detected_language_or_None, clean_text) from vLLM raw output."""
    s = str(raw).strip()
    lang = None
    m = re.match(r"^language\s+([^<\s]+)", s)
    if m:
        lang = m.group(1)
    s = _TAG_RE.sub("", s)
    if "<asr_text>" in s:
        s = s.split("<asr_text>", 1)[1]
    elif lang:
        s = re.sub(r"^language\s+\S+\s*", "", s, count=1)
    return lang, s.strip()


def parse_multipart(ctype, body):
    """Extract form fields + file from a multipart/form-data body (stdlib)."""
    msg = email.message_from_bytes(
        b"MIME-Version: 1.0\r\nContent-Type: "
        + ctype.encode("latin-1", "replace")
        + b"\r\n\r\n"
        + body,
        policy=email.policy.default,
    )
    fields = {}
    file_bytes = None
    filename = None
    file_ct = "application/octet-stream"
    for part in msg.iter_parts():
        if part.is_multipart():
            continue
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        if name == "file":
            file_bytes = payload
            filename = part.get_filename()
            ct = part.get_content_type()
            if ct and ct != "text/plain":
                file_ct = ct
        else:
            try:
                fields[name] = payload.decode("utf-8")
            except Exception:
                fields[name] = payload.decode("latin-1", "replace")
    return fields, file_bytes, filename, file_ct


def build_multipart(fields, files):
    boundary = "----amx" + os.urandom(8).hex()
    parts = []
    for name, value in fields.items():
        parts.append(("--" + boundary).encode("ascii"))
        parts.append(('Content-Disposition: form-data; name="%s"' % name).encode("ascii"))
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))
    for name, (fname, data, ftype) in files.items():
        parts.append(("--" + boundary).encode("ascii"))
        parts.append(
            ('Content-Disposition: form-data; name="%s"; filename="%s"' % (name, fname)).encode("ascii")
        )
        parts.append(("Content-Type: %s" % ftype).encode("ascii"))
        parts.append(b"")
        parts.append(data)
    parts.append(("--" + boundary + "--").encode("ascii"))
    return boundary, b"\r\n".join(parts)


def post_backend(body, ctype, timeout=REQUEST_TIMEOUT):
    req = urllib.request.Request(
        BACKEND + "/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wav_data_bounds(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError("not a RIFF/WAVE file")
    i = 12
    while i + 8 <= len(head):
        cid = head[i:i + 4]
        size = struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"data":
            return i + 8, size
        i += 8 + size + (size & 1)
    raise RuntimeError("no data chunk in WAV")


def make_wav(pcm):
    n = len(pcm)
    hdr = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    hdr += b"fmt " + struct.pack(
        "<IHHIIHH",
        16, 1, TARGET_CHANNELS, TARGET_RATE,
        TARGET_RATE * TARGET_CHANNELS * TARGET_WIDTH,
        TARGET_CHANNELS * TARGET_WIDTH,
        TARGET_WIDTH * 8,
    )
    hdr += b"data" + struct.pack("<I", n)
    return hdr + pcm


def transcribe_verbose(fields, file_bytes, filename, file_ct):
    tmp = tempfile.mkdtemp(prefix="amx-")
    try:
        src = os.path.join(tmp, "input.bin")
        with open(src, "wb") as f:
            f.write(file_bytes)
        wav = os.path.join(tmp, "full.wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
             "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_RATE),
             "-sample_fmt", "s16", "-f", "wav", wav],
            check=True, capture_output=True, timeout=300,
        )
        off, size = wav_data_bounds(wav)
        if size <= 0:
            raise RuntimeError("empty audio after decode")
        bps = TARGET_RATE * TARGET_CHANNELS * TARGET_WIDTH
        duration = size / bps
        chunk_bytes = int(CHUNK_SECONDS * bps)
        ranges = []
        pos = 0
        while pos < size:
            end = min(pos + chunk_bytes, size)
            ranges.append((pos, end))
            pos = end
        min_tail = int(MIN_TAIL_SECONDS * bps)
        if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) < min_tail:
            ranges[-2] = (ranges[-2][0], ranges[-1][1])
            ranges.pop()

        payload = {"response_format": "json"}
        if fields.get("model"):
            payload["model"] = fields["model"]
        if fields.get("language"):
            payload["language"] = fields["language"]

        segments = []
        texts = []
        detected = None
        with open(wav, "rb") as f:
            for idx, (start, end) in enumerate(ranges):
                f.seek(off + start)
                chunk = make_wav(f.read(end - start))
                boundary, body = build_multipart(
                    payload, {"file": ("chunk.wav", chunk, "audio/wav")}
                )
                resp = post_backend(body, "multipart/form-data; boundary=" + boundary)
                try:
                    data = json.loads(resp)
                except Exception:
                    raise RuntimeError("bad backend json: %r" % resp[:200])
                lang, text = parse_raw((data or {}).get("text", ""))
                if not text:
                    continue
                if detected is None and lang:
                    detected = lang
                s = start / bps
                e = end / bps
                texts.append(text)
                segments.append({
                    "id": idx, "seek": round(s, 3), "start": round(s, 3),
                    "end": round(e, 3), "text": text, "tokens": [],
                    "temperature": 0.0, "avg_logprob": 0.0,
                    "compression_ratio": 1.0, "no_speech_prob": 0.0,
                })
        return {
            "task": "transcribe",
            "language": detected or fields.get("language") or None,
            "duration": round(duration, 3),
            "text": " ".join(texts).strip(),
            "segments": segments,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/health"):
            try:
                urllib.request.urlopen(BACKEND + "/health", timeout=5)
                self._send_json(b'{"status":"ok"}')
            except Exception:
                self._send_error_json(503, b'{"status":"unavailable"}')
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "") or ""

        fields, file_bytes, filename, file_ct = {}, None, None, None
        try:
            fields, file_bytes, filename, file_ct = parse_multipart(ctype, body)
        except Exception:
            pass

        want_verbose = str(fields.get("response_format", "")).lower() == "verbose_json"

        if want_verbose and file_bytes:
            try:
                out = transcribe_verbose(fields, file_bytes, filename, file_ct)
                self._send_json(json.dumps(out).encode())
                return
            except Exception as exc:
                # Graceful fallback: plain json so clients still get a transcript.
                try:
                    fb = {"response_format": "json"}
                    if fields.get("model"):
                        fb["model"] = fields["model"]
                    if fields.get("language"):
                        fb["language"] = fields["language"]
                    boundary, fbody = build_multipart(
                        fb, {"file": (filename or "recording.bin", file_bytes, file_ct)}
                    )
                    resp = post_backend(fbody, "multipart/form-data; boundary=" + boundary)
                    data = json.loads(resp)
                    if isinstance(data, dict) and "text" in data:
                        data["text"] = clean_text(data["text"])
                    self._send_json(json.dumps(data).encode())
                except Exception:
                    self._send_error_json(502, json.dumps(
                        {"error": {"message": "verbose_json transcription failed: %s" % exc}}
                    ).encode())
                return

        # Passthrough (unchanged behaviour).
        req = urllib.request.Request(
            BACKEND + "/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": ctype or "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                resp_data = resp.read()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
            return
        except Exception:
            self.send_error(502)
            return
        try:
            data = json.loads(resp_data)
            if isinstance(data, dict) and "text" in data:
                data["text"] = clean_text(data["text"])
            resp_data = json.dumps(data).encode()
        except Exception:
            pass
        self._send_json(resp_data)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
