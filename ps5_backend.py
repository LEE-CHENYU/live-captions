"""Remote GPU backend: a whisper.cpp server on a Linux box (here a PS5 running Linux), reached over SSH.

The server (systemd unit `ps5-whisper-server`, ggml-small on Vulkan) listens on the remote 127.0.0.1:8178.
This module starts it on demand over SSH and forwards it to the Mac's 127.0.0.1:18178 through an SSH
tunnel to PS5_HOST (a direct Ethernet link in the reference setup), so audio stays on the two machines.

Settings (environment / .env): PS5_HOST, PS5_SSH_USER, PS5_SSH_KEY, PS5_KNOWN_HOSTS, CAPTION_LANGUAGE.
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

REMOTE_PORT = 8178
LOCAL_PORT = 18178
SERVICE = "ps5-whisper-server"
WHISPER_RATE = 16_000
PARTIAL_INTERVAL = 0.7  # seconds between provisional updates
MAX_TURN_SECONDS = 25.0  # whisper sees at most 30 s; settle long turns before that
NOISE_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")  # [BLANK_AUDIO], (music), *laughs*

_tunnel_lock = threading.Lock()
_tunnel: subprocess.Popen | None = None


def target() -> str:
    """user@host; use the Ethernet address so the tunnel never falls back to Wi-Fi or a VPN."""
    return f'{os.getenv("PS5_SSH_USER", "root")}@{os.getenv("PS5_HOST", "192.168.77.2")}'


def ssh() -> list[str]:
    """SSH options, read at call time so a .env loaded after import still applies."""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=yes",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
    key = os.getenv("PS5_SSH_KEY", "").strip()
    if key:
        command += ["-i", os.path.expanduser(key), "-o", "IdentitiesOnly=yes"]
    known_hosts = os.getenv("PS5_KNOWN_HOSTS", "").strip()
    if known_hosts:
        command += ["-o", f"UserKnownHostsFile={os.path.expanduser(known_hosts)}"]
    return command


def language() -> str:
    return os.getenv("CAPTION_LANGUAGE", "en").strip() or "en"


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def _answers() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/", timeout=2).read(64)
        return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_server(timeout: float = 45.0) -> None:
    """Start the PS5 service (if needed) and the Ethernet tunnel; wait until it answers.

    Live captions and dictation share the local port: if the other one's tunnel already answers, reuse it.
    """
    global _tunnel
    with _tunnel_lock:
        if _port_open(LOCAL_PORT) and _answers():
            return
        start = subprocess.run(
            ssh() + [target(), f"systemctl start {SERVICE} && systemctl is-active {SERVICE}"],
            capture_output=True, text=True, timeout=30,
        )
        if start.returncode != 0:
            detail = (start.stderr or start.stdout).strip().splitlines()
            if "ExecCondition" in " ".join(detail) or "condition" in " ".join(detail).lower():
                raise RuntimeError("PS5 is in LM mode; the GPU caption server only runs in multitask mode.")
            raise RuntimeError("Cannot start the PS5 caption server over SSH: " + (detail[-1] if detail else "ssh failed"))
        if _tunnel is not None and _tunnel.poll() is None:
            _tunnel.kill()
        _tunnel = subprocess.Popen(
            ssh() + ["-N", "-o", "ExitOnForwardFailure=yes",
                     "-L", f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}", target()],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _tunnel.poll() is not None:
                raise RuntimeError("The Ethernet tunnel to the PS5 closed; is the cable connected?")
            if _answers():
                return
            time.sleep(0.5)
        raise RuntimeError("The PS5 caption server did not answer in time.")


def close_tunnel(stop_server: bool = False) -> None:
    """Close this process's tunnel. The PS5 service keeps running (dictation may still use it; ~0.5 GB of
    GPU memory, idle otherwise); `ps5-mode lm` stops it, or pass stop_server=True."""
    global _tunnel
    with _tunnel_lock:
        if _tunnel is not None and _tunnel.poll() is None:
            _tunnel.kill()
        _tunnel = None
    if stop_server:
        try:
            subprocess.run(ssh() + [target(), f"systemctl stop {SERVICE}"],
                           capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass


def to_whisper_pcm(samples: np.ndarray, rate: int) -> np.ndarray:
    """float32 mono at `rate` -> int16 mono at 16 kHz."""
    if rate != WHISPER_RATE and len(samples):
        n = int(round(len(samples) * WHISPER_RATE / rate))
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def wav_bytes(pcm16: np.ndarray) -> bytes:
    data = pcm16.tobytes()
    header = b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    header += (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + WHISPER_RATE.to_bytes(4, "little")
    header += (WHISPER_RATE * 2).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    return header + b"data" + len(data).to_bytes(4, "little") + data


def inference(wav: bytes, timeout: float = 60.0, **fields: object) -> dict:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{LOCAL_PORT}/inference", data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", NOISE_TAGS.sub(" ", text)).strip()


class Ps5CaptionClient:
    """Live captions from the PS5 GPU, with the same interface as RealtimeCaptionClient.

    Each turn's audio is re-transcribed every PARTIAL_INTERVAL seconds (provisional words, replacing the
    previous guess) and once more in full when the app commits the turn after a pause (settled sentence).
    """

    replaces_partial = True

    def __init__(self, rate: int, on_delta, on_completed, on_status, on_error) -> None:
        self.rate = rate
        self.on_delta = on_delta
        self.on_completed = on_completed
        self.on_status = on_status
        self.on_error = on_error
        self.connected = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.blocks: list[np.ndarray] = []
        self.samples = 0
        self.dirty = False
        self.item_id = uuid.uuid4().hex
        self.commits: queue.Queue[tuple[str, np.ndarray]] = queue.Queue()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def append_audio(self, block: np.ndarray) -> None:
        with self.lock:
            self.blocks.append(block.astype(np.float32, copy=True))
            self.samples += len(block)
            self.dirty = True
            too_long = self.samples / self.rate >= MAX_TURN_SECONDS
        if too_long:
            self.commit()

    def _take_turn(self) -> tuple[str, np.ndarray]:
        with self.lock:
            audio = np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)
            item, self.item_id = self.item_id, uuid.uuid4().hex
            self.blocks, self.samples, self.dirty = [], 0, False
        return item, audio

    def commit(self) -> bool:
        if not self.connected.is_set():
            return False
        self.commits.put(self._take_turn())
        return True

    def _transcribe(self, audio: np.ndarray, partial: bool) -> str:  # partial kept for logging/tuning
        if len(audio) < self.rate * 0.3 or float(np.max(np.abs(audio))) < 0.01:
            return ""
        pcm = to_whisper_pcm(audio, self.rate)
        # whisper-server keeps request parameters between calls, so always reset context and prompt.
        fields: dict[str, object] = {"response_format": "json", "language": language(), "temperature": "0.0",
                                     "no_context": "true", "prompt": ""}
        # No audio_ctx shortcut: on short windows it makes whisper loop ("my fellow Americans" x4, 4.7 s).
        return clean_text(str(inference(wav_bytes(pcm), timeout=20, **fields).get("text", "")))

    def _run(self) -> None:
        try:
            self.on_status("reconnecting")
            ensure_server()
            self.connected.set()
            self.on_status("connected")
        except Exception as error:
            self.on_error(str(error))
            return
        last_partial = 0.0
        while not self.stopping.is_set() or not self.commits.empty():
            try:
                item, audio = self.commits.get(timeout=0.1)
                try:
                    self.on_completed(item, self._transcribe(audio, partial=False))
                except Exception as error:
                    self.on_error(f"PS5 caption request failed: {error}")
                    self.on_completed(item, "")
                finally:
                    self.commits.task_done()
                continue
            except queue.Empty:
                pass
            now = time.monotonic()
            with self.lock:
                due = self.dirty and now - last_partial >= PARTIAL_INTERVAL
                if due:
                    item, audio, self.dirty = self.item_id, np.concatenate(self.blocks), False
            if not due or self.stopping.is_set():
                continue
            last_partial = now
            try:
                text = self._transcribe(audio, partial=True)
                with self.lock:
                    current = item == self.item_id
                if text and current:
                    self.on_delta(item, text, True)
            except Exception as error:
                self.on_error(f"PS5 caption request failed: {error}")

    def finish(self, commit_open_turn: bool, timeout: float = 6.0) -> None:
        if commit_open_turn:
            self.commit()
        deadline = time.monotonic() + timeout
        while self.commits.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


def transcribe_file(session_dir: Path, ffmpeg: Path, duration: float, part_seconds: float) -> tuple[list[dict], list[str], list[dict]]:
    """Full-session transcript on the PS5 GPU, in parts. Returns (segments, texts, raw parts)."""
    ensure_server()
    source = session_dir / "recording.wav"
    segments: list[dict] = []
    texts: list[str] = []
    raw_parts: list[dict] = []
    offset = 0.0
    index = 0
    while offset < max(duration, 0.1):
        length = min(part_seconds, duration - offset) if duration > 0 else part_seconds
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}", "-t", f"{length:.3f}",
             "-i", str(source), "-ac", "1", "-ar", str(WHISPER_RATE), "-f", "wav", "-"],
            capture_output=True, check=True,
        )
        fields: dict[str, object] = {"response_format": "verbose_json", "language": language(), "temperature": "0.0",
                                     "no_context": "true", "prompt": texts[-1][-500:] if texts else ""}
        raw = inference(result.stdout, timeout=1800, **fields)
        raw_parts.append({"part": index, "offset_seconds": offset, "response": raw})
        text = clean_text(str(raw.get("text", "")))
        if text:
            texts.append(text)
        for segment in raw.get("segments", []) or []:
            if isinstance(segment, dict) and clean_text(str(segment.get("text", ""))):
                shifted = dict(segment)
                shifted["text"] = clean_text(str(segment.get("text", "")))
                shifted["start"] = float(segment.get("start", 0.0)) + offset
                shifted["end"] = float(segment.get("end", segment.get("start", 0.0))) + offset
                segments.append(shifted)
        offset += length
        index += 1
        if duration <= 0:
            break
    return segments, texts, raw_parts
