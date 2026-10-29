#!/usr/bin/env python3
"""Word-by-word live captions with corrected turns and queued final transcripts."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
import websocket

import ps5_backend


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "sessions"
ENV_PATH = PROJECT_ROOT / ".env"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")

API_MODEL = "whisper-1"
LIVE_MODEL = "gpt-live-transcribe"
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
DEVICE = None  # Follow the current macOS default input device.
RATE = 24_000
TURN_SILENCE_SECONDS = 0.75
UPLOAD_PART_SECONDS = 20 * 60
HISTORY_LIMIT = 6

BG = "#F5F7FA"
INK = "#172231"
MUTED = "#687486"
FAINT = "#8D98A8"
BLUE = "#2368F2"
BLUE_HOVER = "#1757D6"
PROVISIONAL = "#60799B"
RED = "#D94747"
RED_SOFT = "#FFF4F3"
PANEL = "#FFFFFF"
LINE = "#DDE3EB"
GREEN = "#198754"
AMBER = "#B26A00"

METADATA_LOCK = threading.Lock()

# Caption backend: "openai" (Realtime + whisper-1) or "ps5" (whisper.cpp on the PS5 GPU over the Ethernet
# cable). Default from CAPTION_BACKEND in .env; the header button changes it and saves settings.json.
BACKENDS = {"openai": "OpenAI", "ps5": "PS5 GPU · Ethernet"}
SETTINGS_PATH = PROJECT_ROOT / "settings.json"
PS5_MODEL = "whisper.cpp ggml-small (PS5 GPU)"


def load_backend() -> str:
    try:
        backend = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("backend")
    except (OSError, ValueError):
        backend = None
    backend = backend or os.getenv("CAPTION_BACKEND", "openai").strip().lower()
    return backend if backend in BACKENDS else "openai"


def save_backend(backend: str) -> None:
    SETTINGS_PATH.write_text(json.dumps({"backend": backend}, indent=2) + "\n", encoding="utf-8")


def timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def redact_error(error: object) -> str:
    text = str(error).strip() or error.__class__.__name__
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted API key]", text)
    return text[:600]


def read_metadata(session_dir: Path) -> dict:
    path = session_dir / "session.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def update_metadata(session_dir: Path, **updates: object) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session.json"
    with METADATA_LOCK:
        metadata = read_metadata(session_dir)
        metadata.update(updates)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return metadata


def audio_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as source:
        return source.getnframes() / source.getframerate()


def run_ffmpeg(arguments: list[str]) -> None:
    if not FFMPEG.exists():
        raise RuntimeError("ffmpeg is required but was not found at /opt/homebrew/bin/ffmpeg")
    completed = subprocess.run(
        [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or "ffmpeg could not prepare the recording"
        raise RuntimeError(detail)


def ensure_compressed_audio(session_dir: Path) -> tuple[Path, float]:
    wav_path = session_dir / "recording.wav"
    m4a_path = session_dir / "recording.m4a"
    if not wav_path.exists():
        raise FileNotFoundError("The saved recording.wav file is missing.")
    duration = audio_duration(wav_path)
    if not m4a_path.exists() or m4a_path.stat().st_mtime < wav_path.stat().st_mtime:
        run_ffmpeg(
            [
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-c:a",
                "aac",
                "-b:a",
                "48k",
                str(m4a_path),
            ]
        )
    return m4a_path, duration


def upload_files_for_session(session_dir: Path, duration: float) -> list[Path]:
    m4a_path = session_dir / "recording.m4a"
    if duration <= UPLOAD_PART_SECONDS:
        return [m4a_path]

    parts_dir = session_dir / "api_parts"
    parts_dir.mkdir(exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(session_dir / "recording.wav"),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-f",
            "segment",
            "-segment_time",
            str(UPLOAD_PART_SECONDS),
            "-reset_timestamps",
            "1",
            str(parts_dir / "part-%03d.m4a"),
        ]
    )
    parts = sorted(parts_dir.glob("part-*.m4a"))
    if not parts:
        raise RuntimeError("The recording could not be divided into upload parts.")
    return parts


def response_dictionary(response: object) -> dict:
    if hasattr(response, "model_dump"):
        value = response.model_dump()
        return value if isinstance(value, dict) else {"text": str(value)}
    if isinstance(response, dict):
        return response
    return {"text": getattr(response, "text", str(response))}


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class RealtimeCaptionClient:
    """Streams PCM audio to a Realtime transcription session."""

    def __init__(
        self,
        api_key: str,
        on_delta,
        on_completed,
        on_status,
        on_error,
    ) -> None:
        self.api_key = api_key
        self.on_delta = on_delta
        self.on_completed = on_completed
        self.on_status = on_status
        self.on_error = on_error
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=400)
        self.connected = threading.Event()
        self.stopping = threading.Event()
        self.send_lock = threading.Lock()
        self.pending_condition = threading.Condition()
        self.pending_commits = 0
        self.ws: websocket.WebSocketApp | None = None
        self.connection_thread: threading.Thread | None = None
        self.sender_thread: threading.Thread | None = None

    def start(self) -> None:
        self.connection_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.connection_thread.start()
        self.sender_thread.start()

    def _connection_loop(self) -> None:
        while not self.stopping.is_set():
            self.connected.clear()
            self.ws = websocket.WebSocketApp(
                REALTIME_URL,
                header=[f"Authorization: Bearer {self.api_key}"],
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws.run_forever()
            if not self.stopping.is_set():
                self.on_status("reconnecting")
                time.sleep(1.25)

    def _on_open(self, _ws) -> None:
        self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": RATE},
                            "transcription": {
                                "model": LIVE_MODEL,
                                "languages": ["en"],
                                "delay": "low",
                            },
                            "turn_detection": None,
                        }
                    },
                },
            }
        )

    def _on_message(self, _ws, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        kind = event.get("type")
        if kind == "session.updated":
            self.connected.set()
            self.on_status("connected")
        elif kind == "conversation.item.input_audio_transcription.delta":
            self.on_delta(str(event.get("item_id", "current")), str(event.get("delta", "")))
        elif kind == "conversation.item.input_audio_transcription.completed":
            with self.pending_condition:
                self.pending_commits = max(0, self.pending_commits - 1)
                self.pending_condition.notify_all()
            self.on_completed(
                str(event.get("item_id", "current")),
                str(event.get("transcript", "")).strip(),
            )
        elif kind == "error":
            error = event.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            self.on_error(redact_error(message or "Realtime transcription error"))

    def _on_error(self, _ws, error: object) -> None:
        if not self.stopping.is_set():
            self.connected.clear()
            self.on_error(redact_error(error))

    def _on_close(self, _ws, _status_code, _message) -> None:
        self.connected.clear()

    def _send(self, event: dict) -> None:
        with self.send_lock:
            if self.ws is None:
                raise RuntimeError("Realtime caption connection is not ready.")
            self.ws.send(json.dumps(event))

    def append_audio(self, block: np.ndarray) -> None:
        pcm = np.clip(block, -1.0, 1.0)
        payload = (pcm * 32767.0).astype("<i2").tobytes()
        try:
            self.audio_queue.put_nowait(payload)
        except queue.Full:
            self.on_error("Live caption audio fell behind; the saved recording is unaffected.")

    def _sender_loop(self) -> None:
        while not self.stopping.is_set():
            payload = self.audio_queue.get()
            if payload is None:
                self.audio_queue.task_done()
                break
            sent = False
            while not sent and not self.stopping.is_set():
                if not self.connected.wait(timeout=0.5):
                    continue
                try:
                    self._send(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(payload).decode("ascii"),
                        }
                    )
                    sent = True
                except Exception:
                    self.connected.clear()
            self.audio_queue.task_done()

    def commit(self) -> bool:
        if not self.connected.is_set():
            return False
        with self.pending_condition:
            self.pending_commits += 1
        try:
            self._send({"type": "input_audio_buffer.commit"})
            return True
        except Exception as error:
            with self.pending_condition:
                self.pending_commits = max(0, self.pending_commits - 1)
            self.connected.clear()
            self.on_error(redact_error(error))
            return False

    def finish(self, commit_open_turn: bool, timeout: float = 4.0) -> None:
        drain_deadline = time.monotonic() + min(2.0, timeout / 2)
        while self.audio_queue.unfinished_tasks and time.monotonic() < drain_deadline:
            time.sleep(0.05)
        if commit_open_turn:
            self.commit()
        deadline = time.monotonic() + timeout
        with self.pending_condition:
            while self.pending_commits and time.monotonic() < deadline:
                self.pending_condition.wait(timeout=max(0.05, deadline - time.monotonic()))
        self.stopping.set()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.sender_thread is not None:
            self.sender_thread.join(timeout=2)
        if self.connection_thread is not None:
            self.connection_thread.join(timeout=2)


def transcribe_session_with_api(session_dir: Path) -> dict:
    load_dotenv(ENV_PATH, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")

    m4a_path, duration = ensure_compressed_audio(session_dir)
    upload_paths = upload_files_for_session(session_dir, duration)
    client = OpenAI(api_key=api_key, timeout=900.0)

    merged_segments: list[dict] = []
    merged_text: list[str] = []
    raw_parts: list[dict] = []
    offset = 0.0

    for index, upload_path in enumerate(upload_paths):
        request: dict[str, object] = {
            "model": API_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if merged_text:
            request["prompt"] = merged_text[-1][-500:]
        with upload_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(file=audio_file, **request)
        raw = response_dictionary(response)
        raw_parts.append(
            {
                "part": index,
                "file": str(upload_path.relative_to(session_dir)),
                "offset_seconds": offset,
                "response": raw,
            }
        )
        part_text = str(raw.get("text", "")).strip()
        if part_text:
            merged_text.append(part_text)
        for segment in raw.get("segments", []) or []:
            if not isinstance(segment, dict):
                continue
            shifted = dict(segment)
            shifted["start"] = float(segment.get("start", 0.0)) + offset
            shifted["end"] = float(segment.get("end", segment.get("start", 0.0))) + offset
            merged_segments.append(shifted)
        part_duration = raw.get("duration")
        if isinstance(part_duration, (int, float)) and part_duration > 0:
            offset += float(part_duration)
        else:
            offset += min(UPLOAD_PART_SECONDS, max(0.0, duration - offset))

    transcript_text = "\n\n".join(merged_text).strip()
    if not transcript_text:
        raise RuntimeError("Whisper returned an empty transcript.")
    return write_transcript(session_dir, duration, transcript_text, merged_segments, raw_parts,
                            f"OpenAI {API_MODEL}", API_MODEL, m4a_path.name, len(upload_paths))


def transcribe_session_with_ps5(session_dir: Path) -> dict:
    duration = audio_duration(session_dir / "recording.wav")
    segments, texts, raw_parts = ps5_backend.transcribe_file(session_dir, FFMPEG, duration, UPLOAD_PART_SECONDS)
    transcript_text = "\n\n".join(texts).strip()
    if not transcript_text:
        raise RuntimeError("The PS5 returned an empty transcript.")
    return write_transcript(session_dir, duration, transcript_text, segments, raw_parts,
                            PS5_MODEL, PS5_MODEL, "recording.wav", len(raw_parts))


def transcribe_session(session_dir: Path) -> dict:
    """Final transcript with the backend the session was recorded with (old sessions: OpenAI)."""
    if read_metadata(session_dir).get("backend") == "ps5":
        return transcribe_session_with_ps5(session_dir)
    return transcribe_session_with_api(session_dir)


def write_transcript(session_dir: Path, duration: float, transcript_text: str, merged_segments: list[dict],
                     raw_parts: list[dict], label: str, model: str, compressed_name: str, parts: int) -> dict:

    text_path = session_dir / "transcript.txt"
    text_path.write_text(transcript_text + "\n", encoding="utf-8")

    metadata = read_metadata(session_dir)
    lines = [
        "# Live captions transcript",
        "",
        f"- Recorded: {metadata.get('recorded_at', datetime.now().astimezone().isoformat(timespec='seconds'))}",
        f"- Duration: {timestamp(duration)}",
        f"- Transcription: {label}",
        "- Speaker labels: not available",
        "",
        "## Transcript",
        "",
    ]
    if merged_segments:
        for segment in merged_segments:
            text = str(segment.get("text", "")).strip()
            if text:
                lines.append(f"**[{timestamp(float(segment.get('start', 0.0)))}]** {text}")
                lines.append("")
    else:
        lines.extend([transcript_text, ""])
    markdown_path = session_dir / "transcript.md"
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    raw_path = session_dir / "whisper_api_response.json"
    raw_path.write_text(
        json.dumps(
            {
                "model": model,
                "duration_seconds": duration,
                "parts": raw_parts,
                "segments": merged_segments,
                "text": transcript_text,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "duration_seconds": duration,
        "audio": "recording.wav",
        "compressed_audio": compressed_name,
        "transcript_markdown": markdown_path.name,
        "transcript_text": text_path.name,
        "api_response": raw_path.name,
        "upload_parts": parts,
    }


class CaptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live captions")
        self.root.geometry("920x680+150+70")
        self.root.minsize(720, 560)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)

        families = tkfont.families()
        family = "Avenir Next" if "Avenir Next" in families else "Helvetica"
        self.title_font = tkfont.Font(family=family, size=17, weight="bold")
        self.caption_font = tkfont.Font(family=family, size=27, weight="normal")
        self.status_font = tkfont.Font(family=family, size=12, weight="bold")
        self.small_font = tkfont.Font(family=family, size=11)
        self.tiny_font = tkfont.Font(family=family, size=10)
        self.button_font = tkfont.Font(family=family, size=15, weight="bold")

        self.running = threading.Event()
        self.capture_busy = threading.Event()
        self.shutdown = threading.Event()
        self.model_state = "preparing"
        self.closing_requested = False
        self.stream: sd.InputStream | None = None
        self.audio_lock = threading.Lock()
        self.capture_close_lock = threading.Lock()
        self.writer_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=500)
        self.writer: wave.Wave_write | None = None
        self.writer_thread: threading.Thread | None = None
        self.capture_thread: threading.Thread | None = None
        self.session_token = 0
        self.session_started = 0.0
        self.session_started_wall: datetime | None = None
        self.session_dir: Path | None = None
        self.audio_path: Path | None = None
        self.live_client: RealtimeCaptionClient | ps5_backend.Ps5CaptionClient | None = None
        load_dotenv(ENV_PATH, override=False)
        self.backend = load_backend()
        self.speech_threshold = 0.008
        self.latest_rms = 0.0
        self.last_voice_at = 0.0
        self.turn_has_speech = False
        self.last_commit_at = 0.0
        self.live_item_order: list[str] = []
        self.live_partial: dict[str, str] = {}
        self.live_final: dict[str, str] = {}
        self.caption_note_after_id: str | None = None
        self.task_queue: queue.Queue[Path | None] = queue.Queue()
        self.ui_events: queue.Queue[object] = queue.Queue()
        self.queue_lock = threading.Lock()
        self.queued_sessions: set[Path] = set()
        self.active_transcription: Path | None = None
        self.history_after_id: str | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.bind(
            "<space>",
            lambda _event: self.toggle() if str(self.action["state"]) == "normal" else None,
        )
        self.root.after(50, self.drain_ui_events)
        self.recover_jobs()
        threading.Thread(target=self.transcription_worker, daemon=True).start()
        threading.Thread(target=self.prepare, daemon=True).start()
        self.tick()
        self.refresh_history()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=28, pady=(22, 12))
        tk.Label(header, text="Live captions", font=self.title_font, fg=INK, bg=BG).pack(
            side="left"
        )
        self.indicator = tk.Label(
            header, text="●  Preparing", font=self.small_font, fg=MUTED, bg=BG
        )
        self.indicator.pack(side="right")
        self.backend_button = tk.Button(
            header, text="", command=self.toggle_backend, font=self.small_font, fg=INK, bg=PANEL,
            activebackground=LINE, relief="flat", bd=0, padx=10, pady=4, cursor="hand2", highlightthickness=0,
        )
        self.backend_button.pack(side="right", padx=(0, 14))
        self.update_backend_button()

        caption_shell = tk.Frame(self.root, bg=LINE, padx=1, pady=1)
        caption_shell.pack(fill="both", expand=True, padx=28, pady=(0, 14))
        caption_panel = tk.Frame(caption_shell, bg=PANEL)
        caption_panel.pack(fill="both", expand=True)
        self.committed_caption = tk.Label(
            caption_panel,
            text="",
            font=self.small_font,
            fg=MUTED,
            bg=PANEL,
            justify="left",
            anchor="sw",
            wraplength=800,
            padx=24,
            pady=0,
        )
        self.committed_caption.pack(fill="x", padx=0, pady=(20, 0))
        self.caption = tk.Label(
            caption_panel,
            text="Preparing live captions…",
            font=self.caption_font,
            fg=INK,
            bg=PANEL,
            justify="left",
            anchor="w",
            wraplength=800,
            padx=24,
            pady=12,
        )
        self.caption.pack(fill="both", expand=True)
        self.caption_note = tk.Label(
            caption_panel,
            text="",
            font=self.tiny_font,
            fg=FAINT,
            bg=PANEL,
            anchor="w",
            padx=24,
            pady=0,
        )
        self.caption_note.pack(fill="x", pady=(0, 16))

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=28, pady=(0, 16))
        self.status_var = tk.StringVar(value="Preparing")
        tk.Label(
            controls,
            textvariable=self.status_var,
            font=self.status_font,
            fg=MUTED,
            bg=BG,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        self.action = tk.Button(
            controls,
            text="Start captions",
            command=self.toggle,
            font=self.button_font,
            fg="white",
            bg=BLUE,
            activeforeground="white",
            activebackground=BLUE_HOVER,
            relief="flat",
            bd=0,
            padx=24,
            pady=11,
            cursor="hand2",
            state="disabled",
            highlightthickness=0,
        )
        self.action.pack(side="right")

        history_shell = tk.Frame(self.root, bg=LINE, padx=1, pady=1)
        history_shell.pack(fill="x", padx=28, pady=(0, 22))
        history_panel = tk.Frame(history_shell, bg=PANEL)
        history_panel.pack(fill="both")
        history_header = tk.Frame(history_panel, bg=PANEL)
        history_header.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(
            history_header,
            text="Recent transcriptions",
            font=self.status_font,
            fg=INK,
            bg=PANEL,
        ).pack(side="left")
        self.history_summary = tk.Label(
            history_header, text="No sessions yet", font=self.tiny_font, fg=FAINT, bg=PANEL
        )
        self.history_summary.pack(side="right")
        self.history_rows = tk.Frame(history_panel, bg=PANEL)
        self.history_rows.pack(fill="x", padx=8, pady=(0, 8))

    def ui(self, callback) -> None:
        self.ui_events.put(callback)

    def update_backend_button(self) -> None:
        self.backend_button.configure(text=f"Backend: {BACKENDS[self.backend]}  ⇄")

    def toggle_backend(self) -> None:
        if self.running.is_set() or self.capture_busy.is_set():
            self.status_var.set("Stop captions before switching the backend")
            return
        self.backend = "ps5" if self.backend == "openai" else "openai"
        save_backend(self.backend)
        self.update_backend_button()
        self.model_state = "preparing"
        self.action.configure(state="disabled")
        self.refresh_indicator()
        threading.Thread(target=self.prepare, daemon=True).start()

    def drain_ui_events(self) -> None:
        try:
            for _ in range(100):
                try:
                    callback = self.ui_events.get_nowait()
                except queue.Empty:
                    break
                try:
                    callback()
                finally:
                    self.ui_events.task_done()
            if self.root.winfo_exists():
                self.root.after(50, self.drain_ui_events)
        except tk.TclError:
            pass

    def set_caption(self, text: str, color: str = INK) -> None:
        self.committed_caption.configure(text="")
        self.caption.configure(text=text, fg=color)
        self.set_caption_note("")

    def set_caption_note(self, text: str, color: str = FAINT) -> None:
        self.caption_note.configure(text=text, fg=color)

    def set_indicator(self, text: str, color: str) -> None:
        self.indicator.configure(text=f"●  {text}", fg=color)

    def prepare(self) -> None:
        try:
            load_dotenv(ENV_PATH, override=False)
            if self.backend == "ps5":
                self.ui(lambda: self.status_var.set("Starting the PS5 GPU caption server over Ethernet…"))
                ps5_backend.ensure_server()
            elif not os.getenv("OPENAI_API_KEY", "").strip():
                raise RuntimeError("OPENAI_API_KEY is missing from .env.")
            self.ui(self.ready)
        except Exception as error:
            self.model_state = "error"
            message = redact_error(error)
            self.ui(
                lambda: (
                    self.set_caption("Could not load the speech model", RED),
                    self.status_var.set(message[:120]),
                    self.set_indicator("Model unavailable", RED),
                )
            )

    def ready(self) -> None:
        self.model_state = "ready"
        try:
            input_name = str(sd.query_devices(kind="input")["name"])
        except Exception:
            input_name = "default microphone"
        self.set_caption("Press Start captions when you’re ready.", MUTED)
        self.status_var.set(
            f"Ready  •  {input_name}  •  {BACKENDS[self.backend]}  •  Live words refine after each sentence"
        )
        self.action.configure(state="normal")
        self.refresh_indicator()

    def refresh_indicator(self) -> None:
        if self.model_state == "preparing":
            self.set_indicator("Preparing", MUTED)
        elif self.model_state == "error":
            self.set_indicator("Model unavailable", RED)
        elif self.running.is_set():
            if self.live_client is not None and self.live_client.connected.is_set():
                self.set_indicator("Live audio streaming", RED)
            else:
                self.set_indicator("Connecting live captions", AMBER)
        elif self.capture_busy.is_set():
            self.set_indicator("Saving recording", BLUE)
        elif self.active_transcription is not None:
            self.set_indicator("Transcribing on the PS5" if self.backend == "ps5" else "Uploading to OpenAI", BLUE)
        elif self.queued_sessions:
            self.set_indicator("Transcription queued", AMBER)
        else:
            self.set_indicator("Ready", GREEN)

    def toggle(self) -> None:
        if self.capture_busy.is_set():
            return
        if self.running.is_set():
            self.stop_session()
        else:
            self.start_session()

    def unique_session_dir(self, now: datetime) -> Path:
        base = OUTPUT_ROOT / now.strftime("%Y-%m-%d_%H-%M-%S")
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = Path(f"{base}-{suffix}")
            suffix += 1
        return candidate

    def start_session(self) -> None:
        now = datetime.now().astimezone()
        self.session_token += 1
        token = self.session_token
        load_dotenv(ENV_PATH, override=False)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if self.backend == "openai" and not api_key:
            self.set_caption("OPENAI_API_KEY is missing from .env.", RED)
            return
        self.session_started_wall = now
        self.session_dir = self.unique_session_dir(now)
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.audio_path = self.session_dir / "recording.wav"
        update_metadata(
            self.session_dir,
            recorded_at=now.isoformat(timespec="seconds"),
            status="recording",
            error=None,
            sample_rate=RATE,
            channels=1,
            backend=self.backend,
            live_model=PS5_MODEL if self.backend == "ps5" else LIVE_MODEL,
            live_audio_uploaded=self.backend == "openai",
            final_model=PS5_MODEL if self.backend == "ps5" else API_MODEL,
            audio="recording.wav",
        )
        self.writer = wave.open(str(self.audio_path), "wb")
        self.writer.setnchannels(1)
        self.writer.setsampwidth(2)
        self.writer.setframerate(RATE)
        self.writer_queue = queue.Queue(maxsize=500)
        self.writer_thread = threading.Thread(target=self.write_audio, daemon=True)
        self.writer_thread.start()

        self.live_item_order = []
        self.live_partial = {}
        self.live_final = {}
        self.speech_threshold = 0.008
        self.latest_rms = 0.0
        self.last_voice_at = 0.0
        self.turn_has_speech = False
        self.last_commit_at = 0.0
        client_class = (
            (lambda **kw: ps5_backend.Ps5CaptionClient(rate=RATE, **kw))
            if self.backend == "ps5"
            else (lambda **kw: RealtimeCaptionClient(api_key=api_key, **kw))
        )
        self.live_client = client_class(
            on_delta=lambda item_id, delta, replace=False: self.ui(
                lambda: self.handle_live_delta(token, item_id, delta, replace)
            ),
            on_completed=lambda item_id, transcript: self.ui(
                lambda: self.handle_live_completed(token, item_id, transcript)
            ),
            on_status=lambda status: self.ui(
                lambda: self.handle_live_status(token, status)
            ),
            on_error=lambda message: self.ui(
                lambda: self.handle_live_error(token, message)
            ),
        )
        self.live_client.start()
        self.session_started = time.monotonic()
        self.running.set()
        self.action.configure(text="Stop & transcribe", bg=RED, activebackground="#BE3838")
        self.status_var.set("Connecting live captions and starting the microphone")
        self.set_caption("Listen for a moment…", MUTED)
        self.refresh_indicator()
        self.refresh_history()
        self.capture_thread = threading.Thread(
            target=self.capture_loop, args=(token,), daemon=True
        )
        self.capture_thread.start()

    def audio_callback(self, indata, _frames, _timing, flags) -> None:
        if not self.running.is_set():
            return
        if flags:
            self.ui(lambda: self.status_var.set(f"Audio status: {flags}"))
        block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        now = time.monotonic()
        try:
            self.writer_queue.put_nowait(block)
        except queue.Full:
            self.ui(lambda: self.status_var.set("Recording buffer fell behind"))
        if self.live_client is not None:
            self.live_client.append_audio(block)
        with self.audio_lock:
            self.latest_rms = rms
            if rms >= self.speech_threshold:
                self.last_voice_at = now
                self.turn_has_speech = True

    def write_audio(self) -> None:
        while True:
            block = self.writer_queue.get()
            if block is None:
                self.writer_queue.task_done()
                break
            pcm = np.clip(block, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype("<i2")
            if self.writer is not None:
                self.writer.writeframesraw(pcm.tobytes())
            self.writer_queue.task_done()

    def capture_loop(self, token: int) -> None:
        try:
            self.stream = sd.InputStream(
                device=DEVICE,
                channels=1,
                samplerate=RATE,
                dtype="float32",
                callback=self.audio_callback,
                blocksize=RATE // 10,
            )
            self.stream.start()
            noise_levels: list[float] = []
            started = time.monotonic()
            while self.running.is_set() and time.monotonic() - started < 1.2:
                time.sleep(0.1)
                with self.audio_lock:
                    noise_levels.append(self.latest_rms)
            baseline = float(np.median(noise_levels)) if noise_levels else 0.002
            self.speech_threshold = max(0.006, baseline * 3.2)
            self.ui(
                lambda: self.show_listening_prompt(token)
            )

            while self.running.is_set() and token == self.session_token:
                time.sleep(0.1)
                with self.audio_lock:
                    has_speech = self.turn_has_speech
                    last_voice = self.last_voice_at
                now = time.monotonic()
                if (
                    has_speech
                    and last_voice
                    and now - last_voice >= TURN_SILENCE_SECONDS
                    and now - self.last_commit_at >= TURN_SILENCE_SECONDS
                    and self.live_client is not None
                    and self.live_client.commit()
                ):
                    with self.audio_lock:
                        self.turn_has_speech = False
                    self.last_commit_at = now
        except Exception as error:
            if self.running.is_set() and token == self.session_token:
                self.running.clear()
                self.capture_busy.set()
                message = redact_error(error)
                threading.Thread(
                    target=self.finalize_capture_failure,
                    args=(token, message),
                    daemon=True,
                ).start()

    def show_listening_prompt(self, token: int) -> None:
        if token != self.session_token or not self.running.is_set():
            return
        if self.live_partial or self.live_final:
            return
        self.set_caption("Speak naturally. Words will appear as you talk.", MUTED)
        self.set_caption_note("Unfinished words may change after a short pause")

    def handle_live_status(self, token: int, status: str) -> None:
        if token != self.session_token or not self.running.is_set():
            return
        if status == "connected":
            self.set_indicator("Live audio streaming", RED)
            if not self.live_partial and not self.live_final:
                self.set_caption_note("Connected • unfinished words may change after a pause")
        elif status == "reconnecting":
            self.set_indicator("Reconnecting live captions", AMBER)
            self.set_caption_note("Reconnecting live words • recording continues locally", AMBER)

    def handle_live_error(self, token: int, message: str) -> None:
        if token != self.session_token or not self.running.is_set():
            return
        self.set_caption_note(
            f"Live words interrupted: {message[:110]} • recording continues", RED
        )

    def handle_live_delta(self, token: int, item_id: str, delta: str, replace: bool = False) -> None:
        if token != self.session_token or not self.running.is_set() or not delta:
            return
        if item_id in self.live_final:  # a late provisional guess for an already settled sentence
            return
        if item_id not in self.live_item_order:
            self.live_item_order.append(item_id)
        # OpenAI streams word deltas; the PS5 re-transcribes the whole turn and replaces the guess.
        self.live_partial[item_id] = delta if replace else self.live_partial.get(item_id, "") + delta
        self.render_live_caption()
        self.set_caption_note("Unfinished words may change after a short pause", FAINT)

    def handle_live_completed(self, token: int, item_id: str, transcript: str) -> None:
        if token != self.session_token:
            return
        if item_id not in self.live_item_order:
            self.live_item_order.append(item_id)
        provisional = self.live_partial.pop(item_id, "").strip()
        final_text = transcript.strip() or provisional
        if not final_text:
            return
        self.live_final[item_id] = final_text
        self.render_live_caption()
        corrected = bool(
            provisional
            and normalized_words(provisional) != normalized_words(final_text)
        )
        note = "Sentence refined" if corrected else "Sentence settled"
        color = BLUE if corrected else GREEN
        self.set_caption_note(note, color)
        if self.caption_note_after_id is not None:
            try:
                self.root.after_cancel(self.caption_note_after_id)
            except tk.TclError:
                pass
        self.caption_note_after_id = self.root.after(
            1600, lambda: self.restore_live_note(token)
        )

    def restore_live_note(self, token: int) -> None:
        self.caption_note_after_id = None
        if token == self.session_token and self.running.is_set():
            self.set_caption_note("Listening for the next words", FAINT)

    def render_live_caption(self) -> None:
        finalized = [
            self.live_final[item_id]
            for item_id in self.live_item_order
            if item_id in self.live_final
        ]
        provisional = " ".join(
            self.live_partial[item_id].strip()
            for item_id in self.live_item_order
            if item_id in self.live_partial and self.live_partial[item_id].strip()
        )
        if provisional:
            self.committed_caption.configure(text=" ".join(finalized[-2:])[-420:])
            self.caption.configure(text=provisional[-520:], fg=PROVISIONAL)
        elif finalized:
            self.committed_caption.configure(text=" ".join(finalized[-3:-1])[-420:])
            self.caption.configure(text=finalized[-1][-520:], fg=INK)

    def close_capture_resources(self) -> None:
        with self.capture_close_lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            if self.writer is not None:
                self.writer_queue.put(None)
                if self.writer_thread is not None:
                    self.writer_thread.join()
                self.writer.close()
                self.writer = None
                self.writer_thread = None

    def close_live_captions(self) -> None:
        client = self.live_client
        if client is None:
            return
        with self.audio_lock:
            commit_open_turn = self.turn_has_speech
            self.turn_has_speech = False
        client.finish(commit_open_turn=commit_open_turn)
        self.live_client = None

    def stop_session(self) -> None:
        token = self.session_token
        self.running.clear()
        self.capture_busy.set()
        self.action.configure(text="Saving recording…", state="disabled", bg=BLUE)
        self.status_var.set("Finishing the recording")
        self.set_caption("Adding this session to the transcription queue…", MUTED)
        self.refresh_indicator()
        threading.Thread(target=self.finalize_recording, args=(token,), daemon=True).start()

    def finalize_recording(self, token: int) -> None:
        self.close_capture_resources()
        self.close_live_captions()
        if self.capture_thread is not None and self.capture_thread is not threading.current_thread():
            self.capture_thread.join(timeout=15)
        if token != self.session_token or self.session_dir is None or self.audio_path is None:
            return
        try:
            duration = audio_duration(self.audio_path)
            update_metadata(
                self.session_dir,
                status="queued",
                duration_seconds=duration,
                queued_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                error=None,
            )
            session_dir = self.session_dir
            self.enqueue_session(session_dir, metadata_already_queued=True)
            self.ui(lambda: self.recording_queued(session_dir))
        except Exception as error:
            message = redact_error(error)
            update_metadata(
                self.session_dir,
                status="failed",
                error=message,
                failed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            self.ui(lambda: self.capture_failed(message))

    def finalize_capture_failure(self, token: int, message: str) -> None:
        self.close_capture_resources()
        self.close_live_captions()
        if token != self.session_token or self.session_dir is None:
            return
        update_metadata(
            self.session_dir,
            status="failed",
            error=f"Recording failed: {message}",
            failed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.ui(lambda: self.capture_failed(message))

    def recording_queued(self, session_dir: Path) -> None:
        self.capture_busy.clear()
        self.status_var.set("Recording saved and queued for transcription")
        self.set_caption("Recording saved. You can start another session while it transcribes.", MUTED)
        self.action.configure(
            text="Start new session", state="normal", bg=BLUE, activebackground=BLUE_HOVER
        )
        self.refresh_indicator()
        self.refresh_history()
        self.maybe_close()

    def capture_failed(self, message: str) -> None:
        self.capture_busy.clear()
        self.status_var.set("Recording stopped; see the failed task below")
        self.set_caption(message[:300], RED)
        self.action.configure(
            text="Start new session", state="normal", bg=BLUE, activebackground=BLUE_HOVER
        )
        self.refresh_indicator()
        self.refresh_history()
        self.maybe_close()

    def recover_jobs(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for session_dir in sorted(
            (path for path in OUTPUT_ROOT.iterdir() if path.is_dir()), reverse=False
        ):
            metadata = read_metadata(session_dir)
            status = metadata.get("status")
            if status == "queued":
                self.enqueue_session(session_dir, metadata_already_queued=True)
            elif status in {"recording", "transcribing"}:
                update_metadata(
                    session_dir,
                    status="failed",
                    error="The app closed before this task finished. The saved recording can be retried.",
                    failed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )

    def enqueue_session(self, session_dir: Path, metadata_already_queued: bool = False) -> bool:
        session_dir = session_dir.resolve()
        with self.queue_lock:
            if session_dir in self.queued_sessions or session_dir == self.active_transcription:
                return False
            self.queued_sessions.add(session_dir)
        if not metadata_already_queued:
            update_metadata(
                session_dir,
                status="queued",
                error=None,
                queued_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        self.task_queue.put(session_dir)
        self.ui(self.refresh_history)
        self.ui(self.refresh_indicator)
        return True

    def transcription_worker(self) -> None:
        while not self.shutdown.is_set():
            session_dir = self.task_queue.get()
            if session_dir is None:
                self.task_queue.task_done()
                break
            self.active_transcription = session_dir
            update_metadata(
                session_dir,
                status="transcribing",
                started_transcribing_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                error=None,
            )
            self.ui(self.refresh_history)
            self.ui(self.refresh_indicator)
            try:
                files = transcribe_session(session_dir)
                update_metadata(
                    session_dir,
                    **files,
                    status="completed",
                    error=None,
                    completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            except Exception as error:
                update_metadata(
                    session_dir,
                    status="failed",
                    error=redact_error(error),
                    failed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            finally:
                with self.queue_lock:
                    self.queued_sessions.discard(session_dir)
                self.active_transcription = None
                self.task_queue.task_done()
                self.ui(self.transcription_finished)

    def transcription_finished(self) -> None:
        self.refresh_history()
        self.refresh_indicator()
        self.maybe_close()

    def retry_session(self, session_dir: Path) -> None:
        if not (session_dir / "recording.wav").exists():
            self.status_var.set("Cannot retry: recording.wav is missing")
            return
        if self.enqueue_session(session_dir):
            self.status_var.set("Transcription queued again")
            self.refresh_history()

    def copy_transcript(self, session_dir: Path) -> None:
        path = session_dir / "transcript.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
            self.status_var.set("Transcript copied")
        except OSError as error:
            self.status_var.set(f"Could not copy transcript: {redact_error(error)[:80]}")

    def open_session(self, session_dir: Path) -> None:
        subprocess.run(["open", str(session_dir)], check=False)

    def session_dirs(self) -> list[Path]:
        if not OUTPUT_ROOT.exists():
            return []
        return sorted(
            (path for path in OUTPUT_ROOT.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )

    def friendly_session_name(self, session_dir: Path, metadata: dict) -> str:
        value = metadata.get("recorded_at")
        try:
            moment = datetime.fromisoformat(str(value))
            return moment.strftime("%b %-d, %-I:%M %p")
        except (TypeError, ValueError):
            return session_dir.name.replace("_", "  ")

    def add_history_row(self, session_dir: Path, metadata: dict, row_index: int) -> None:
        status = str(metadata.get("status", "unknown"))
        labels = {
            "recording": ("Recording", RED),
            "queued": ("Queued", AMBER),
            "transcribing": ("Transcribing", BLUE),
            "completed": ("Complete", GREEN),
            "failed": ("Failed", RED),
        }
        status_label, status_color = labels.get(status, (status.title(), MUTED))
        row = tk.Frame(self.history_rows, bg=PANEL, height=40)
        row.pack(fill="x")
        row.pack_propagate(False)
        if row_index:
            tk.Frame(row, bg=LINE, height=1).pack(fill="x", side="top")

        tk.Label(
            row,
            text=self.friendly_session_name(session_dir, metadata),
            font=self.small_font,
            fg=INK,
            bg=PANEL,
            width=19,
            anchor="w",
        ).pack(side="left", padx=(8, 4))
        tk.Label(
            row,
            text=f"●  {status_label}",
            font=self.tiny_font,
            fg=status_color,
            bg=PANEL,
            width=14,
            anchor="w",
        ).pack(side="left")

        if status == "failed":
            detail = str(metadata.get("error") or "Transcription failed")
        elif status == "queued":
            detail = "Waiting in queue"
        elif status == "transcribing":
            detail = "Sending saved audio to OpenAI"
        elif status == "completed":
            detail = timestamp(float(metadata.get("duration_seconds") or 0))
        else:
            detail = ""
        tk.Label(
            row,
            text=detail[:58],
            font=self.tiny_font,
            fg=MUTED,
            bg=PANEL,
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(4, 8))

        if status == "completed" and (session_dir / "transcript.txt").exists():
            self.history_button(
                row, "Copy", lambda path=session_dir: self.copy_transcript(path), BLUE
            ).pack(side="right", padx=(2, 8))
        elif status == "failed" and (session_dir / "recording.wav").exists():
            self.history_button(
                row, "Retry", lambda path=session_dir: self.retry_session(path), RED
            ).pack(side="right", padx=(2, 8))
        self.history_button(
            row, "Open", lambda path=session_dir: self.open_session(path), MUTED
        ).pack(side="right", padx=2)

    def history_button(self, parent: tk.Widget, text: str, command, color: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=self.tiny_font,
            fg=color,
            bg=PANEL,
            activeforeground=color,
            activebackground=PANEL,
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
        )

    def refresh_history(self) -> None:
        try:
            for child in self.history_rows.winfo_children():
                child.destroy()
            sessions = self.session_dirs()
            records = [(path, read_metadata(path)) for path in sessions]
            if not records:
                tk.Label(
                    self.history_rows,
                    text="Completed and failed transcripts will appear here.",
                    font=self.small_font,
                    fg=FAINT,
                    bg=PANEL,
                    anchor="w",
                ).pack(fill="x", padx=8, pady=(2, 10))
                self.history_summary.configure(text="No sessions yet")
            else:
                counts = {"queued": 0, "transcribing": 0, "failed": 0}
                for _, metadata in records:
                    status = str(metadata.get("status", ""))
                    if status in counts:
                        counts[status] += 1
                pending = counts["queued"] + counts["transcribing"]
                summary_parts = []
                if pending:
                    summary_parts.append(f"{pending} active")
                if counts["failed"]:
                    summary_parts.append(f"{counts['failed']} failed")
                if not summary_parts:
                    summary_parts.append(f"{len(records)} saved")
                self.history_summary.configure(text="  •  ".join(summary_parts))
                for index, (session_dir, metadata) in enumerate(records[:HISTORY_LIMIT]):
                    self.add_history_row(session_dir, metadata, index)
            self.refresh_indicator()
            if self.root.winfo_exists() and self.history_after_id is None:
                self.history_after_id = self.root.after(1500, self.scheduled_history_refresh)
        except tk.TclError:
            pass

    def scheduled_history_refresh(self) -> None:
        self.history_after_id = None
        self.refresh_history()

    def tick(self) -> None:
        if self.running.is_set():
            elapsed = timestamp(time.monotonic() - self.session_started)
            mode = "live words connected"
            if self.live_client is None or not self.live_client.connected.is_set():
                mode = "connecting live words"
            self.status_var.set(f"Recording {elapsed}  •  {mode}")
        try:
            if self.root.winfo_exists():
                self.root.after(1000, self.tick)
        except tk.TclError:
            pass

    def pending_work(self) -> bool:
        return bool(self.active_transcription or self.queued_sessions or self.capture_busy.is_set())

    def destroy(self) -> None:
        self.shutdown.set()
        threading.Thread(target=ps5_backend.close_tunnel, daemon=True).start()  # tunnel only; server stays for dictation
        self.root.after(400, self.root.destroy)

    def maybe_close(self) -> None:
        if self.closing_requested and not self.running.is_set() and not self.pending_work():
            self.destroy()

    def request_close(self) -> None:
        if self.running.is_set():
            self.closing_requested = True
            self.stop_session()
            return
        if self.pending_work():
            self.closing_requested = True
            self.status_var.set("Closing after queued transcripts finish")
            return
        self.destroy()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    CaptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
