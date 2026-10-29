# Live captions

A small macOS app that shows responsive live captions, records each session, and creates a final timestamped transcript.

## How it works

- Live microphone audio streams to OpenAI `gpt-live-transcribe`, which returns words progressively and a finalized sentence after each short pause.
- Provisional words appear in blue-gray. The finalized sentence replaces them in dark text, correcting earlier wording when needed.
- The microphone recording is saved locally as WAV and M4A.
- After a session stops, its compressed audio is queued and uploaded to the OpenAI `whisper-1` transcription API.
- The history panel shows queued, active, completed, and failed jobs. Completed transcripts can be copied in one click; failed jobs can be retried from their saved recording.
- The app records the current default macOS microphone. It does not independently capture system audio.

## Run

Double-click `run.command`, or run:

```sh
./run.command
```

The API key is loaded from `.env`. That file is excluded from Git. Live and post-session API transcription are billed through the OpenAI project associated with the key.

## Saved sessions

Each session is stored under `sessions/YYYY-MM-DD_HH-MM-SS/` with:

- `recording.wav`: original microphone recording
- `recording.m4a`: compressed recording sent to the API, or the source for upload parts
- `transcript.txt`: copy-ready transcript
- `transcript.md`: timestamped transcript
- `whisper_api_response.json`: merged raw API responses
- `session.json`: task status and metadata used by the history panel

If transcription fails, the recording is retained and the session remains retryable.

## Backends: OpenAI or PS5 GPU (2026-10-29)

The header button switches the backend (saved in `settings.json`; default from `CAPTION_BACKEND` in `.env`):

- **OpenAI**: live words from `gpt-live-transcribe`, final transcript from `whisper-1` (needs `OPENAI_API_KEY`).
- **PS5 GPU · Ethernet**: whisper.cpp `ggml-small` on the PS5 Linux GPU (Vulkan). `ps5_backend.py` starts
  `ps5-whisper-server` on the remote host over SSH and tunnels it to `127.0.0.1:18178` through the cable
  (`PS5_HOST`, a direct Ethernet link in the reference setup), so audio never leaves the two machines and no API key is needed. Live words are
  re-transcribed about every 0.7 s and settle after each pause; the final transcript runs on the PS5 too.
  On the reference host the server is stopped whenever the GPU is reserved for local LLMs.

`CAPTION_LANGUAGE` (default `en`, or `auto`) sets the language for the PS5 backend.
Sessions remember their backend, so a retry uses the same one.

## Dictation

Push-to-talk dictation into any app (hold Right Option) lives in its own project:
[live-dictation](https://github.com/LEE-CHENYU/live-dictation). It uses the same remote GPU server.

## Setup

```sh
conda create -n live_captions python=3.12 -y
conda run -n live_captions pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY and/or the PS5_* SSH settings
./run.command
```

The remote backend expects a Linux host with a Vulkan build of whisper.cpp `whisper-server` running as the
systemd unit `ps5-whisper-server` on its `127.0.0.1:8178`, started with `-nc` (no carried-over context:
without it the server reuses earlier text and live captions "predict" speech). Each request also sends
`no_context=true` and an explicit `prompt`, because whisper-server keeps request parameters between calls.
