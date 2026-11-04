"""Cuecard: hear a question, get a suggested answer. Speech-to-text and LLM both run locally."""
import json
import os
import queue
import threading
import tkinter as tk
import urllib.request
from tkinter import ttk

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper.cpp expects 16 kHz mono float32

# Handy (https://handy.computer) already keeps a whisper.cpp model on disk, so reuse it
# instead of downloading another copy. Falls back to pywhispercpp's auto-download.
HANDY_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.pais.handy", "models", "ggml-small.bin")
WHISPER_MODEL = os.environ.get("CUECARD_WHISPER_MODEL") or (
    HANDY_MODEL if os.path.exists(HANDY_MODEL) else "base.en"
)
OLLAMA_URL = os.environ.get("CUECARD_OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("CUECARD_LLM", "llama3.2:3b")
SYSTEM_PROMPT = (
    "You help a candidate in a live interview. You get a rough speech transcript of the "
    "interviewer's question. Reply with an answer the candidate can say out loud: first person, "
    "3 to 6 sentences, concrete, no preamble, no markdown. If the transcript holds several "
    "questions, answer the last one."
)


class Recorder:
    """Collects microphone audio between start() and stop()."""

    def __init__(self):
        self._chunks = []
        self._stream = None

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._on_audio
        )
        self._stream.start()

    def _on_audio(self, data, frames, time, status):
        self._chunks.append(data.copy())

    def stop(self):
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)[:, 0]


_whisper = None


def transcribe(audio):
    """Turn 16 kHz mono float32 audio into text. Loads the model on first use."""
    global _whisper
    if _whisper is None:
        from pywhispercpp.model import Model

        _whisper = Model(WHISPER_MODEL, print_progress=False, print_realtime=False)
    text = " ".join(seg.text.strip() for seg in _whisper.transcribe(audio)).strip()
    return "" if text == "[BLANK_AUDIO]" else text


def stream_answer(question):
    """Yield answer text piece by piece from the local Ollama server."""
    body = json.dumps({
        "model": LLM_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        yield from parse_chunks(resp)


def parse_chunks(lines):
    """Ollama streams one JSON object per line; pull out the text of each."""
    for line in lines:
        if not line.strip():
            continue
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        yield msg.get("message", {}).get("content", "")
        if msg.get("done"):
            return


class App:
    """Tk is single-threaded: slow work runs on worker threads and reports back through a queue."""

    def __init__(self, root):
        self.root = root
        self.recorder = Recorder()
        self.events = queue.Queue()

        root.title("Cuecard")
        root.geometry("680x600")
        root.minsize(480, 400)

        bar = ttk.Frame(root, padding=(12, 12, 12, 6))
        bar.pack(fill="x")
        self.start_btn = ttk.Button(bar, text="Start", command=self.start)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop)
        for btn in (self.start_btn, self.stop_btn):
            btn.pack(side="left", padx=(0, 6))
        self.status = ttk.Label(bar, text="Press Start and ask a question.", foreground="#555")
        self.status.pack(side="right")

        ttk.Label(root, text="Transcribed question (you can edit it)").pack(anchor="w", padx=12)
        self.question = tk.Text(root, height=5, wrap="word", font=("Segoe UI", 11), padx=8, pady=6)
        self.question.pack(fill="x", padx=12, pady=(2, 10))

        self._set_buttons(start=True, stop=False)
        root.after(50, self._drain_events)

    def _set_buttons(self, start, stop):
        self.start_btn.state(["!disabled"] if start else ["disabled"])
        self.stop_btn.state(["!disabled"] if stop else ["disabled"])

    def _in_background(self, work):
        def run():
            try:
                work()
            except Exception as e:  # surface any failure in the window instead of a dead thread
                self.events.put(("error", str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _drain_events(self):
        while not self.events.empty():
            kind, value = self.events.get()
            if kind == "status":
                self.status.config(text=value)
            elif kind == "question":
                self.question.delete("1.0", "end")
                self.question.insert("1.0", value)
                self.status.config(text="Done." if value else "Didn't catch any speech. Try again.")
                self._set_buttons(start=True, stop=False)
            elif kind == "error":
                self.status.config(text=f"Error: {value}")
                self._set_buttons(start=True, stop=False)
        self.root.after(50, self._drain_events)

    def start(self):
        try:
            self.recorder.start()
        except Exception as e:
            self.status.config(text=f"Microphone error: {e}")
            return
        self._set_buttons(start=False, stop=True)
        self.status.config(text="Listening...")

    def stop(self):
        audio = self.recorder.stop()
        self._set_buttons(start=False, stop=False)
        self.status.config(text="Transcribing...")
        self._in_background(lambda: self.events.put(("question", transcribe(audio))))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
