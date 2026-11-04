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
_whisper_lock = threading.Lock()  # one whisper.cpp context, used by one thread at a time


def transcribe(audio):
    """Turn 16 kHz mono float32 audio into text. Loads the model on first use."""
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from pywhispercpp.model import Model

            _whisper = Model(WHISPER_MODEL, print_progress=False, print_realtime=False)
        segments = _whisper.transcribe(audio)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return "" if text == "[BLANK_AUDIO]" else text


def _post(path, payload):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req)


def warm_up():
    """Load both models before the first click. A cold Ollama load can take over a minute."""
    _post("/api/generate", {"model": LLM_MODEL, "keep_alive": "30m"}).close()
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))


def stream_answer(question):
    """Yield answer text piece by piece from the local Ollama server."""
    with _post("/api/chat", {
        "model": LLM_MODEL,
        "stream": True,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }) as resp:
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
        self.suggest_btn = ttk.Button(bar, text="Suggest Answer", command=self.suggest)
        for btn in (self.start_btn, self.stop_btn, self.suggest_btn):
            btn.pack(side="left", padx=(0, 6))
        self.status = ttk.Label(bar, text="Press Start and ask a question.", foreground="#555")
        self.status.pack(side="right")

        ttk.Label(root, text="Transcribed question (you can edit it)").pack(anchor="w", padx=12)
        self.question = tk.Text(root, height=5, wrap="word", font=("Segoe UI", 11), padx=8, pady=6)
        self.question.pack(fill="x", padx=12, pady=(2, 10))

        ttk.Label(root, text=f"Suggested answer ({LLM_MODEL})").pack(anchor="w", padx=12)
        self.answer = tk.Text(root, wrap="word", font=("Segoe UI", 11), padx=8, pady=6,
                              state="disabled", background="#f7f7f5")
        self.answer.pack(fill="both", expand=True, padx=12, pady=(2, 12))

        self._set_buttons(start=True, stop=False, suggest=True)
        root.after(50, self._drain_events)
        self.status.config(text="Loading models...")
        self._in_background(lambda: (warm_up(), self.events.put(("status", "Ready. Press Start."))))

    def _set_buttons(self, start, stop, suggest):
        for btn, on in ((self.start_btn, start), (self.stop_btn, stop), (self.suggest_btn, suggest)):
            btn.state(["!disabled"] if on else ["disabled"])

    def _append_answer(self, text):
        self.answer.config(state="normal")
        self.answer.insert("end", text)
        self.answer.see("end")
        self.answer.config(state="disabled")

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
                self.status.config(text="Ready. Press Suggest Answer." if value
                                   else "Didn't catch any speech. Try again.")
                self._set_buttons(start=True, stop=False, suggest=True)
            elif kind == "token":
                self._append_answer(value)
            elif kind == "done":
                self.status.config(text="Done.")
                self._set_buttons(start=True, stop=False, suggest=True)
            elif kind == "error":
                self.status.config(text=f"Error: {value}")
                self._set_buttons(start=True, stop=False, suggest=True)
        self.root.after(50, self._drain_events)

    def start(self):
        try:
            self.recorder.start()
        except Exception as e:
            self.status.config(text=f"Microphone error: {e}")
            return
        self._set_buttons(start=False, stop=True, suggest=False)
        self.status.config(text="Listening...")

    def stop(self):
        audio = self.recorder.stop()
        self._set_buttons(start=False, stop=False, suggest=False)
        self.status.config(text="Transcribing...")
        self._in_background(lambda: self.events.put(("question", transcribe(audio))))

    def suggest(self):
        question = self.question.get("1.0", "end").strip()
        if not question:
            self.status.config(text="Nothing to answer yet. Record or type a question.")
            return
        self.answer.config(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.config(state="disabled")
        self._set_buttons(start=False, stop=False, suggest=False)
        self.status.config(text="Thinking...")

        def work():
            for piece in stream_answer(question):
                self.events.put(("token", piece))
            self.events.put(("done", None))

        self._in_background(work)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
