"""Cuecard: hear a question, get a suggested answer. Speech-to-text and LLM both run locally."""
import os

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper.cpp expects 16 kHz mono float32

# Handy (https://handy.computer) already keeps a whisper.cpp model on disk, so reuse it
# instead of downloading another copy. Falls back to pywhispercpp's auto-download.
HANDY_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.pais.handy", "models", "ggml-small.bin")
WHISPER_MODEL = os.environ.get("CUECARD_WHISPER_MODEL") or (
    HANDY_MODEL if os.path.exists(HANDY_MODEL) else "base.en"
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
