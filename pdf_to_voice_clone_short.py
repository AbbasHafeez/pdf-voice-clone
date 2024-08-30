#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → Audiobook (XTTS v2, CPU, Sequential)

- XTTS v2 only (multilingual + optional voice cloning).
- Auto-detect page count; choose full doc / page range / first N words.
- Progress callback usable from Streamlit.
- MP3 export if FFmpeg present, else WAV fallback (no crash).
- Converts reference audio/video to 16 kHz mono WAV.
- No multiprocessing, no threading — clean & predictable.

pip install TTS pydub PyMuPDF
(Recommended) Install FFmpeg for MP3 and non-WAV reference formats.
"""

from __future__ import annotations
import os, re, uuid, argparse, tempfile, shutil
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Torchaudio's I/O functions now support .*backend dispatch.*",
    category=UserWarning,
)

import fitz  # PyMuPDF
from pydub import AudioSegment
from pydub.utils import which as which_ffmpeg
from TTS.api import TTS

# --------------------------- Logging ---------------------------

def _debug(msg: str):
    print(f"[INFO] {msg}")

def _fail(msg: str):
    raise SystemExit(f"[ERROR] {msg}")

# --------------------------- FFmpeg helpers ---------------------------

def have_ffmpeg() -> bool:
    return which_ffmpeg("ffmpeg") is not None

# --------------------------- PDF helpers ---------------------------

def get_pdf_meta(pdf_path: Path) -> Tuple[int, str]:
    """Return (page_count, title)"""
    with fitz.open(pdf_path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        return doc.page_count, title

def extract_text_from_pdf(pdf_path: Path, start_page: Optional[int], end_page: Optional[int]) -> str:
    """
    Extract plain text from page range [start_page, end_page], 1-based inclusive.
    If start/end are None → entire doc. Clamps to [1, n] and swaps if inverted.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        _fail(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        sp = 1 if not start_page else max(1, min(start_page, n))
        ep = n if not end_page else max(1, min(end_page, n))
        if ep < sp:
            sp, ep = ep, sp
        _debug(f"Extracting pages {sp}..{ep} of {n}")
        chunks = []
        for i in range(sp - 1, ep):
            page = doc.load_page(i)
            chunks.append(page.get_text("text"))
        text = "\n".join(chunks)

    # Normalize whitespace a bit
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# --------------------------- Text chunking ---------------------------

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!۔])\s+")  # include Urdu full stop (۔)

def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def pack_sentences(sentences: List[str], max_chars: int = 240) -> List[str]:
    """Greedy packing of sentences into chunks within max_chars."""
    chunks: List[str] = []
    buf: List[str] = []
    total = 0
    for s in sentences:
        s_len = len(s) + (1 if buf else 0)
        if total + s_len <= max_chars:
            buf.append(s)
            total += s_len
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf = [s]
            total = len(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def slice_text_first_n_words(text: str, n_words: int) -> str:
    words = text.split()
    if not words:
        return ""
    n = max(1, int(n_words))
    return " ".join(words[:n])

# --------------------------- Audio helpers ---------------------------

def ensure_wav_mono_16k(input_path: Path, work_dir: Path) -> Path:
    """
    Convert any audio/video to 16kHz mono WAV.
    If FFmpeg is missing and input isn't WAV, raise a helpful error.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        _fail(f"Voice reference not found: {input_path}")

    ext = input_path.suffix.lower()
    out = work_dir / f"voice_{uuid.uuid4().hex}.wav"

    _debug(f"Normalizing reference -> {out.name} (16k mono wav)")
    # Use pure-wav loader if possible to avoid FFmpeg requirement
    if ext == ".wav":
        audio = AudioSegment.from_wav(input_path)
    else:
        if not have_ffmpeg():
            raise RuntimeError("FFmpeg is not installed. Install it or upload a .wav reference file.")
        audio = AudioSegment.from_file(input_path)  # needs FFmpeg for non-wav types

    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(out, format="wav")
    return out

# --------------------------- XTTS v2 (singleton) ---------------------------

_TTS: Optional[TTS] = None
_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

def _get_tts() -> TTS:
    global _TTS
    if _TTS is None:
        _debug(f"Loading Coqui TTS (CPU): {_MODEL_ID}")
        _TTS = TTS(_MODEL_ID).to("cpu")
    return _TTS

# --------------------------- Synthesis (sequential) ---------------------------

ProgressFn = Optional[Callable[[int, int, str], None]]

def synthesize_chunks_to_wavs(
    chunks: List[str],
    lang: str,
    speaker_wav: Optional[Path],
    work_dir: Path,
    progress_fn: ProgressFn = None,
) -> List[Path]:
    """
    Generate per-chunk WAV files with XTTS v2 voice cloning (sequential).
    """
    tts = _get_tts()
    out_paths: List[Path] = []
    total = len(chunks)
    for idx, text in enumerate(chunks, 1):
        out_wav = work_dir / f"chunk_{idx:04d}.wav"
        preview = text[:64] + ("..." if len(text) > 64 else "")
        if progress_fn:
            progress_fn(idx, total, f"start: {preview}")
        if speaker_wav is not None:
            tts.tts_to_file(text=text, file_path=str(out_wav),
                            speaker_wav=str(speaker_wav), language=lang)
        else:
            tts.tts_to_file(text=text, file_path=str(out_wav), language=lang)
        if progress_fn:
            progress_fn(idx, total, f"done: {preview}")
        out_paths.append(out_wav)
    return out_paths

def concat_wavs_to_audio(wavs: List[Path], silence_ms: int, out_path: Path) -> Path:
    """
    Append WAVs with silence and export as MP3 if FFmpeg exists, else WAV.
    """
    if not wavs:
        _fail("No audio chunks to concatenate.")
    export_mp3 = have_ffmpeg()
    ext = ".mp3" if export_mp3 else ".wav"
    out = out_path.with_suffix(ext)

    _debug(f"Concatenating {len(wavs)} chunks → {out.name}")
    gap = AudioSegment.silent(duration=max(0, int(silence_ms)))
    final = AudioSegment.silent(duration=0)
    for w in wavs:
        seg = AudioSegment.from_wav(w)
        final += seg + gap
    if export_mp3:
        final.export(out, format="mp3")
    else:
        final.export(out, format="wav")
    return out

# --------------------------- Orchestrator ---------------------------

def convert_pdf_to_cloned_audiobook(
    pdf_path: str | Path,
    voice_ref_path: Optional[str | Path],
    out_path: str | Path,
    *,
    # scope
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    word_limit: Optional[int] = None,
    # synthesis
    lang: str = "en",
    max_chars: int = 240,
    silence_ms: int = 300,
    # ui
    progress_fn: ProgressFn = None,
) -> Path:
    """
    End-to-end: PDF → text (scoped) → chunk → XTTS (with optional ref voice) → audio file
    Returns final file path (.mp3 if FFmpeg, else .wav)
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)

    # Work scratch directory
    work_root = Path(tempfile.mkdtemp(prefix="pdf2voice_"))
    work_audio = work_root / "audio"
    work_audio.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Extract text
        text = extract_text_from_pdf(pdf_path, start_page, end_page)
        if word_limit:
            text = slice_text_first_n_words(text, int(word_limit))
        if not text.strip():
            _fail("No extractable text in the selected scope.")

        # 2) Chunk
        sentences = split_into_sentences(text)
        chunks = pack_sentences(sentences, max_chars=max_chars)
        if not chunks:
            _fail("Chunking produced no items (check max_chars or input text).")
        _debug(f"Will synthesize {len(chunks)} chunks (max_chars={max_chars}).")

        # 3) Normalize reference voice (optional)
        speaker_wav: Optional[Path] = None
        if voice_ref_path:
            speaker_wav = ensure_wav_mono_16k(Path(voice_ref_path), work_audio)

        # 4) TTS per chunk (sequential)
        wavs = synthesize_chunks_to_wavs(
            chunks=chunks,
            lang=lang,
            speaker_wav=speaker_wav,
            work_dir=work_audio,
            progress_fn=progress_fn,
        )

        # 5) Concat to final
        final = concat_wavs_to_audio(wavs, silence_ms=silence_ms, out_path=out_path)

        # 6) Move to requested path (respect chosen name but keep actual ext)
        final.parent.mkdir(parents=True, exist_ok=True)
        target = out_path.with_suffix(final.suffix)
        shutil.move(str(final), str(target))
        _debug(f"Done → {target}")
        return target

    finally:
        try:
            shutil.rmtree(work_root, ignore_errors=True)
        except Exception:
            pass

# --------------------------- CLI ---------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF → Voice-Cloned Audiobook (XTTS v2, Sequential)")
    p.add_argument("--pdf", required=True, help="Path to PDF file")
    p.add_argument("--voice", help="Reference voice file (mp3/wav/mp4/etc.)", default=None)
    p.add_argument("--out", required=True, help="Output file name (mp3/wav decided automatically)")
    p.add_argument("--lang", default="en", help="Language code for XTTS (e.g., en, ur)")
    p.add_argument("--max_chars", type=int, default=240, help="Max characters per chunk")
    p.add_argument("--silence_ms", type=int, default=300, help="Silence inserted between chunks (ms)")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--full", action="store_true", help="Use full document")
    scope.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"),
                       help="Page range 1-based inclusive")
    scope.add_argument("--first_words", type=int, help="Use only the first N words")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    start = end = None
    if args.pages:
        start, end = args.pages
    word_limit = args.first_words if args.first_words and not args.full else None

    convert_pdf_to_cloned_audiobook(
        pdf_path=args.pdf,
        voice_ref_path=args.voice,
        out_path=args.out,
        start_page=start if not args.full else None,
        end_page=end if not args.full else None,
        word_limit=word_limit,
        lang=args.lang,
        max_chars=args.max_chars,
        silence_ms=args.silence_ms,
        progress_fn=None,
    )
