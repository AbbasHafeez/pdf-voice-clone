from pathlib import Path
from typing import List
from pypdf import PdfReader
from pydub import AudioSegment
from TTS.api import TTS
from nltk.tokenize import sent_tokenize
import nltk, tempfile, sys

# ===== Defaults (Streamlit overrides these) =====
PDF_PATH = "life_3_0.pdf"
VOICE_REF = "file.mp3"
OUT_FILE = "audiobook_in_my_voice.mp3"
LANG = "en"
START_PAGE = 0
END_PAGE = 0
MAX_CHARS = 280           # a bit larger => fewer calls => faster
ADD_SIL_MS = 150          # less padding between chunks => faster overall

# --- NLTK tokenizer (one-time) ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# --- Cache the XTTS model globally so Streamlit reruns don't reload it ---
_TTS = None
_DEVICE = "cpu"

def _get_device():
    global _DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "cuda"
            torch.set_float32_matmul_precision("high")
        else:
            _DEVICE = "cpu"
            torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE

def get_tts():
    """Load XTTS once per process; reuse for all subsequent calls."""
    global _TTS
    if _TTS is not None:
        return _TTS
    dev = _get_device()
    _TTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    return _TTS

def _extract_text(pdf: Path, a: int, b: int) -> str:
    r = PdfReader(str(pdf))
    b = min(b, len(r.pages) - 1)
    if a < 0 or a > b:
        raise SystemExit("[ERROR] Bad page range")
    txt = "\n".join((r.pages[i].extract_text() or "") for i in range(a, b + 1)).strip()
    if not txt:
        raise SystemExit("[ERROR] No extractable text (OCR may be required)")
    return txt

def _chunks(s: str, lim: int) -> List[str]:
    out, buf = [], ""
    for sent in [x.strip() for x in sent_tokenize(s) if x.strip()]:
        if len(sent) > lim:
            if buf: out.append(buf); buf = ""
            cur = ""
            for w in sent.split():
                nxt = (cur + " " + w).strip()
                if len(nxt) > lim:
                    if cur: out.append(cur)
                    cur = w
                else:
                    cur = nxt
            if cur: out.append(cur)
            continue
        nxt = (buf + " " + sent).strip() if buf else sent
        if len(nxt) > lim:
            out.append(buf); buf = sent
        else:
            buf = nxt
    if buf: out.append(buf)
    if not out: raise SystemExit("[ERROR] Empty chunks")
    return out

def _norm_ref(src: Path, dst: Path):
    # mono, 16kHz, 16-bit PCM
    AudioSegment.from_file(str(src)).set_frame_rate(16000).set_channels(1).set_sample_width(2).export(dst, format="wav")

def main():
    pdf, ref, outp = Path(PDF_PATH), Path(VOICE_REF), Path(OUT_FILE)
    if not pdf.exists(): raise SystemExit(f"[ERROR] PDF not found: {pdf}")
    if not ref.exists(): raise SystemExit(f"[ERROR] Voice sample not found: {ref}")

    text = _extract_text(pdf, START_PAGE, END_PAGE)
    parts = _chunks(text, MAX_CHARS)

    work = Path(tempfile.mkdtemp(prefix="xtts_"))
    ref_wav = work / "ref.wav"
    # If we've already normalized the same file path in this run, reuse it
    if not ref_wav.exists():
        _norm_ref(ref, ref_wav)

    tts = get_tts()
    pad = AudioSegment.silent(ADD_SIL_MS)
    segs = []
    # Speed tip: wrap inference for PyTorch
    try:
        import torch
        inference_ctx = torch.inference_mode()
    except Exception:
        class _Dummy:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        inference_ctx = _Dummy()

    with inference_ctx:
        for i, chunk in enumerate(parts, 1):
            p = work / f"c{i:04}.wav"
            tts.tts_to_file(text=chunk, speaker_wav=str(ref_wav), language=LANG, file_path=str(p))
            segs.append(AudioSegment.from_wav(p) + pad)

    audio = AudioSegment.silent(150)  # tiny head
    for s in segs: audio += s
    audio.export(outp, format=(outp.suffix[1:] or "mp3"))
    print("✅ Done:", outp)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e); sys.exit(1)
