# --- stability & noise control (set BEFORE importing streamlit) ---
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")  # safer on Windows/OneDrive
os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")                # quieter logs

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Torchaudio's I/O functions now support .*backend dispatch.*",
    category=UserWarning,
)
# ------------------------------------------------------------------

import tempfile
from pathlib import Path
import streamlit as st
import fitz  # PyMuPDF

from pdf_to_voice_clone_short import (
    convert_pdf_to_cloned_audiobook,
    get_pdf_meta,
    have_ffmpeg,
)

st.set_page_config(page_title="PDF → Cloned Audiobook (XTTS-v2)", page_icon="🎧", layout="centered")

st.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:0">📚 → 🎧 Cloned Audiobook</h1>
      <p style="margin-top:4px;color:#888">XTTS-v2 (multilingual, voice cloning). Clean sequential pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    c1, c2 = st.columns([1, 1])
    with c1:
        pdf_file = st.file_uploader("PDF file", type=["pdf"], help="Upload the document to narrate")
    with c2:
        ref_file = st.file_uploader(
            "Reference voice (optional)",
            type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "aac"],
            help="Short voice sample to clone. WAV works even without FFmpeg."
        )

# Show PDF metadata if uploaded
pdf_path = None
page_count = None
title = ""
if pdf_file:
    tmp_dir = Path(tempfile.gettempdir()) / "pdf2voice_streamlit"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / ("in_" + pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())
    try:
        page_count, title = get_pdf_meta(pdf_path)
    except Exception:
        page_count, title = None, ""

    if page_count:
        st.info(f"📄 **Pages:** {page_count}" + (f" • **Title:** {title}" if title else ""))

st.divider()

# Scope controls
scope = st.radio(
    "Select scope",
    ["Entire document", "Page range", "First N words"],
    horizontal=True,
)

c1, c2, c3 = st.columns(3)
if scope == "Entire document":
    start_page = end_page = None
    word_limit = None
elif scope == "Page range":
    default_end = page_count or 1
    with c1:
        start_page = st.number_input("Start page (1-based)", min_value=1, value=1, step=1)
    with c2:
        end_page = st.number_input("End page (1-based)", min_value=1, value=int(default_end), step=1)
    word_limit = None
else:  # First N words
    with c1:
        word_limit = st.number_input("N words", min_value=50, value=500, step=50)
    start_page = end_page = None

# Synthesis settings
c1, c2, c3 = st.columns(3)
with c1:
    lang = st.text_input("Language code", value="en", help="e.g., en, ur")
with c2:
    max_chars = st.slider("Max chars per chunk", min_value=120, max_value=600, value=240, step=10)
with c3:
    silence_ms = st.slider("Silence between chunks (ms)", min_value=0, max_value=2000, value=300, step=50)

out_name = st.text_input("Output file name (without extension)", value="audiobook")

# Export note
if not have_ffmpeg():
    st.warning(
        "FFmpeg not detected. Export will be **.wav** and non-WAV reference files may not convert. "
        "Install FFmpeg for MP3 export and broader input support."
    )

# Progress UI
progress_bar = st.progress(0, text="Waiting…")
status = st.empty()

def make_progress_fn():
    # Backend calls this with (idx, total, message) in sequential order
    def fn(idx: int, total: int, message: str):
        pct = int((idx / max(1, total)) * 100)
        progress_bar.progress(pct, text=f"{message} ({idx}/{total})")
        status.write(f"🗣️ {message}")
    return fn

go = st.button("Generate 🎧", type="primary", use_container_width=True, disabled=not pdf_file)

if go:
    # Save reference voice if any
    voice_path = None
    if ref_file:
        tmp_dir = Path(tempfile.gettempdir()) / "pdf2voice_streamlit"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        voice_path = tmp_dir / ("ref_" + ref_file.name)
        with open(voice_path, "wb") as f:
            f.write(ref_file.read())

    out_base = Path(tempfile.gettempdir()) / "pdf2voice_streamlit" / out_name  # ext decided automatically

    progress_fn = make_progress_fn()

    with st.spinner("Synthesizing… first model load can take a moment on CPU"):
        try:
            final_path = convert_pdf_to_cloned_audiobook(
                pdf_path=pdf_path,
                voice_ref_path=voice_path,
                out_path=out_base,
                start_page=None if scope != "Page range" else int(start_page),
                end_page=None if scope != "Page range" else int(end_page),
                word_limit=None if scope != "First N words" else int(word_limit),
                lang=lang.strip() or "en",
                max_chars=int(max_chars),
                silence_ms=int(silence_ms),
                progress_fn=progress_fn,
            )
        except Exception as e:
            progress_bar.progress(0, text="Error")
            status.empty()
            st.exception(e)
            st.stop()

    progress_bar.progress(100, text="Done")
    status.empty()

    # Show download + in-page player
    st.success("All set! Download or play below.")
    with open(final_path, "rb") as f:
        data = f.read()
        mime = "audio/mpeg" if final_path.suffix.lower() == ".mp3" else "audio/wav"
        st.download_button(
            label="⬇️ Download",
            data=data,
            file_name=final_path.name,
            mime=mime,
            use_container_width=True,
        )
        st.audio(data, format=mime)
