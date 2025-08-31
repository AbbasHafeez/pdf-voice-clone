import warnings
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions now support")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import tempfile
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
import pdf_to_voice_clone_short as tts

st.set_page_config(page_title="PDF → Audiobook (Your Voice)", page_icon="🎧", layout="centered")
st.title("🎧 PDF → Audiobook in Your Voice")

# Stable workdir per session
if "workdir" not in st.session_state:
    st.session_state.workdir = Path(tempfile.mkdtemp(prefix="audiobook_"))

pdf   = st.file_uploader("PDF", type=["pdf"])
voice = st.file_uploader("Your voice (mp3/mp4/wav…)", type=["mp3","mp4","wav","m4a","aac","flac","ogg"])

# Page count (nice-to-have)
total = None
if pdf:
    try:
        r = PdfReader(pdf)
        total = len(r.pages)
        st.caption(f"Detected {total} pages (0-based).")
    except Exception as e:
        st.warning(f"Could not read page count: {e}")
    finally:
        pdf.seek(0)

c1, c2 = st.columns(2)
with c1: a = st.number_input("Start page (inclusive)", min_value=0, value=0, step=1)
with c2: b = st.number_input("End page (inclusive)",   min_value=0, value=(total - 1 if total else 0), step=1)

name = st.text_input("Output filename", value="audiobook_in_my_voice.mp3")
lang = st.selectbox("Language", ["en","de","fr","es","it","pt","ru","zh","ja","ko","ar"], index=0)

# Lazy-initialize model once so first click pre-warms; later clicks are fast
with st.sidebar:
    if st.button("Preload voice model (faster first run)"):
        _ = tts.get_tts()
        st.success("Model loaded and cached.")

go = st.button("Generate", type="primary", use_container_width=True)

if go:
    if not pdf or not voice:
        st.error("Upload both a PDF and a voice sample."); st.stop()

    w = st.session_state.workdir
    pdf_p   = w / "uploaded.pdf"
    voice_p = w / f"voice{Path(voice.name).suffix or '.wav'}"
    out_p   = w / (name or "audiobook_in_my_voice.mp3")

    pdf_p.write_bytes(pdf.read())
    voice_p.write_bytes(voice.read())

    if total:
        a = int(max(0, min(int(a), total - 1)))
        b = int(max(int(a), min(int(b), total - 1)))
    else:
        a, b = int(a), int(max(a, b))

    # Override backend globals
    tts.PDF_PATH   = str(pdf_p)
    tts.VOICE_REF  = str(voice_p)
    tts.OUT_FILE   = str(out_p)
    tts.LANG       = lang
    tts.START_PAGE = a
    tts.END_PAGE   = b

    with st.status("Synthesizing… (model is cached after first run)", expanded=False):
        try:
            tts.main()
        except SystemExit as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"Failed: {e}"); st.stop()

    if out_p.exists():
        st.session_state.audio_bytes = out_p.read_bytes()
        st.session_state.audio_name  = out_p.name
        st.success("Done! Your audiobook is ready.")
    else:
        st.error("No output produced.")

# Render from session_state (survives reruns)
if "audio_bytes" in st.session_state:
    data  = st.session_state.audio_bytes
    fname = st.session_state.get("audio_name", "audiobook.mp3")
    mime  = "audio/mpeg" if fname.lower().endswith(".mp3") else "audio/wav"
    st.audio(data, format=("audio/mp3" if mime == "audio/mpeg" else "audio/wav"))
    st.download_button("Download", data=data, file_name=fname, mime=mime, use_container_width=True)

    with st.expander("Reset"):
        if st.button("Clear generated audio"):
            st.session_state.pop("audio_bytes", None)
            st.session_state.pop("audio_name",  None)
            st.rerun()
