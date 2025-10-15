# video-summarizer.py

import os
# We move whisper and transformers imports below
import ffmpeg # Relatively light
import nltk # Relatively light
from nltk.tokenize import sent_tokenize

# Optional extractive summarizer (relatively light)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Global model caches
WHISPER_MODEL = None
SUMMARIZER_PIPELINE = None


# Ensure punkt is available
nltk.download("punkt", quiet=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

VIDEO_PATH           = "video2.mp4"
AUDIO_PATH           = "audio.wav"
TRANSCRIPT_PATH      = "transcript.txt"
SUMMARY_ABS_PATH     = "summary_abstractive.txt"
SUMMARY_EXT_PATH     = "summary_extractive.txt"

# Use Hugging Face’s BART from the Hub for abstractive summarization
ABSTRACTIVE_MODEL_ID = "facebook/bart-base" # NOTE: Changed to 'bart-base' for better memory fit


# --- Model Loading & Caching Functions (Lazy Loading) ---

def get_whisper_model():
    """Loads and caches the Whisper model on the first call."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper
        print("--- LAZY LOADING: Loading Whisper model (tiny) ---")
        WHISPER_MODEL = whisper.load_model("tiny", device="cpu")
        print("--- LAZY LOADING: Whisper model loaded successfully ---")
    return WHISPER_MODEL

def get_summarizer_pipeline():
    """Loads and caches the Hugging Face summarizer pipeline on the first call."""
    global SUMMARIZER_PIPELINE
    if SUMMARIZER_PIPELINE is None:
        from transformers import pipeline
        print(f"--- LAZY LOADING: Loading Summarizer pipeline ({ABSTRACTIVE_MODEL_ID}) ---")
        SUMMARIZER_PIPELINE = pipeline("summarization", model=ABSTRACTIVE_MODEL_ID)
        print("--- LAZY LOADING: Summarizer pipeline loaded successfully ---")
    return SUMMARIZER_PIPELINE

# --------------------------------------------------------------------------------


# ─── STEP 1: AUDIO EXTRACTION ─────────────────────────────────────────────────

def extract_audio(video_path: str, audio_path: str = AUDIO_PATH) -> str|None:
    try:
        ffmpeg.input(video_path)\
              .output(audio_path, format="wav", acodec="pcm_s16le", ar="16000")\
              .run(overwrite_output=True, quiet=True)
        print(f"✅ Audio extracted: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"❌ Audio extraction error: {e}")
        return None

# ─── STEP 2: TRANSCRIPTION ────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str|None:
    """
    Returns the full stitched transcript from Whisper.
    """
    try:
        model = get_whisper_model() # <<< CRITICAL CHANGE: Use cached model
        result = model.transcribe(audio_path, fp16=False)
        # Stitch all segments to preserve everything
        full_text = " ".join(seg["text"].strip() for seg in result["segments"])
        print(f"✅ Transcription done ({len(full_text.split())} words).")
        return full_text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

def save_transcript(text: str, path: str = TRANSCRIPT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Full transcript saved: {path}")

# ─── STEP 3a: ABSTRACTIVE SUMMARY ──────────────────────────────────────────────

def abstractive_summary(text: str, output_path: str = SUMMARY_ABS_PATH):
    summarizer = get_summarizer_pipeline() # <<< CRITICAL CHANGE: Use cached pipeline
    # one big chunk (or chunk if you want)
    summary = summarizer(text, max_length=300, min_length=100, do_sample=False)[0]["summary_text"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Abstractive summary saved: {output_path}")

# ─── STEP 3b: EXTRACTIVE SUMMARY ──────────────────────────────────────────────

def extractive_summary(text: str, num_sentences: int = 20, output_path: str = SUMMARY_EXT_PATH):
    parser     = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary    = summarizer(parser.document, num_sentences)
    out_text   = " ".join(str(sent) for sent in summary)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"✅ Extractive summary ({num_sentences} sentences) saved: {output_path}")

# ─── MAIN WORKFLOW ────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    audio = extract_audio(VIDEO_PATH)
    if not audio:
        return

    transcript = transcribe_audio(audio)
    if not transcript:
        return

    # 1) Save the *entire* transcript
    save_transcript(transcript)

    # 2) (Optional) Abstractive summary
    abstractive_summary(transcript)

    # 3) (Optional) Extractive summary
    extractive_summary(transcript, num_sentences=20)

if __name__ == "__main__":
    main()
