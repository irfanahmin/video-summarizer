# pipeline.py

import whisper
import ffmpeg
import nltk
from transformers import pipeline as hf_pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# download punkt once
nltk.download("punkt", quiet=True)

# paths & model IDs
AUDIO_PATH = "audio.wav"
ABSTRACTIVE_MODEL_ID = "facebook/bart-large-cnn"

def extract_audio(video_path: str) -> str:
    """Extract WAV at 16 kHz from video."""
    ffmpeg.input(video_path) \
          .output(AUDIO_PATH, format="wav", acodec="pcm_s16le", ar="16000") \
          .run(overwrite_output=True, quiet=True)
    return AUDIO_PATH

def transcribe_audio(audio_path: str) -> str:
    """Run Whisper-small and stitch all segments."""
    model = whisper.load_model("small", device="cpu")
    result = model.transcribe(audio_path, fp16=False)
    return " ".join(seg["text"].strip() for seg in result["segments"])

def save_transcript(text: str, path: str = "transcript.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def abstractive_summary(text: str) -> str:
    """Return BART‑large‑cnn summary."""
    summarizer = hf_pipeline("summarization", model=ABSTRACTIVE_MODEL_ID)
    return summarizer(text, max_length=300, min_length=100, do_sample=False)[0]["summary_text"]

def extractive_summary(text: str, num_sentences: int = 20) -> str:
    """Return top-N sentences via LexRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sents = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in sents)
