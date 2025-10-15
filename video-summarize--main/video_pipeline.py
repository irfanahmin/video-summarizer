# video_pipeline.py

import os
import sys
import traceback
import nltk
# NOTE: Removed 'from transformers import pipeline as hf_pipeline'
# NOTE: Removed 'try/except import whisper'

# Sumy dependencies (relatively light, can stay)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Import ffmpeg with error handling (relatively light, can stay)
try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python package not found. Installing...")
    # NOTE: This line might fail on Render unless run in a build step,
    # but we keep it for local development compatibility.
    os.system(f"{sys.executable} -m pip install ffmpeg-python")
    import ffmpeg


# --- Global Model Caching Variables ---
# Models will be loaded into these variables on first use.
WHISPER_MODEL = None
SUMMARIZER_PIPELINE = None
# --------------------------------------

# paths & model IDs
AUDIO_PATH = "audio.wav"
ABSTRACTIVE_MODEL_ID = "facebook/bart-base"


# --- Model Loading & Caching Functions (Lazy Loading) ---

def get_whisper_model():
    """Loads and caches the Whisper model on first call."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            import whisper 
            # --- CRITICAL CHANGE: Use the English-only version to save memory ---
            print("--- LAZY LOADING: Loading Whisper model (tiny.en) ---") 
            WHISPER_MODEL = whisper.load_model("tiny.en", device="cpu") # <-- CHANGED TO 'tiny.en'
            print("--- LAZY LOADING: Whisper model loaded successfully ---")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
    return WHISPER_MODEL


def get_summarizer_pipeline():
    """Loads and caches the Hugging Face summarizer pipeline on the first call."""
    global SUMMARIZER_PIPELINE
    if SUMMARIZER_PIPELINE is None:
        try:
            # ⚠️ Import heavy library inside the function to prevent global startup load
            from transformers import pipeline as hf_pipeline
            print(f"--- LAZY LOADING: Loading Summarizer pipeline ({ABSTRACTIVE_MODEL_ID}) ---")
            SUMMARIZER_PIPELINE = hf_pipeline("summarization", model=ABSTRACTIVE_MODEL_ID)
            print("--- LAZY LOADING: Summarizer pipeline loaded successfully ---")
        except Exception as e:
            raise Exception(f"Failed to load summarization model: {str(e)}")
    return SUMMARIZER_PIPELINE

# --------------------------------------------------------------------------------


# download required NLTK models (relatively light)
def ensure_nltk_dependencies():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        print("NLTK models downloaded successfully")
    except Exception as e:
        error_msg = f"Error downloading NLTK models: {e}"
        print(error_msg)
        raise Exception(error_msg)

# Ensure NLTK dependencies are available
ensure_nltk_dependencies()


def extract_audio(video_path: str) -> str:
    """Extract WAV at 16 kHz from video."""
    try:
        # Check if input file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(AUDIO_PATH) or '.', exist_ok=True)
            
        # Try to run ffmpeg
        try:
            ffmpeg.input(video_path) \
                  .output(AUDIO_PATH, format="wav", acodec="pcm_s16le", ar="16000") \
                  .run(overwrite_output=True, quiet=True)
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise Exception(
                "Error processing video with FFmpeg. Make sure FFmpeg is properly installed "
                "and the video file is not corrupted."
            ) from e
            
        # Verify output file was created
        if not os.path.exists(AUDIO_PATH):
            raise Exception("Audio extraction failed: output file was not created")
            
        return AUDIO_PATH
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in extract_audio: {error_msg}\nStack trace:\n{stack_trace}")
        raise Exception(f"Error processing video: {error_msg}") from e

def transcribe_audio(audio_path: str) -> str:
    """Run Whisper-small and stitch all segments."""
    try:
        # 1. Validate input file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Audio file found at: {audio_path}")
        print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
            
        # 2. Load Whisper model using the CACHING GETTER
        try:
            print("Loading Whisper model...")
            model = get_whisper_model() # <--- CRITICAL CHANGE: Use lazy-loaded model
            print("Whisper model ready.")
        except Exception as model_error:
            raise Exception(f"Failed to load Whisper model: {str(model_error)}")
        
        # 3. Transcribe audio with progress updates
        print("Starting audio transcription...")
        try:
            result = model.transcribe(
                audio_path,
                fp16=False,
                language="en",
                task="transcribe"
            )
            print("Transcription completed successfully")
        except Exception as transcribe_error:
            raise Exception(f"Transcription process failed: {str(transcribe_error)}")
        
        # 4. Validate and process results...
        if not result:
            raise Exception("Transcription failed: empty result")
            
        if "segments" not in result:
            raise Exception("Transcription failed: no segments in result")
            
        if not result["segments"]:
            raise Exception("Transcription failed: segments list is empty")
        
        # 5. Join segments and validate transcript
        transcript = " ".join(seg["text"].strip() for seg in result["segments"])
        
        if not transcript.strip():
            raise Exception("Transcription failed: empty transcript generated")
            
        print(f"Generated transcript length: {len(transcript)} characters")
        return transcript
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in transcribe_audio: {error_msg}")
        print(f"Stack trace:\n{stack_trace}")
        raise Exception(f"Error transcribing audio: {error_msg}") from e

def save_transcript(text: str, path: str = "transcript.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def abstractive_summary(text: str) -> str:
    """Return BART summary."""
    summarizer = get_summarizer_pipeline() # <--- CRITICAL CHANGE: Use lazy-loaded pipeline
    return summarizer(text, max_length=300, min_length=100, do_sample=False)[0]["summary_text"]

def extractive_summary(text: str, num_sentences: int = 20) -> str:
    """Return top-N sentences via LexRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sents = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in sents)

def generate_structured_notes(text: str) -> dict:
    """Convert summary text into structured notes with headings and points."""
    # NLTK imports are relatively light and already guarded by try/except block.
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    
    # NLTK data should already be downloaded at this point
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    structured_notes = {
        'headings': [],
        'points': {}
    }
    
    current_heading = 'Main Points'
    structured_notes['headings'].append(current_heading)
    structured_notes['points'][current_heading] = []
    
    for sentence in sentences:
        # Check if sentence looks like a heading (starts with key phrases or is short)
        words = word_tokenize(sentence)
        tags = pos_tag(words)
        
        is_heading = False
        # Check for heading indicators
        if len(words) <= 7 and any(tag.startswith('NN') for word, tag in tags[:2]):
            is_heading = True
        elif any(phrase in sentence.lower() for phrase in ['firstly', 'secondly', 'finally', 'in conclusion', 'to summarize']):
            is_heading = True
        elif sentence.endswith(':'):
            is_heading = True
            
        if is_heading:
            current_heading = sentence.rstrip(':')
            if current_heading not in structured_notes['headings']:
                 structured_notes['headings'].append(current_heading)
                 structured_notes['points'][current_heading] = []
        else:
            # Clean up the sentence
            clean_sentence = sentence.strip()
            # Ensure current heading exists before appending (safety check)
            if current_heading not in structured_notes['points']:
                 structured_notes['points'][current_heading] = []
            if clean_sentence:
                structured_notes['points'][current_heading].append(clean_sentence)
    
    return structured_notes
