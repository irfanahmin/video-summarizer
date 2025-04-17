# main.py

import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import (
    extract_audio,
    transcribe_audio,
    save_transcript,
    abstractive_summary,
    extractive_summary
)

app = FastAPI()

# Allow CORS (if you serve frontend separately or need AJAX)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets under /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    # 1) Save uploaded video
    video_path = file.filename
    with open(video_path, "wb") as f:
        f.write(await file.read())

    try:
        # 2) Extract audio, transcribe, save transcript
        audio_path = extract_audio(video_path)
        transcript = transcribe_audio(audio_path)
        save_transcript(transcript)

        # 3) Generate summaries
        summary_abs = abstractive_summary(transcript)
        summary_ext = extractive_summary(transcript)

        # 4) Clean up temp files
        os.remove(video_path)
        os.remove(audio_path)

        # 5) Return JSON
        return JSONResponse({
            "transcript": transcript,
            "summary_abstractive": summary_abs,
            "summary_extractive": summary_ext
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
