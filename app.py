from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper_timestamped
import os
import uuid

app = FastAPI()

# Load model once globally
model = whisper_timestamped.load_model("base")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_location = f"temp/{file_id}_{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Transcribe using whisper_timestamped
    try:
        result = whisper_timestamped.transcribe(model, file_location)
        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            } for seg in result["segments"]
        ]
        return {"segments": segments}
    
    finally:
        os.remove(file_location)  # Clean up temp file
