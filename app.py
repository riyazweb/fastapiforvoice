from fastapi import FastAPI, UploadFile, File
import whisper_timestamped
import uuid
import os

app = FastAPI()

# Load Whisper model
model = whisper_timestamped.load_model("base")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filepath = f"/tmp/{file_id}_{file.filename}"

    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        result = whisper_timestamped.transcribe(model, filepath)
        segments = [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip()
            }
            for s in result["segments"]
        ]
        return {"segments": segments}
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("FastAPI is running at http://127.0.0.1:8000")
