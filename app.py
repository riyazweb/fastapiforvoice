import os
from fastapi import FastAPI, UploadFile, File
import speech_recognition as sr
import tempfile

app = FastAPI()

# Initialize the SpeechRecognition recognizer only once
recognizer = sr.Recognizer()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Read the uploaded audio file as bytes
    audio_bytes = await file.read()

    # Save the bytes to a temporary file (with .wav suffix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name

    try:
        # Load the audio file using the SpeechRecognition library
        with sr.AudioFile(temp_audio_file_path) as source:
            audio = recognizer.record(source)  # Record the audio file
        
        # Recognize speech using Google's speech recognition engine
        text = recognizer.recognize_google(audio)
        return {"recognized_text": text}

    except sr.UnknownValueError:
        return {"error": "Sorry, I couldn't understand the audio."}
    except sr.RequestError as e:
        return {"error": f"Request failed; {e}"}
    finally:
        # Always remove the temporary file to free up space
        os.remove(temp_audio_file_path)

# This section is used only when running locally (not by Gunicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
