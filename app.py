from fastapi import FastAPI, UploadFile, File
import speech_recognition as sr
from io import BytesIO
import tempfile

app = FastAPI()

# Initialize recognizer
recognizer = sr.Recognizer()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Convert the uploaded file to a byte stream
    audio_bytes = await file.read()
    
    # Save the audio data into a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name

    try:
        # Load the audio file using speech_recognition
        with sr.AudioFile(temp_audio_file_path) as source:
            audio = recognizer.record(source)  # Record the audio file

        # Recognize speech using Google's Speech API
        text = recognizer.recognize_google(audio)
        return {"recognized_text": text}

    except sr.UnknownValueError:
        return {"error": "Sorry, I couldn't understand the audio."}
    except sr.RequestError as e:
        return {"error": f"Request failed; {e}"}
    finally:
        # Cleanup the temporary file
        os.remove(temp_audio_file_path)

# To run the server with Uvicorn locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
