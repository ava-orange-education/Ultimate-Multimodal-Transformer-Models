!pip install -q openai-whisper google-generativeai gtts librosa pydub
!pip install -q torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118  # For GPU
!apt update && apt install -y ffmpeg  # Audio processing
import whisper
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import librosa
import numpy as np
from IPython.display import Audio, display
import os
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
# Configure Gemini API (get free key from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY = "GEMINI_API_KEY"  # Replace with your key
genai.configure(api_key=GEMINI_API_KEY)
# Safety settings for production
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
]
print("All libraries loaded. Ready for voice VQA pipeline!")
def load_audio(audio_path, sr=16000):
    """Load and resample audio for Whisper."""
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio.astype(np.float32)

def display_audio(audio_path):
    """Display audio in notebook."""
    display(Audio(audio_path))
# Create sample files directory
os.makedirs("/samples", exist_ok=True)
print("Sample directory ready: samples/")
import speech_recognition as sr
def live_voice_vqa(image_path):
    """Live mic → Whisper → Gemini VQA"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now (3s)...")
        audio = r.listen(source, timeout=3)
    # Save temp file for Whisper
    with open("live_query.wav", "wb") as f:
        f.write(audio.get_wav_data())
    return voice_vqa("live_query.wav", image_path)
# Load Whisper (use 'base' for speed, 'large' for accuracy)
print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper ready")
# Configure Gemini
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # Fast + multimodal
    generation_config={"temperature": 0.1, "max_output_tokens": 200},
    safety_settings=safety_settings
)
print("Gemini ready")
def voice_vqa_v2(audio_path, image_path):
    """Ultra-simple version using file upload""" 
    # 1. Transcribe
    print("Transcribing...")
    result = whisper_model.transcribe(audio_path)
    question = result["text"].strip()
    print(f"Heard: '{question}'") 
    # 2. Upload image once
    img_file = genai.upload_file(image_path)   
    # 3. Generate
    print("Gemini reasoning...")
    response = gemini_model.generate_content([question, img_file])
    answer = response.text.strip()
    print(f"Answer: {answer}")
    return question, answer
# Test it!
question, answer = voice_vqa_v2(
    "/samples/Q1.wav", 
    "/samples/kids_play.jpg"
)
