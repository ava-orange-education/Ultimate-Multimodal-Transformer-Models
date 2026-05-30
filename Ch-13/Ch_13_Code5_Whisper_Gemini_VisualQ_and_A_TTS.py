!pip install -q openai-whisper google-generativeai gtts librosa pydub
!pip install -q torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118  # For GPU
!apt update && apt install -y ffmpeg  # Audio processing
# Core multimodal + TTS (Whisper/Gemini already installed)
!pip install -q gtts pydub
# Audio conversion utilities
!apt update && apt install -y ffmpeg
print("Installations complete for voice-in/voice-out accessibility pipeline!")
import google.generativeai as genai
from IPython.display import Audio, display
from pathlib import Path
import numpy as np
import librosa
import whisper
import os
from google.colab import drive
drive.mount('/content/drive')
from gtts import gTTS
from pydub import AudioSegment
from IPython.display import Audio, display
import os
def save_tts(text, filename="caption.mp3", lang="en"):
    """Text-to-Speech with auto-save."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    return filename
def play_caption(caption_text):
    audio_file = save_tts(caption_text)
    print(f"Playing: {audio_file}")
    display(Audio(audio_file))
print("TTS pipeline ready!")
print("Test TTS: play_caption('Hello, this is an accessibility test.')")
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
]
# Load Whisper (use 'base' for speed, 'large' for accuracy)
print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper ready")
# Configure Gemini
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", # Fast + multimodal
    generation_config={"temperature": 0.1, "max_output_tokens": 200},
    safety_settings=safety_settings
)
print("Gemini ready")
def accessible_caption(image_path, voice_context_path):
    """Complete accessibility pipeline: Voice → Caption → TTS"""
    # 1. Transcribe voice context (accented speech)
    print("Transcribing context...")
    result = whisper_model.transcribe(voice_context_path)
    context = result["text"].strip()
    print(f"Context: '{context}'")  
    # 2. Generate contextual caption
    print("Generating caption...")
    img_file = genai.upload_file(image_path)
    prompt = f"Generate a DETAILED, accessible caption for this image, 
    considering user context: '{context}'. Include directions 
    (left/right), colors, distances, potential obstacles, and 
    navigation info."
    response = gemini_model.generate_content([prompt, img_file])
    caption = response.text.strip()
    print(f"Caption: {caption}")  
    # 3. Convert to speech
    print("Generating speech...")
    audio_file = save_tts(caption, "accessible_caption.mp3")
    return context, caption, audio_file
print("Function ready!")
# Test with our files
print("Testing accessibility pipeline...")
context, caption, audio_file = accessible_caption(
    "/samples/market.jpg",
    "/samples/context.wav"
)
# Auto-play the result
play_caption(caption)
print(f"Saved: {audio_file}")
