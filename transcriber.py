
import os
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
WHISPER_MODEL = "whisper-large-v3"

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

client = Groq(api_key=API_KEY)


def transcribe_audio(audio_bytes, filename="answer.webm"):
    """Transcribe audio bytes recorded in the browser."""

    if not audio_bytes:
        raise ValueError("No audio recording was received.")

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    result = client.audio.transcriptions.create(
        file=audio_file,
        model=WHISPER_MODEL,
        response_format="text",
    )

    if isinstance(result, str):
        return result.strip()

    return getattr(result, "text", "").strip()
