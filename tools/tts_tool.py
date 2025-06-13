# /Users/surfiniaburger/Desktop/app/tools/tts_tool.py
import logging
import os
import io
from typing import Optional, Literal

from google.cloud import texttospeech

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") # Ensure this is set in your environment

# TTS Config - Consider making these configurable if needed by the agent
TTS_VOICE_NAME_PERSON_A = "en-US-Chirp3-HD-Puck" # Example standard voice
TTS_VOICE_NAME_PERSON_B = "en-US-Chirp3-HD-Aoede" # Another example standard voice
AUDIO_ENCODING = texttospeech.AudioEncoding.MP3
AUDIO_SAMPLE_RATE_HZ = 24000 # Good quality, common for MP3

logger = logging.getLogger(__name__)

try:
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("Google Cloud TTS client initialized successfully for tts_tool.")
except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud TTS client in tts_tool: {e}", exc_info=True)
    tts_client = None

def synthesize_speech_segment(
    text_to_synthesize: str,
    speaker: Literal["PERSON_A", "PERSON_B"] = "PERSON_A",
    language_code: str = "en-US",
) -> Optional[bytes]:
    """
    Synthesizes speech from text for a single segment and returns the audio content as bytes.

    Args:
        text_to_synthesize (str): The text to be converted to speech.
        speaker (Literal["PERSON_A", "PERSON_B"]): Specifies which predefined voice to use.
                                                   Defaults to "PERSON_A".
        language_code (str): The language code (e.g., "en-US"). Defaults to "en-US".

    Returns:
        Optional[bytes]: The MP3 audio data as bytes, or None if synthesis fails.
                         The LLM agent calling this should handle the bytes (e.g., by passing
                         them to another tool for video assembly).
    """
    if not tts_client:
        logger.error("TTS client not initialized. Cannot synthesize speech.")
        return None
    if not text_to_synthesize:
        logger.warning("No text provided for speech synthesis.")
        return None

    selected_voice_name = TTS_VOICE_NAME_PERSON_A if speaker == "PERSON_A" else TTS_VOICE_NAME_PERSON_B

    try:
        logger.info(f"Synthesizing speech for speaker '{speaker}' (voice: {selected_voice_name}): \"{text_to_synthesize[:50]}...\"")
        synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=selected_voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=AUDIO_ENCODING,
            sample_rate_hertz=AUDIO_SAMPLE_RATE_HZ,
            # speaking_rate=1.0, # Optional: Adjust speed
            # pitch=0.0, # Optional: Adjust pitch
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        logger.info(f"Successfully synthesized audio for speaker '{speaker}'.")
        return response.audio_content

    except Exception as e:
        logger.error(f"Error during TTS synthesis for speaker '{speaker}': {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example Usage (for testing the tool directly)
    logging.basicConfig(level=logging.INFO)
    sample_text_a = "Hello, this is speaker A. Welcome to the report."
    audio_bytes_a = synthesize_speech_segment(sample_text_a, speaker="PERSON_A")
    if audio_bytes_a:
        with open("speaker_a_test.mp3", "wb") as f:
            f.write(audio_bytes_a)
        print("Saved speaker_a_test.mp3")

    sample_text_b = "And this is speaker B. We have some interesting findings today."
    audio_bytes_b = synthesize_speech_segment(sample_text_b, speaker="PERSON_B")
    if audio_bytes_b:
        with open("speaker_b_test.mp3", "wb") as f:
            f.write(audio_bytes_b)
        print("Saved speaker_b_test.mp3")