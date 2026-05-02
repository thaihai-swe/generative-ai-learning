"""
Audio Transcription Module
Handles conversion of audio files to text using OpenAI's Whisper API
"""

import os
from openai import OpenAI
from pathlib import Path


class MeetingTranscriber:
    """Transcribes meeting audio to text"""

    def __init__(self, api_key=None):
        """
        Initialize the transcriber with OpenAI API

        Args:
            api_key (str, optional): OpenAI API key. Uses OPENAI_API_KEY env var if not provided
        """
        self.client = OpenAI(api_key=api_key)
        self.supported_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']

    def transcribe_file(self, audio_file_path, language=None):
        """
        Transcribe an audio file to text

        Args:
            audio_file_path (str): Path to the audio file
            language (str, optional): ISO-639-1 language code (e.g., 'en', 'es', 'fr')

        Returns:
            dict: Transcription result containing 'text' and 'language'

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(audio_file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        print(f"Transcribing: {audio_file_path}")

        with open(audio_file_path, 'rb') as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                prompt="This is a business meeting. Please transcribe all conversations accurately."
            )

        result = {
            'text': transcript.text,
            'language': language or 'en',
            'file': str(file_path.name)
        }

        print(f"Transcription completed. Length: {len(transcript.text)} characters")
        return result

    def get_supported_formats(self):
        """Get list of supported audio formats"""
        return self.supported_formats


class TranscriptionResult:
    """Wrapper for transcription results"""

    def __init__(self, text, language='en', filename=''):
        self.text = text
        self.language = language
        self.filename = filename
        self.paragraphs = self._split_paragraphs()
        self.sentences = self._split_sentences()

    def _split_paragraphs(self):
        """Split transcription into paragraphs"""
        # Simple split by double newlines or periods followed by multiple spaces
        return [p.strip() for p in self.text.split('\n\n') if p.strip()]

    def _split_sentences(self):
        """Split transcription into sentences"""
        import re
        # Split by periods, question marks, exclamation marks
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"TranscriptionResult(length={len(self.text)}, language={self.language})"
