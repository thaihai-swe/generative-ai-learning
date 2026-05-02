"""
AI Meeting Assistant Package
A comprehensive tool for transcribing and summarizing business meetings

Uses base-provider for flexible LLM configuration
"""

from .transcriber import MeetingTranscriber, TranscriptionResult
from .summarizer import MeetingSummarizer
from .audio_processor import AudioRecorder, AudioFileManager, ReportGenerator
from .main import MeetingAssistant
from .config import get_llm_provider, close_llm_provider, reset_provider

__version__ = "2.0.0"
__author__ = "AI Learning"

__all__ = [
    'MeetingAssistant',
    'MeetingTranscriber',
    'TranscriptionResult',
    'MeetingSummarizer',
    'AudioRecorder',
    'AudioFileManager',
    'ReportGenerator',
    'get_llm_provider',
    'close_llm_provider',
    'reset_provider',
]
