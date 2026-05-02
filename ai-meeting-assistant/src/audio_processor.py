"""
Audio Processing Module
Handles recording, file handling, and audio format conversions
"""

import os
import json
from pathlib import Path
from datetime import datetime


class AudioRecorder:
    """Records audio from microphone"""

    def __init__(self, sample_rate=16000, channels=1):
        """
        Initialize audio recorder

        Args:
            sample_rate (int): Sample rate in Hz (default 16000)
            channels (int): Number of audio channels (default 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False

    def record_meeting(self, duration_seconds, output_file):
        """
        Record audio for specified duration

        Args:
            duration_seconds (int): Duration to record in seconds
            output_file (str): Path to save the recording

        Returns:
            bool: True if recording was successful
        """
        try:
            import sounddevice as sd
            import soundfile as sf

            print(f"Recording for {duration_seconds} seconds...")
            print("Press Ctrl+C to stop early")

            # Record audio
            audio_data = sd.rec(
                int(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels
            )
            sd.wait()

            # Save to file
            sf.write(output_file, audio_data, self.sample_rate)
            print(f"Recording saved to: {output_file}")

            return True

        except ImportError:
            print("Error: sounddevice and soundfile are required for recording.")
            print("Install with: pip install sounddevice soundfile")
            return False
        except Exception as e:
            print(f"Error during recording: {e}")
            return False


class AudioFileManager:
    """Manages audio files and metadata"""

    def __init__(self, data_directory='./data'):
        """
        Initialize the file manager

        Args:
            data_directory (str): Directory for storing audio files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_directory / 'meetings_metadata.json'

    def save_audio_metadata(self, filename, meeting_info):
        """
        Save metadata about a meeting recording

        Args:
            filename (str): Audio file name
            meeting_info (dict): Meeting information
        """
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'title': meeting_info.get('title', 'Untitled Meeting'),
            'participants': meeting_info.get('participants', []),
            'duration': meeting_info.get('duration', 0),
            'notes': meeting_info.get('notes', '')
        }

        # Load existing metadata
        all_metadata = []
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)

        # Add new metadata
        all_metadata.append(metadata)

        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def get_audio_files(self):
        """
        Get list of all audio files in the data directory

        Returns:
            list: List of audio file paths
        """
        audio_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.webm', '.mpeg', '.mpga'}
        audio_files = [
            f for f in self.data_directory.iterdir()
            if f.suffix.lower() in audio_extensions
        ]
        return sorted(audio_files)

    def get_meeting_history(self):
        """
        Get history of recorded meetings

        Returns:
            list: List of meeting metadata
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return []

    def delete_meeting(self, filename):
        """
        Delete a meeting recording and its metadata

        Args:
            filename (str): Name of the file to delete

        Returns:
            bool: True if successful
        """
        file_path = self.data_directory / filename

        try:
            if file_path.exists():
                file_path.unlink()

            # Remove from metadata
            metadata = self.get_meeting_history()
            metadata = [m for m in metadata if m['filename'] != filename]

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"Error deleting meeting: {e}")
            return False


class ReportGenerator:
    """Generates and saves meeting reports"""

    def __init__(self, output_directory='./output'):
        """
        Initialize report generator

        Args:
            output_directory (str): Directory for saving reports
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def save_report_to_file(self, meeting_title, report_data):
        """
        Save meeting report to a file

        Args:
            meeting_title (str): Title of the meeting
            report_data (dict): Report data containing summary, key points, etc.

        Returns:
            str: Path to saved report
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{meeting_title.replace(' ', '_')}_{timestamp}.txt"
        filepath = self.output_directory / filename

        # Format and write report
        with open(filepath, 'w') as f:
            f.write(f"Meeting Report: {meeting_title}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if 'participants' in report_data and report_data['participants']:
                f.write("PARTICIPANTS\n")
                f.write("-" * 40 + "\n")
                for participant in report_data['participants']:
                    f.write(f"• {participant}\n")
                f.write("\n")

            if 'summary' in report_data:
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(report_data['summary'] + "\n\n")

            if 'key_points' in report_data and report_data['key_points']:
                f.write("KEY POINTS\n")
                f.write("-" * 40 + "\n")
                for point in report_data['key_points']:
                    f.write(f"• {point}\n")
                f.write("\n")

            if 'action_items' in report_data:
                f.write("ACTION ITEMS\n")
                f.write("-" * 40 + "\n")
                f.write(report_data['action_items'] + "\n\n")

            if 'transcription' in report_data:
                f.write("FULL TRANSCRIPTION\n")
                f.write("-" * 40 + "\n")
                f.write(report_data['transcription'] + "\n")

        return str(filepath)

    def save_report_as_json(self, meeting_title, report_data):
        """
        Save meeting report as JSON for programmatic access

        Args:
            meeting_title (str): Title of the meeting
            report_data (dict): Report data

        Returns:
            str: Path to saved JSON file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{meeting_title.replace(' ', '_')}_{timestamp}.json"
        filepath = self.output_directory / filename

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

        return str(filepath)
