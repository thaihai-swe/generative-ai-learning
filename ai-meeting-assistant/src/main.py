"""
AI Meeting Assistant - Main Application
A comprehensive tool for transcribing and summarizing business meetings

Uses base-provider for flexible LLM configuration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from transcriber import MeetingTranscriber, TranscriptionResult
from summarizer import MeetingSummarizer
from audio_processor import AudioFileManager, ReportGenerator, AudioRecorder
from config import get_llm_provider, close_llm_provider


class MeetingAssistant:
    """Main application class for the AI Meeting Assistant"""

    def __init__(self):
        """Initialize the meeting assistant"""
        load_dotenv()

        # Initialize LLM provider (uses base-provider)
        try:
            llm_provider = get_llm_provider()
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LLM provider: {e}\n"
                "Please ensure base-provider is installed and API key is configured"
            )

        self.transcriber = MeetingTranscriber(api_key=os.getenv('OPENAI_API_KEY'))
        self.summarizer = MeetingSummarizer(provider=llm_provider)
        self.file_manager = AudioFileManager()
        self.report_generator = ReportGenerator()
        self.recorder = AudioRecorder()

    def process_meeting(self, audio_file_path, meeting_title="Meeting", language=None):
        """
        Process a complete meeting: transcribe and summarize

        Args:
            audio_file_path (str): Path to the audio file
            meeting_title (str): Title for the meeting
            language (str): ISO-639-1 language code (optional)

        Returns:
            dict: Complete meeting report
        """
        print(f"\n{'='*80}")
        print(f"Processing Meeting: {meeting_title}")
        print(f"{'='*80}\n")

        # Step 1: Transcribe audio
        print("STEP 1: Transcribing audio...")
        print("-" * 40)
        transcription_result = self.transcriber.transcribe_file(audio_file_path, language)
        transcription_text = transcription_result['text']
        print(f"✓ Transcription completed\n")

        # Step 2: Generate comprehensive report
        print("STEP 2: Analyzing meeting and generating report...")
        print("-" * 40)
        report = self.summarizer.generate_full_report(transcription_text)
        report['transcription'] = transcription_text
        report['audio_file'] = Path(audio_file_path).name
        report['meeting_title'] = meeting_title
        print(f"✓ Analysis completed\n")

        # Step 3: Save reports
        print("STEP 3: Saving reports...")
        print("-" * 40)
        text_report_path = self.report_generator.save_report_to_file(meeting_title, report)
        json_report_path = self.report_generator.save_report_as_json(meeting_title, report)
        print(f"✓ Text report saved to: {text_report_path}")
        print(f"✓ JSON report saved to: {json_report_path}\n")

        # Display summary
        self._display_report_summary(report)

        return report

    def interactive_mode(self):
        """Run the assistant in interactive mode"""
        print("\n" + "="*80)
        print("AI MEETING ASSISTANT - Interactive Mode")
        print("="*80 + "\n")

        while True:
            print("\nOptions:")
            print("1. Process existing audio file")
            print("2. Record new meeting")
            print("3. View meeting history")
            print("4. Exit")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                self._process_file_interactive()
            elif choice == '2':
                self._record_meeting_interactive()
            elif choice == '3':
                self._view_history()
            elif choice == '4':
                print("Exiting... Goodbye!")
                break
            else:
                print("Invalid option. Please select 1-4.")

    def _process_file_interactive(self):
        """Interactive file processing"""
        audio_files = self.file_manager.get_audio_files()

        if not audio_files:
            print("\nNo audio files found in data directory.")
            file_path = input("Enter path to audio file: ").strip()
        else:
            print("\nAvailable audio files:")
            for i, f in enumerate(audio_files, 1):
                print(f"{i}. {f.name}")

            choice = input("Select file number or enter path: ").strip()
            try:
                file_path = str(audio_files[int(choice) - 1])
            except (ValueError, IndexError):
                file_path = choice

        meeting_title = input("Enter meeting title: ").strip() or "Meeting"
        language = input("Enter language code (e.g., 'en', 'es') or press Enter for auto: ").strip() or None

        try:
            self.process_meeting(file_path, meeting_title, language)
        except Exception as e:
            print(f"\nError processing meeting: {e}")

    def _record_meeting_interactive(self):
        """Interactive meeting recording"""
        print("\nStarting meeting recording...")

        meeting_title = input("Enter meeting title: ").strip() or "Meeting"
        duration = int(input("Enter recording duration in seconds (default 60): ").strip() or "60")

        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{meeting_title.replace(' ', '_')}_{timestamp}.wav"
        filepath = self.file_manager.data_directory / filename

        if self.recorder.record_meeting(duration, str(filepath)):
            self.file_manager.save_audio_metadata(filename, {
                'title': meeting_title,
                'duration': duration
            })

            process_now = input("Process meeting now? (y/n): ").strip().lower() == 'y'
            if process_now:
                self.process_meeting(str(filepath), meeting_title)
        else:
            print("Recording failed.")

    def _view_history(self):
        """Display meeting history"""
        history = self.file_manager.get_meeting_history()

        if not history:
            print("\nNo meeting history found.")
            return

        print("\n" + "="*80)
        print("MEETING HISTORY")
        print("="*80 + "\n")

        for i, meeting in enumerate(history, 1):
            print(f"{i}. {meeting['title']}")
            print(f"   File: {meeting['filename']}")
            print(f"   Date: {meeting['timestamp']}")
            if meeting['participants']:
                print(f"   Participants: {', '.join(meeting['participants'])}")
            print()

    def _display_report_summary(self, report):
        """Display a formatted summary of the report"""
        print("\n" + "="*80)
        print("MEETING REPORT SUMMARY")
        print("="*80 + "\n")

        print("SUMMARY:")
        print("-" * 40)
        print(report.get('summary', 'N/A') + "\n")

        if report.get('key_points'):
            print("KEY POINTS:")
            print("-" * 40)
            for i, point in enumerate(report['key_points'], 1):
                print(f"{i}. {point}")
            print()

        if report.get('participants'):
            print("PARTICIPANTS:")
            print("-" * 40)
            for participant in report['participants']:
                print(f"• {participant}")
            print()


def main():
    """Main entry point"""
    try:
        assistant = MeetingAssistant()

        # Check for command line arguments
        if len(sys.argv) > 1:
            # Process file from command line
            audio_file = sys.argv[1]
            meeting_title = sys.argv[2] if len(sys.argv) > 2 else "Meeting"
            language = sys.argv[3] if len(sys.argv) > 3 else None

            assistant.process_meeting(audio_file, meeting_title, language)
        else:
            # Interactive mode
            assistant.interactive_mode()

    except KeyError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY or MEETING_ASSISTANT_API_KEY in your .env file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up LLM provider resources
        close_llm_provider()


if __name__ == "__main__":
    main()
