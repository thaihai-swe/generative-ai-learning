# AI Meeting Assistant

An intelligent business meeting assistant that transcribes audio recordings and generates comprehensive meeting summaries, key points, action items, and participant lists using OpenAI's Whisper and supported LLM providers (OpenAI, Ollama, LocalAI, vLLM, etc.).

**v2.0** - Now powered by [base-provider](../base-provider) for flexible multi-provider LLM support! 🚀

## Features

- **Audio Transcription**: Convert meeting audio to text using OpenAI's Whisper API
  - Supports multiple audio formats (MP3, WAV, M4A, MP4, WebM, etc.)
  - Automatic language detection or manual language specification

- **Meeting Summarization**: Generate concise summaries of meeting content
  - Multiple summary styles (concise, detailed, bullet-point)
  - Customizable summary length
  - **NEW**: Works with any OpenAI-compatible LLM provider!

- **Key Points Extraction**: Automatically extract the most important discussion points

- **Action Items**: Identify tasks, responsibilities, and deadlines mentioned in the meeting

- **Participant Identification**: Automatically extract and list meeting participants

- **Recording**: Record meetings directly from your microphone

- **Report Generation**: Export comprehensive meeting reports in text and JSON formats

- **Meeting History**: Maintain a searchable history of all processed meetings

- **Multi-Provider Support** ✨ **NEW in v2.0**
  - OpenAI (GPT-4, GPT-3.5-Turbo)
  - Ollama (run locally for free!)
  - LocalAI
  - vLLM
  - Any OpenAI-compatible endpoint

## What's New in v2.0

The application has been refactored to use [base-provider](../base-provider), a flexible LLM abstraction layer:

✨ **Multi-Provider Support**
- Switch between OpenAI, Ollama, LocalAI, vLLM without changing code
- Use environment variables to configure provider and settings
- Support for OpenAI-compatible endpoints

🎯 **Better Architecture**
- Centralized LLM configuration in `src/config.py`
- Cleaner, more maintainable code using abstract interfaces
- Full backward compatibility (existing `.env` still works)

💰 **Cost Savings**
- Use free local models (Ollama, LocalAI) instead of OpenAI
- Run everything on your machine for privacy
- Easy to test with different models

📖 **Full Documentation**
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - Migration guide from v1.0
- [VIRTUAL_ENV_GUIDE.md](VIRTUAL_ENV_GUIDE.md) - Virtual environment best practices
- See [base-provider/README.md](../base-provider/README.md) for provider docs

**Version History:**
- v2.0 (Latest) - Multi-provider support with base-provider
- v1.0 - Initial release with direct OpenAI integration

For a detailed upgrade guide, see [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md).

## Project Structure

```
ai-meeting-assistant/
├── src/
│   ├── main.py              # Main application entry point
│   ├── transcriber.py       # Audio transcription module
│   ├── summarizer.py        # Meeting analysis and summarization
│   ├── audio_processor.py   # Audio handling and report generation
│   └── __init__.py
├── data/                    # Directory for audio recordings
├── output/                  # Directory for generated reports
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
└── README.md               # This file
```

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Virtual environment manager (venv comes with Python)

### 2. Setup with Virtual Environment

1. Clone or download the project to your local machine

2. Create a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv

# On Windows
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

4. Install dependencies in the virtual environment:
```bash
pip install -r requirements.txt
```

5. **NEW in v2.0**: Install base-provider for multi-provider LLM support:
```bash
# From ai-meeting-assistant directory
pip install -e ../base-provider
```

6. Create a `.env` file in the project root directory:
```bash
cp .env.example .env
```

7. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

Or configure for other providers (see Configuration below)

8. Deactivate the virtual environment when done:
```bash
deactivate
```

**Why use a virtual environment?**
- Isolates project dependencies from system Python
- Prevents conflicts between different projects
- Makes it easy to reproduce the exact environment
- Best practice for Python development

## Configuration

### Environment Variables

The application now supports multiple LLM providers through environment variables.

#### Option 1: OpenAI (Default)
```bash
# .env
OPENAI_API_KEY=sk-your-key-here
MEETING_ASSISTANT_MODEL=gpt-3.5-turbo  # Optional
```

#### Option 2: Ollama (Local & Free!)
```bash
# First, start Ollama: ollama serve
# Then pull a model: ollama pull llama2

# .env
MEETING_ASSISTANT_PROVIDER_TYPE=openai
MEETING_ASSISTANT_MODEL=llama2
MEETING_ASSISTANT_API_BASE_URL=http://localhost:11434/v1
```

#### Option 3: LocalAI
```bash
# .env
MEETING_ASSISTANT_PROVIDER_TYPE=openai
MEETING_ASSISTANT_MODEL=gpt4all-j
MEETING_ASSISTANT_API_BASE_URL=http://localhost:8080/v1
```

#### Option 4: vLLM
```bash
# .env
MEETING_ASSISTANT_PROVIDER_TYPE=openai
MEETING_ASSISTANT_MODEL=meta-llama/Llama-2-7b-chat-hf
MEETING_ASSISTANT_API_BASE_URL=http://localhost:8000/v1
```

### Switching Providers

The beauty of v2.0 is that you can switch providers **without changing any code** - just update `.env`:

```bash
# To use OpenAI:
OPENAI_API_KEY=sk-...

# To use Ollama (after stopping OpenAI):
MEETING_ASSISTANT_API_BASE_URL=http://localhost:11434/v1
MEETING_ASSISTANT_MODEL=llama2
```

Same code, different provider!

## Usage

### Interactive Mode

Run the application in interactive mode:
```bash
python src/main.py
```

This will present a menu with options to:
1. Process existing audio files
2. Record new meetings
3. View meeting history
4. Exit

### Command Line Mode

Process a specific audio file directly:
```bash
python src/main.py /path/to/audio.mp3 "Meeting Title" en
```

Arguments:
- `audio_file`: Path to the audio file (required)
- `meeting_title`: Title for the meeting (optional, default: "Meeting")
- `language`: ISO-639-1 language code (optional, auto-detected if omitted)

### Example Workflow

1. **Record a meeting:**
   ```bash
   python src/main.py
   # Select option 2 (Record new meeting)
   # Enter meeting title and duration
   ```

2. **Or use an existing recording:**
   ```bash
   python src/main.py
   # Select option 1 (Process existing audio file)
   # Select or enter the audio file path
   ```

3. **View results:**
   - Text report saved in `output/` directory
   - JSON report for programmatic access
   - Full transcription included in reports

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- MP4 (.mp4)
- MPEG (.mpeg)
- MPGA (.mpga)
- WebM (.webm)

## Generated Reports

The assistant generates two types of reports:

### Text Report
A formatted document containing:
- Meeting title and timestamp
- Participant list
- Executive summary
- Key points (typically 5)
- Action items with assignments
- Full meeting transcription

### JSON Report
Machine-readable format containing:
- summary
- key_points (list)
- action_items (string)
- participants (list)
- transcription (full text)
- audio_file (filename)
- meeting_title

## Module Documentation

### MeetingTranscriber
Handles audio-to-text conversion:
```python
from transcriber import MeetingTranscriber

transcriber = MeetingTranscriber()
result = transcriber.transcribe_file("meeting.mp3", language="en")
print(result['text'])
```

### MeetingSummarizer
Generates summaries and extracts information:
```python
from summarizer import MeetingSummarizer

summarizer = MeetingSummarizer()
summary = summarizer.summarize(transcription_text)
key_points = summarizer.extract_key_points(transcription_text)
action_items = summarizer.extract_action_items(transcription_text)
report = summarizer.generate_full_report(transcription_text)
```

### AudioFileManager
Manages audio files and metadata:
```python
from audio_processor import AudioFileManager

manager = AudioFileManager()
audio_files = manager.get_audio_files()
history = manager.get_meeting_history()
```

### ReportGenerator
Generates and saves reports:
```python
from audio_processor import ReportGenerator

generator = ReportGenerator()
text_path = generator.save_report_to_file("Meeting", report_data)
json_path = generator.save_report_as_json("Meeting", report_data)
```

## Cost Considerations

The application uses OpenAI's APIs:
- **Whisper API**: ~$0.006 per minute of audio
- **GPT-3.5 Turbo API**: ~$0.0015 per 1K input tokens, $0.002 per 1K output tokens

For a typical 1-hour meeting:
- Transcription: ~$0.36
- Summary + Analysis: ~$0.02-0.05
- **Total: ~$0.40-0.50 per meeting**

## Troubleshooting

### "OPENAI_API_KEY not found"
Ensure your `.env` file exists and contains:
```
OPENAI_API_KEY=your_actual_api_key
```

### "Unsupported format"
The audio file format is not supported. Convert to WAV or MP3 using tools like FFmpeg:
```bash
ffmpeg -i input.m4a -acodec libmp3lame -ab 192k output.mp3
```

### "Permission denied" for audio recording
Ensure your system allows Python to access the microphone. Grant microphone permissions in system settings.

### Poor transcription quality
- Ensure clear audio with minimal background noise
- Specify the correct language code
- Test with a shorter recording first

## For Developers & AI Learners

**New to AI and curious how this works?** Read [CONCEPTS_AND_WORKFLOW.md](CONCEPTS_AND_WORKFLOW.md) - A comprehensive guide covering:

- **Core AI Concepts**: Neural networks, LLMs, NLP, embeddings, attention mechanisms
- **How This Project Uses AI**: The complete AI pipeline from audio to reports
- **Detailed Workflow**: Step-by-step explanation of what happens at each stage
- **API Technologies**: Understanding Whisper and GPT-3.5 Turbo
- **Architecture & Design**: Code organization and design patterns
- **Code Deep Dive**: Understanding each module with detailed explanations
- **Best Practices**: Virtual environments, error handling, prompt engineering, testing
- **Learning Resources**: Recommended courses, papers, and next steps

This guide is designed for:
- ✅ AI learners wanting to understand how modern AI systems work
- ✅ Developers new to LLM APIs wanting to learn best practices
- ✅ Anyone curious about prompt engineering and API integration
- ✅ Students studying NLP and machine learning

**Quick navigation:**
- [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - **NEW**: v2.0 refactoring & multi-provider support
- [VIRTUAL_ENV_GUIDE.md](VIRTUAL_ENV_GUIDE.md) - Understand virtual environments
- [CONCEPTS_AND_WORKFLOW.md](CONCEPTS_AND_WORKFLOW.md) - Learn the AI concepts & architecture
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Index of all docs
- [../base-provider/README.md](../base-provider/README.md) - base-provider documentation
- [examples.py](examples.py) - See code examples

## Advanced Configuration

You can customize the behavior by modifying parameters in the code:

### Transcriber Settings
- Change the Whisper prompt in `transcriber.py`
- Add language-specific prompts for better results

### Summarizer Settings
- Adjust `max_tokens` for longer/shorter summaries
- Modify `temperature` for different creativity levels (0.3-0.7)
- Change the number of key points extracted

### Audio Settings
- Modify sample rate in `AudioRecorder` (default 16000 Hz)
- Change number of channels (1=mono, 2=stereo)

## Future Enhancements

Possible improvements to the application:
- Real-time transcription with streaming audio
- Speaker diarization (identifying who said what)
- Custom keywords and topic detection
- Integration with calendar systems
- Email delivery of reports
- Multi-language support for mixed-language meetings
- Vector database for searching past meetings
- Custom templates for different meeting types

## License

This project is part of the Generative AI Learning series.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the OpenAI API documentation: https://platform.openai.com/docs
3. Check audio file format compatibility

## API Reference

### MeetingAssistant

#### `process_meeting(audio_file_path, meeting_title, language=None)`
Process a complete meeting and generate reports.

**Parameters:**
- `audio_file_path` (str): Path to audio file
- `meeting_title` (str): Title for the meeting
- `language` (str, optional): ISO-639-1 language code

**Returns:** dict with keys: summary, key_points, action_items, participants, transcription

#### `interactive_mode()`
Start the interactive menu system.

### TranscriptionResult

#### `paragraphs`
List of text paragraphs from transcription

#### `sentences`
List of individual sentences from transcription

## Changelog

### Version 1.0.0
- Initial release
- Audio transcription with Whisper
- Meeting summarization
- Key point extraction
- Action item identification
- Participant detection
- Report generation (text and JSON)
- Interactive and command-line modes
- Meeting history tracking
