# AI Meeting Assistant - Quick Start Guide

## Setup (5 minutes)

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**Not familiar with virtual environments?**
See [VIRTUAL_ENV_GUIDE.md](VIRTUAL_ENV_GUIDE.md) for a complete guide.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### Step 4: Run the Application
```bash
python src/main.py
```

## Features

- **Transcribe**: Convert meeting audio to text (Whisper API)
- **Summarize**: Generate concise meeting summaries (GPT-3.5 Turbo)
- **Extract**: Identify key points, action items, and participants
- **Report**: Export comprehensive meeting reports in text and JSON
- **Record**: Capture meetings directly from your microphone

## Usage Examples

```bash
# Interactive mode
python src/main.py

# Process specific audio file
python src/main.py /path/to/audio.mp3 "Meeting Title" en

# Run examples
python examples.py
```

## Project Structure

```
ai-meeting-assistant/
├── venv/                    # Virtual environment (created during setup)
├── src/
│   ├── main.py             # Main application
│   ├── transcriber.py      # Audio → Text conversion
│   ├── summarizer.py       # Analysis & summarization
│   └── audio_processor.py  # Audio handling & reports
├── data/                   # Your audio files
├── output/                 # Generated reports
├── examples.py             # Usage examples
└── requirements.txt        # Python dependencies
```

## Next Steps

1. See [README.md](README.md) for complete documentation
2. Check [CONCEPTS_AND_WORKFLOW.md](CONCEPTS_AND_WORKFLOW.md) for learning material
3. Check [VIRTUAL_ENV_GUIDE.md](VIRTUAL_ENV_GUIDE.md) to understand virtual environments better
4. Run `python examples.py` to see usage examples
5. Read the docstrings in `src/` files for API details

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure `.env` file exists with your API key

**"Module not found"**
- Did you activate the virtual environment? (`source venv/bin/activate`)
- Did you run `pip install -r requirements.txt`?

**Audio recording fails**
- Check microphone permissions in system settings
- Install sounddevice: `pip install sounddevice soundfile`

## Virtual Environment Reminder

Always activate the virtual environment before working:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**Don't know what a virtual environment is?**
→ Read [VIRTUAL_ENV_GUIDE.md](VIRTUAL_ENV_GUIDE.md)
