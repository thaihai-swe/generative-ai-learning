# Generative AI Learning Repository

A collection of simple yet practical applications built while learning Generative AI technologies. This repository demonstrates various AI/ML concepts including image captioning, function calling with LLMs, chatbots, and more.

## 📚 Table of Contents

- [Projects Overview](#projects-overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Project Details](#project-details)
- [Resources](#resources)

## 🎯 Projects Overview

| Project | Description | Tech Stack |
|---------|-------------|-----------|
| **Caption Picture** | Generates captions for images using pre-trained models | Python, Gradio, Transformers, Pillow |
| **Function Calling** | AI-powered stock analysis using OpenAI function calling | Python, OpenAI API, yfinance, Pydantic |
| **Telegram Bot Chat** | Multi-user Telegram chatbot with custom personas | Python, Telegram API, LLM (OpenAI-compatible) |
| **Python ChatGPT Console App** | Interactive console application for ChatGPT conversations | Python, OpenAI API |
| **Simple Vector DB** | Lightweight vector database implementation | Python |
| **Hugging Face Examples** | Collection of HuggingFace Transformers tutorials | Python, Transformers, Datasets |
| **Chroma DB** | Vector database with persistence | Python, Chroma |

## 📋 Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Virtual Environment** (recommended)
- API Keys (as needed for specific projects):
  - OpenAI API key (for ChatGPT and function-calling projects)
  - Telegram Bot Token (for telegram-bot-chat)

## 📁 Repository Structure

```
generative-ai-learning/
├── caption-picture/              # Image captioning application
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── function-calling/             # Stock analysis with function calling
│   ├── function-calling.py
│   ├── requirements.txt
│   └── README.md
├── telegram-bot-chat/            # Telegram chatbot
│   ├── bot.py
│   ├── requirements.txt
│   └── README.md
├── python-chatgpt-console-app/   # ChatGPT console interface
│   ├── src/
│   │   ├── main.py
│   │   ├── chatgpt_client.py
│   │   ├── context_manager.py
│   │   ├── utils.py
│   │   └── temp.py
│   ├── requirements.txt
│   └── README.md
├── hugging-face/                 # HuggingFace Transformers examples
│   ├── AutoClasses and AutoTokenizer.py
│   ├── Document Q&A pdf.py
│   ├── Loading datasets.py
│   ├── Manipulating datasets.py
│   ├── Text classification.py
│   ├── text generation pipeline.py
│   └── Text summarization.py
├── simple-vector-db/             # Vector database implementation
│   └── simple-vector-db.py
├── chroma_db/                    # Chroma vector database
│   └── chroma.sqlite3
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd generative-ai-learning
```

### 2. Choose a Project
Each project has its own directory with dependencies. Navigate to your desired project:

```bash
cd <project-name>
```

### 3. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys (if needed)
Create a `.env` file in the project directory:
```bash
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_token  # For telegram-bot-chat
```

## 📖 Project Details

### 🖼️ Caption Picture
Generates descriptive captions for images using Hugging Face transformers and a Gradio web interface.

**Features:**
- Enter image URL to generate captions
- Pre-trained image captioning model
- Simple web interface with Gradio

**Quick Start:**
```bash
cd caption-picture
pip install -r requirements.txt
python app.py
```

---

### 📊 Function Calling
Demonstrates advanced LLM function calling with real-time stock analysis, including price retrieval and technical trend analysis.

**Features:**
- Stock symbol lookup
- Real-time price retrieval
- Technical trend analysis
- Investment recommendations
- Interactive CLI

**Quick Start:**
```bash
cd function-calling
pip install -r requirements.txt
python function-calling.py
```

**Requirements:**
- OpenAI API key
- yfinance for stock data

---

### 🤖 Telegram Bot Chat
Multi-user Telegram chatbot with streaming responses and customizable personas.

**Features:**
- Multi-user support with separate contexts
- Streaming responses
- Custom personas (assistant, coder, teacher, creative, analyst)
- Conversation management
- Custom system prompts

**Quick Start:**
```bash
cd telegram-bot-chat
pip install -r requirements.txt
# Set TELEGRAM_BOT_TOKEN in .env
python bot.py
```

---

### 💬 Python ChatGPT Console App
Interactive console application for engaging in ChatGPT conversations.

**Features:**
- Real-time chat interface
- Context management
- Easy setup

**Quick Start:**
```bash
cd python-chatgpt-console-app
pip install -r requirements.txt
# Set OPENAI_API_KEY in .env
python src/main.py
```

---

### 🧠 Hugging Face Examples
Collection of tutorials covering various Hugging Face Transformers use cases:

- **AutoClasses and AutoTokenizer.py** - Model and tokenizer auto-loading
- **Document Q&A pdf.py** - Question answering on PDF documents
- **Loading datasets.py** - Working with HuggingFace datasets
- **Manipulating datasets.py** - Data preprocessing and manipulation
- **Text classification.py** - Text classification tasks
- **text generation pipeline.py** - Text generation pipelines
- **Text summarization.py** - Document summarization

---

### 🗄️ Simple Vector DB
Lightweight vector database implementation for learning purposes.

---

### 💾 Chroma DB
Vector database with persistent storage capabilities.

---

## 💡 Learning Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://gradio.app/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Chroma Documentation](https://docs.trychroma.com/)

## 🛠️ Development Tips

- Use virtual environments to isolate dependencies
- Keep API keys in `.env` files and add them to `.gitignore`
- Each project is self-contained with its own `requirements.txt`
- Run projects individually and test them thoroughly

## 📝 Notes

- This repository is for learning and experimentation purposes
- Some projects require API keys (OpenAI, Telegram)
- Make sure to review each project's README for specific setup instructions
- API usage may incur costs depending on the service

## 📄 License

This repository is for educational purposes.

---

**Happy Learning! 🚀**
