# Virtual Environment Guide

A comprehensive guide to understanding and using Python virtual environments - essential for professional Python development.

---

## Table of Contents

1. [What is a Virtual Environment?](#what-is-a-virtual-environment)
2. [Why Use Virtual Environments?](#why-use-virtual-environments)
3. [Setup Instructions](#setup-instructions)
4. [Common Commands](#common-commands)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)
7. [Advanced Usage](#advanced-usage)

---

## What is a Virtual Environment?

### Simple Explanation

A **virtual environment** is a self-contained Python installation isolated from your system Python.

Think of it like this:
```
Your Computer
├── System Python (3.11.0)
│   └── global packages (numpy, pandas, django, etc.)
│
└── Virtual Environment for Project A (isolated copy)
    └── project-specific packages (fastapi==0.104.1, only)
```

### What Gets Isolated?

1. **Python interpreter**: Your venv has its own Python version
2. **Installed packages**: Each project can have different versions
3. **Scripts**: Executables are stored in the venv

### What Doesn't Get Isolated?

- System libraries (OS-level stuff)
- Your code and files
- Environment variables (unless configured)

---

## Why Use Virtual Environments?

### Problem 1: Version Conflicts

**Without virtual environment:**
```
Project A needs: numpy==1.20.0
Project B needs: numpy==1.24.0

You can only install one version globally!
→ One project breaks
```

**With virtual environment:**
```
Project A (venv-a)
└── numpy==1.20.0

Project B (venv-b)
└── numpy==1.24.0

Both work perfectly!
```

### Problem 2: System Contamination

**Without virtual environment:**
```bash
pip install some-library

# This installs globally
# If the library is buggy, your system Python is now broken
# Other projects using system Python are also affected
```

**With virtual environment:**
```bash
source venv/bin/activate
pip install some-library

# Only the venv is affected
# System Python stays clean
# Other projects unaffected
```

### Problem 3: Reproducibility

**Without virtual environment:**
```bash
# On your machine
pip install requests
# Latest version installed (whatever that is)

# On team member's machine
pip install requests
# Different latest version installed!
# Code behaves differently for different team members
```

**With virtual environment + requirements.txt:**
```bash
# requirements.txt specifies exact versions
requests==2.31.0

# On your machine
pip install -r requirements.txt
# Version 2.31.0 installed

# On team member's machine
pip install -r requirements.txt
# Same version 2.31.0 installed
# Code works identically!
```

### Summary: Why?

| Aspect | Without venv | With venv |
|--------|-------------|----------|
| **Package isolation** | ❌ Global conflicts | ✅ Fully isolated |
| **Reproducibility** | ❌ Everyone has different versions | ✅ Exact same versions |
| **System safety** | ❌ Can break system Python | ✅ System Python never touched |
| **Easy cleanup** | ❌ Hard to remove packages | ✅ Delete folder to reset |
| **Team collaboration** | ❌ "Works on my machine" | ✅ Works everywhere |

---

## Setup Instructions

### Step 1: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
```

**On Windows:**
```bash
python -m venv venv
```

**What just happened?**
```
Created folder: venv/
├── bin/                 (macOS/Linux) or Scripts/ (Windows)
│   ├── python          # Python interpreter
│   ├── pip             # Package manager
│   └── activate        # Activation script
├── lib/
│   └── python3.x/
│       └── site-packages/   # Package installation directory
└── pyvenv.cfg          # Configuration file
```

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**On Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**How do you know it worked?**
```bash
# Look at your terminal prompt
# It changes from:
(base) user@machine:~/project $

# To:
(venv) user@machine:~/project $
                 ↑
             This shows the venv name
```

### Step 3: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install specific packages
pip install openai python-dotenv
```

**What's happening:**
```
pip install openai

# With venv activated, pip installs to:
venv/lib/python3.x/site-packages/openai/

# NOT to /usr/local/lib/python3.x/site-packages/
```

### Step 4: Use the Virtual Environment

```bash
# Always activate before working
source venv/bin/activate

# Run your code
python src/main.py

# Or run with the full path (without activating)
./venv/bin/python src/main.py
```

### Step 5: Deactivate When Done

```bash
deactivate

# Prompt returns to normal
user@machine:~/project $
```

---

## Common Commands

### Check Which Python You're Using

```bash
# With venv activated
which python
# Output: /Users/name/project/venv/bin/python

# Without venv
which python
# Output: /usr/local/bin/python
```

### Check Active Packages

```bash
# Show all installed packages
pip list

# Output:
Package         Version
-----           -------
openai          1.3.8
python-dotenv   1.0.0
sounddevice     0.4.6
...
```

### Freeze Current Environment

```bash
# Save current environment state
pip freeze > requirements.txt

# This creates a file with exact versions:
# openai==1.3.8
# python-dotenv==1.0.0
# ...
```

### Install from requirements.txt

```bash
# Install all specified packages
pip install -r requirements.txt

# Update package
pip install --upgrade openai

# Uninstall package
pip uninstall openai
```

### Check Package Version

```bash
pip show openai

# Output:
Name: openai
Version: 1.3.8
Summary: The official Python library for the OpenAI API
Location: /Users/name/project/venv/lib/python3.11/site-packages
...
```

---

## File Structure

### Typical Project Layout

```
ai-meeting-assistant/           # Project folder
├── venv/                        # Virtual environment (created by you)
│   ├── bin/                     # Executables and activation scripts
│   ├── lib/                     # Installed packages
│   └── pyvenv.cfg              # Configuration
│
├── src/                         # Your code
│   ├── main.py
│   ├── transcriber.py
│   └── summarizer.py
│
├── data/                        # Your data files
├── output/                      # Generated files
├── requirements.txt             # Dependencies (COMMIT THIS)
├── .env                         # Secrets (DON'T COMMIT THIS)
├── .gitignore                   # Tell git to ignore venv/
└── README.md                    # Documentation
```

### .gitignore Setup

```bash
# Create or edit .gitignore file
echo "venv/" >> .gitignore

# This tells git to ignore:
venv/          # Never commit the venv folder
.env           # Never commit secrets
__pycache__/   # Python cache
*.pyc          # Compiled Python files
```

**Why not commit venv/?**
- It's large (100MB+)
- Platform-specific
- Easily recreated from requirements.txt
- Team members create their own

### Recreating Environment

```bash
# If you clone a project without venv/
git clone https://github.com/someone/project.git
cd project

# Create new venv
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Done! You have the exact same environment
```

---

## Troubleshooting

### Problem: "command not found: activate"

**You're probably on Windows using bash.**
```bash
# Windows Git Bash / MINGW
source venv/Scripts/activate

# NOT
source venv/bin/activate
```

### Problem: "pip: command not found" or "pip: permission denied"

```bash
# Make sure venv is activated
source venv/bin/activate

# The prompt should show (venv)
(venv) user@machine:~/project $

# If it doesn't, your venv isn't activated
# Try again with full path:
./venv/bin/python -m pip install package-name
```

### Problem: "ModuleNotFoundError: No module named 'openai'"

```bash
# Probable cause: venv not activated
# Check your prompt
(base) user@machine $           # ❌ Wrong venv
(venv) user@machine $           # ✅ Correct venv

# Activate it
source venv/bin/activate

# Install the module
pip install openai
```

### Problem: "Python version mismatch"

```bash
# Check your Python version
python --version
# Python 3.9.0

# Check what venv expects
cat venv/pyvenv.cfg
# home = /usr/local/bin
# version_info = 3.11.0
# ↑ Mismatch!

# Solution: Recreate venv with correct Python
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
```

### Problem: "pip install fails"

```bash
# Make sure pip is up to date
pip install --upgrade pip

# Try installing again
pip install openai

# Check for permission errors
# (should not happen inside venv, but just in case)
pip install --user openai  # Not recommended in venv!
```

### Problem: "venv takes too much space"

```bash
# If your venv is 1GB+ and taking up space
rm -rf venv
python3 -m venv venv --copies
source venv/bin/activate
pip install -r requirements.txt

# --copies uses less space than symlinks
```

---

## Best Practices

### 1. **Always Use Virtual Environments**

```bash
# ❌ DON'T
pip install requests  # Installs globally

# ✅ DO
python3 -m venv venv
source venv/bin/activate
pip install requests
```

### 2. **Name Your Virtual Environment "venv"**

```bash
# ✅ Standard name (everyone expects it)
python3 -m venv venv

# ❌ Avoid unusual names
python3 -m venv my-special-env  # Don't do this
```

### 3. **Freeze Dependencies**

```bash
# After installing packages
pip freeze > requirements.txt

# Team members can recreate your environment exactly
pip install -r requirements.txt
```

### 4. **Add venv/ to .gitignore**

```bash
# In .gitignore file
venv/
env/
ENV/
.venv/

# Don't commit the venv folder
```

### 5. **Update requirements.txt Regularly**

```bash
# When you add new packages
pip install new-package
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add new-package dependency"
```

### 6. **Test Your requirements.txt**

```bash
# Create fresh venv to test
python3 -m venv test-env
source test-env/bin/activate
pip install -r requirements.txt

# If it works, your requirements.txt is correct
rm -rf test-env
```

### 7. **Use requirements.txt for Production**

```bash
# ❌ DON'T use pip install without version specs
pip install flask

# ✅ DO use explicit versions
flask==3.0.0
```

### 8. **Document Python Version**

```bash
# In README.md or setup instructions
Python 3.8 or higher required

# Or in requirements file as comment
# Python 3.8+

# Or create setup.py with python_requires
python_requires='>=3.8'
```

---

## Advanced Usage

### Using Different Python Versions

```bash
# Check available Python versions
ls /usr/local/bin/python*
# python3.9, python3.10, python3.11

# Create venv with specific version
python3.11 -m venv venv

# Verify
source venv/bin/activate
python --version
# Python 3.11.0
```

### Using virtualenv (More Features)

```bash
# Install virtualenv
pip install virtualenv

# Create venv with virtualenv
virtualenv venv

# Or with specific Python version
virtualenv -p python3.11 venv

# Activate (same as venv)
source venv/bin/activate
```

### Using Poetry (Dependency Management)

```bash
# Modern alternative to venv + requirements.txt
pip install poetry

# Create project
poetry new my-project

# Or initialize in existing project
poetry init

# Install dependencies
poetry install

# Add new dependency
poetry add openai

# Creates pyproject.toml and poetry.lock
```

### Using Conda (From Anaconda)

```bash
# If using Anaconda/Miniconda
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Install packages
conda install openai

# List environments
conda env list
```

### System-wide vs User Installation

```bash
# ❌ NOT recommended - affects system
sudo pip install package

# ✅ Better - user installation
pip install --user package

# ✅ BEST - use virtual environment
source venv/bin/activate
pip install package
```

### Create reusable Template

```bash
# Create a template directory
mkdir python-template
cd python-template
python3 -m venv venv
source venv/bin/activate

# Create directory structure
mkdir src data output
touch README.md requirements.txt .gitignore

# Add to .gitignore
echo -e "venv/\n.env\n__pycache__/" > .gitignore

# Install common packages for your workflow
pip install python-dotenv requests

# Freeze
pip freeze > requirements.txt

# Now you have a template for future projects!
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Create venv | `python3 -m venv venv` |
| Activate (macOS/Linux) | `source venv/bin/activate` |
| Activate (Windows) | `venv\Scripts\activate` |
| Deactivate | `deactivate` |
| Install packages | `pip install -r requirements.txt` |
| Save packages | `pip freeze > requirements.txt` |
| Check packages | `pip list` |
| Remove package | `pip uninstall package-name` |
| List venv info | `pip show package-name` |
| Create with specific Python | `python3.11 -m venv venv` |
| Delete venv | `rm -rf venv` |

---

## Conclusion

Virtual environments are **essential** for professional Python development:

✅ **Use for every project**
✅ **Always commit requirements.txt**
✅ **Never commit venv/ folder**
✅ **Include venv/ in .gitignore**
✅ **Recreate from requirements.txt**

This practice ensures:
- Your code works on any machine
- Team members have identical environments
- No system Python pollution
- Easy onboarding for new developers

Happy coding! 🐍
