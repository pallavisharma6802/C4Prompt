# Prompt Compressor (MVP)

A small web app to compress prompts for LLMs using simple heuristics (remove filler words, collapse repeated phrases into references, and count tokens). This is a minimal MVP inspired by the Reddit post about prompt compression.

Quick start (macOS / zsh):

1. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app.main:app --reload
```

4. Open http://127.0.0.1:8000/static/index.html in your browser.

Notes:
- Token counting uses a simple word-based fallback if `tiktoken` is not installed. If you want accurate OpenAI token counts, install `tiktoken`.
- This project implements conservative heuristics only. For more aggressive compression you can add a model-based rewrite step.
