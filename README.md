<<<<<<< HEAD
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
=======
# GreenTok

Compress LLM prompts to reduce tokens and CO2 emissions.

## Overview

GreenTok removes unnecessary words from prompts before sending them to language models. It uses rule-based filtering and extractive summarization to achieve 30-70% compression while preserving meaning.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: Add API key for real-time carbon intensity
export ELECTRICITY_MAPS_API_KEY='your-key'
```

## Usage

```bash
cd backend
python main.py
```

## How It Works

1. **Rule-based compression** - Removes greetings, hedging, politeness markers, and fillers
2. **Extractive summarization** - Keeps semantically important sentences using embeddings
3. **Semantic validation** - Ensures compressed output preserves meaning (>0.75 similarity)
4. **Energy tracking** - Calculates net CO2 savings using real-time grid data

## Configuration

Edit `backend/config/fillers.json` to customize patterns.

## Structure

```
backend/
├── main.py
├── config/fillers.json
├── core/
│   ├── rule_based.py
│   ├── extractive.py
│   └── validator.py
└── utils/co2_estimator.py
```


>>>>>>> ba129abcd812c1ae036c23ca4b6fdc1f05c11ec7
