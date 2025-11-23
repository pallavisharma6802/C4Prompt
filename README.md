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
cd prompt_compressor
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
prompt_compressor/
├── main.py
├── config/fillers.json
├── core/
│   ├── rule_based.py
│   ├── extractive.py
│   └── validator.py
└── utils/co2_estimator.py
```


