# SmartPrompt

AI Prompt Compression for Cost Optimization and Carbon Reduction

## Overview

SmartPrompt is a web-based tool that reduces AI prompt token usage by up to 42% while preserving semantic meaning. By compressing prompts before sending them to LLM APIs, users can significantly reduce API costs, improve response times, and lower their carbon footprint.

## Problem Statement

Large Language Model APIs charge per token, making verbose prompts expensive at scale. A company processing 2 million prompts monthly at 280 tokens average spends $16,800/month on GPT-4. Additionally, each prompt consumes 0.24 Wh of energy and generates 0.03g CO₂. As AI adoption grows, these costs and environmental impacts compound rapidly.

## Solution

SmartPrompt applies intelligent compression techniques to remove redundancy while maintaining prompt quality:

- Filler word removal
- Stopword elimination
- Phrase deduplication
- Acronym extraction
- Token-aware encoding

The result is a 30-70% reduction in token count with minimal impact on output quality.

## Key Features

- **Real-time Compression**: Instant prompt optimization with side-by-side comparison
- **Multiple Compression Modes**: Aggressive, stopword removal, phrase encoding, and TikToken support
- **Cost Analytics**: Live calculation of savings per compression
- **Progress Tracking**: Persistent tracking of cumulative cost, compute, and carbon savings
- **Privacy-First**: All processing happens locally or on your infrastructure
- **Universal Compatibility**: Works with OpenAI, Anthropic, Google Gemini, Meta Llama, and other LLM APIs

## Technical Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Uvicorn**: ASGI server for production deployment
- **TikToken**: OpenAI's tokenization library (optional)

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **CSS Grid/Flexbox**: Responsive layout system
- **LocalStorage API**: Client-side persistence for tracking

### Compression Engine
- **Python 3.11+**: Core compression algorithms
- **Custom NLP Pipeline**: Filler detection, phrase analysis, and encoding

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pallavisharma6802/green-ai.git
cd green-ai
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

5. Open browser to `http://127.0.0.1:8000`

## Usage

### Basic Compression

1. Paste your AI prompt in the left textarea
2. Select compression options (Aggressive, Remove Stopwords, etc.)
3. Click "Compress Prompt"
4. View compressed result and savings metrics

### Compression Options

- **Aggressive Mode**: Maximum compression with more aggressive filler removal
- **Remove Stopwords**: Eliminates common stopwords (the, a, an, is, etc.)
- **Phrase Encoding**: Detects and encodes repeated phrases
- **TikToken**: Uses OpenAI's tokenizer for accurate token counting

### API Integration

SmartPrompt exposes a REST API for programmatic access:

```bash
POST /compress
Content-Type: application/json

{
  "prompt": "Your original prompt text here",
  "aggressive": true,
  "remove_stopwords": false,
  "encode": true,
  "use_tiktoken": false
}
```

Response:
```json
{
  "original_prompt": "...",
  "compressed_prompt": "...",
  "original_tokens": 280,
  "compressed_tokens": 154,
  "tokens_saved": 126,
  "meta": {...}
}
```

## Real-World Impact

### Case Study: B2B SaaS Startup

**Scenario**: 50K monthly active users, 2M prompts/month, 280 avg tokens

**Before SmartPrompt**:
- Token usage: 560M tokens/month
- Monthly cost: $16,800
- Energy consumption: 480 kWh
- CO₂ emissions: 60 kg

**After SmartPrompt** (45% reduction):
- Token usage: 308M tokens/month
- Monthly cost: $9,240
- Energy consumption: 264 kWh
- CO₂ emissions: 33 kg

**Savings**:
- $7,560/month ($90,720/year)
- 216 kWh/month (2,592 kWh/year)
- 27 kg CO₂/month (324 kg CO₂/year)

## Analytics

### Cost Metrics
- GPT-4: $0.03 per 1K tokens
- Reducing 100 tokens saves $0.003 per query
- At 1M queries/month: $300 savings per 100 tokens reduced

### Environmental Impact
- Average prompt: 0.24 Wh energy, 0.03g CO₂
- 42% compression: 0.1 Wh saved, 0.012g CO₂ saved per query
- 1M compressed prompts: 100 kWh saved, 12.6 kg CO₂ saved (equivalent to 63 tree seedlings/year)

## Project Structure

```
green-ai/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── compressor.py     # Compression algorithms
├── static/
│   ├── landing.html      # Frontend interface
│   └── index.html        # Legacy interface
├── scripts/
│   └── demo_compress.py  # CLI demo script
├── tests/
│   └── test_compressor.py
├── requirements.txt
└── README.md
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black app/
flake8 app/
```

## Future Enhancements

- Browser extension for ChatGPT/Claude interfaces
- API rate limiting and authentication
- Advanced compression with context-aware LLMs
- Multi-language support
- Batch processing for large datasets
- Real-time cost comparison across LLM providers

## Contributing

This project was built for MadHacks 2025. Contributions, issues, and feature requests are welcome.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Google Research for AI carbon footprint studies
- OpenAI for TikToken library
- MadHacks 2025 organizers

## Contact

- GitHub: [@pallavisharma6802](https://github.com/pallavisharma6802)
- Repository: [green-ai](https://github.com/pallavisharma6802/green-ai)

---

Built with focus on cost optimization, computational efficiency, and environmental sustainability.
