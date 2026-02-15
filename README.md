# langchain-quiz

A CLI tool that generates multiple-choice quizzes from text files using LangChain and local LLMs via Ollama. Optionally enriches questions with web search via Tavily.

## Quick Start

```bash
pip install -r requirements.txt
python quiz.py article.txt
```

This generates 5 medium-difficulty questions and runs an interactive quiz in your terminal.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with at least one model pulled

## Usage

```bash
# Interactive quiz from a file
python quiz.py article.txt

# Pipe text in
curl -s https://example.com/article | python quiz.py

# Customize generation
python quiz.py article.txt -m qwen3:8b -n 10 -d hard

# JSON output
python quiz.py article.txt --json

# Web-enriched mode (deeper questions with source citations)
python quiz.py article.txt --web --tavily-key tvly-YOUR_KEY
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Ollama model name | `llama3` |
| `-u, --url` | Ollama base URL | `http://localhost:11434` |
| `-n, --num-questions` | Number of questions | `5` |
| `-d, --difficulty` | `easy`, `medium`, or `hard` | `medium` |
| `-t, --temperature` | LLM temperature | `0.7` |
| `--json` | Output as JSON instead of interactive mode | off |
| `--web` | Enrich questions with Tavily web search | off |
| `--tavily-key` | Tavily API key (also reads `TAVILY_API_KEY` env var) | - |

## Web-Enriched Mode

With `--web`, the tool:

1. Extracts key topics from your text using the LLM
2. Searches each topic via [Tavily](https://tavily.com) for supplementary context
3. Feeds both the original text and web results into quiz generation

This produces deeper questions and richer explanations with source URLs. It works without a Tavily key by default -- just omit `--web` and it behaves like a standard offline quiz generator.

## JSON Output Format

```json
{
  "questions": [
    {
      "question": "What is the primary mirror of the JWST made of?",
      "options": ["Aluminum", "Gold-plated beryllium", "Glass", "Carbon fiber"],
      "correct_index": 1,
      "explanation": "The JWST uses 18 hexagonal gold-plated beryllium segments...",
      "sources": ["https://example.com/jwst-specs"]
    }
  ]
}
```

The `sources` field is populated when using `--web` mode, empty otherwise.
