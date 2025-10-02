# Strands Meta-Agent (Ollama, Assistants-as-Tools, Model Switching)

Production-ready skeleton for a meta-agent builder using Strands-style assistants,
local Ollama models (Gemma3 270M/1B, Llama 3.2 1B/3B), and role-based routing.
See `config/models.yaml` to choose defaults and `app/main.py --list-models/--switch` for runtime switching.

## Quick start
1) Install Ollama and pull models:
   ```bash
   ollama serve &
   ollama pull gemma3:270m
   ollama pull gemma3:1b
   ollama pull llama3.2
   ollama pull llama3.2:1b
   ```
2) Python env and deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install httpx pyyaml pandas openpyxl
   export OLLAMA_BASE_URL="http://127.0.0.1:11434"
   ```
3) Run mentor loop demo:
   ```bash
   python app/main.py --list-models
   python app/main.py --switch coordinator llama3_2_1b
   python app/main.py
   ```

Excel intake: Put answers in `intake/intake.xlsx` then run:
```bash
python app/intake/excel_intake.py --xlsx app/intake/intake.xlsx
```
