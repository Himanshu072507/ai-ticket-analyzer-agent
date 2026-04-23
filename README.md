# AI Ticket Analyzer Agent

Upload an Excel/CSV of support tickets and get a crisp AI summary with charts, KPIs, and prioritized action items.

## Setup

```bash
cd ~/ticket-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a free key at https://aistudio.google.com/app/apikey. You can also paste it directly into the app's sidebar at runtime instead of using a .env file.

### Local (no-key) option — Ollama

If you have [Ollama](https://ollama.com) installed, select **Ollama (local)** in the sidebar to run fully offline.

```bash
ollama pull llama3.1:8b   # one-time, ~4.9 GB
ollama serve              # usually auto-starts
```

The app probes `http://localhost:11434` and lists your installed models in a dropdown.

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Upload `sample_tickets.xlsx` to try it immediately.

## What it does

1. Auto-detects ticket columns (category, description, department, date, priority, status). Let's you remap if detection is off.
2. Renders KPI cards + 5 charts: category bar, department bar, volume over time, status donut, category x department heatmap.
3. Calls either Gemini (`gemini-2.5-flash` with fallback to 2.0-flash / flash-lite) or a local Ollama model to produce: top themes, priority ranking, department spotlight, and concrete recommendations.
4. Export summary as Markdown.

## Files

- `app.py` — Streamlit UI
- `analyzer.py` — Gemini + Ollama backends, structured JSON output
- `charts.py` — Plotly chart builders
- `sample_tickets.xlsx` — 50 realistic sample rows
