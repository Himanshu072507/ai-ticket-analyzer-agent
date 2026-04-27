# AI Ticket Analyzer Agent

Upload an Excel/CSV of support tickets and get a crisp AI summary with charts, KPIs, and prioritized action items.

<img width="1470" height="832" alt="image" src="https://github.com/user-attachments/assets/b7601589-52fb-4f21-8ac0-431937f0faa5" />
<img width="1470" height="828" alt="image" src="https://github.com/user-attachments/assets/c0539522-d47d-4ac5-ba05-32502709c013" />
<img width="1470" height="832" alt="image" src="https://github.com/user-attachments/assets/68092c40-8b0f-4400-988b-85a3eb728e6a" />

## Setup

```bash
cd ~/ticket-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add GEMINI_API_KEY and/or GROQ_API_KEY
```

Pick whichever cloud key you want to use:

- **Gemini** — free key at https://aistudio.google.com/app/apikey
- **Groq** — free key at https://console.groq.com/keys (uses `llama-3.3-70b-versatile` by default)

You can also paste either key directly into the app's sidebar at runtime instead of using a `.env` file.

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
3. **AI Drill-Down**: a one-paragraph headline narrative, a ranked category list, and lazy per-category clustering. Click a category to see 3–5 AI-named sub-issues; expand a sub-issue for trend, examples, root cause, owner, and suggested fix.
4. Export summary as Markdown.

## Files

- `app.py` — Streamlit UI
- `analyzer.py` — Gemini + Groq + Ollama backends, structured JSON output
- `charts.py` — Plotly chart builders
- `sample_tickets.xlsx` — 50 realistic sample rows

## Running tests

```bash
source .venv/bin/activate
pip install pytest pytest-mock      # one-time
pytest -q                            # core suite (no real API calls)
pytest -q -m smoke                   # also run the Streamlit AppTest smoke test
```

Provider calls are mocked — no Gemini key or Ollama daemon required to run tests.

## Deploy to Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **New app**, pick repo `Himanshu072507/ai-ticket-analyzer-agent`, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   GROQ_API_KEY   = "your-groq-key"
   ```

   (Either key alone is enough; provide whichever provider you plan to use.)
4. Click **Deploy**. The first build installs everything in `requirements.txt` and gives you a public `https://*.streamlit.app` URL.

Ollama is local-only — pick **Gemini** or **Groq** in the deployed app. The Ollama option still appears but won't be able to reach a daemon from the cloud.
