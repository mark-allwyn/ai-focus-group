# Focus Group Chatbot (Streamlit)

Get diverse, first-person perspectives from multiple personas, all “in the same room.” Ask a question once, watch personas respond in sequence (random first speaker each turn), and **re-ask the same question** so personas can revise their answers with “self memory.”

---

## ✨ Features

* **Multi-persona chat**: several predefined personas answer in-character, concisely, and in **first person** (no emojis).
* **Random speaking order per turn**: first speaker is chosen randomly each time; later speakers can see earlier replies **from the same turn** and react.
* **Re-ask same question**: one-click **self-memory re-run** of the last question. Each persona sees their **own** previous answer and can update it.
* **No cross-turn carryover (by default)**: each new question starts fresh unless you use the “Re-ask” feature.
* **Optional previous-turn context**: in Chat, tick **Use previous turn as context** to share the last question and trimmed persona replies with the group, so answers can build on them.
* **Personas from file**: personas are loaded from `personas.json`. Editing/deleting personas in the UI **persists** back to that file.
* **Persona management**: activate/deactivate, edit, and delete personas; tooltips explain each field.
* **Transcript tools**: download as `.txt`, clear history, and **summarize the full conversation**.
* **Multi-model**: choose default LLM (OpenAI, Gemini, Claude). Per-persona overrides supported.
* **Optional web search**: Tavily-backed `web_search` tool (via OpenAI tools) for fresh facts; citations woven into first-person persona answers.

---

## 🚀 Quickstart

### 1) Prerequisites

* Python **3.10 – 3.12** recommended
* A modern browser
* At least one API key (OpenAI, Gemini, or Claude). Tavily key is optional (required only for web search).

### 2) Install

```bash
# Clone your repo
git clone https://github.com/<your-org-or-user>/persona-focus-group.git
cd persona-focus-group

# Create & activate venv (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt  # or: pip install streamlit openai google-generativeai anthropic requests
```

### 3) Configure secrets

Create `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml

# Use at least one of these providers. The app lets you pick the default in the sidebar.
OPENAI_API_KEY = "sk-...optional"
GEMINI_API_KEY = "AIza...optional"
CLAUDE_API_KEY = "sk-ant-...optional"

# Optional: for the web_search tool when personas need fresh info
TAVILY_API_KEY = "tvly-...optional"
```

> You only need **one** of the model keys to start. Add Tavily if you want web search.

### 4) (Optional) Seed personas file

If `personas.json` is missing or invalid, the app writes seed personas automatically.
To provide your own upfront, create `personas.json` in the repo root:

```json
[
  {
    "name": "Zara Chen",
    "age_group": "Gen Z",
    "occupation": "Computer Science Student",
    "location": "San Francisco, CA",
    "cultural_background": "Chinese-American",
    "education_level": "Bachelor's",
    "personality_traits": ["Creative","Tech-savvy","Idealistic","Impatient"],
    "hobbies": ["Gaming","TikTok creation","Anime","Hackathons"],
    "tech_savviness": "Expert",
    "goals_motivations": "Build innovative apps that make a social impact",
    "pain_points": "Student debt, finding authentic connections, climate anxiety",
    "speaking_style": "Casual with tech slang, direct",
    "affinity_keywords": ["innovation","sustainability","diversity","disruption"],
    "llm_model": "Default"
  }
  // ...add more personas
]
```

### 5) Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## 🧭 Using the App

### Chat tab

* **Ask Personas**: type your question and click “Ask Personas.”

  * The **first speaker is random** each turn.
  * Later speakers see earlier replies from this turn and may agree/disagree or add nuance.
* **Re-ask same question**: re-runs the **last question**.
  Each persona sees **their own previous answer** and is instructed to **reflect and update** (do not repeat verbatim; say what they keep/change and why). A new turn is saved and labeled `"(self-memory re-run)"`.

### Persona Management tab

* **Activate/Deactivate** personas (these are the ones who will respond in the next turn).
* **Edit** persona fields (tooltips explain each field).
  Changes are **saved back to `personas.json`** automatically.
* **Delete** a persona (also persisted to `personas.json`).

### Transcript tab

* **Download Transcript (.txt)**: saves the full chat history with speakers in order.
* **Clear Transcript**: removes all turns from this session.
* **Summarize Conversation**: generates an overview, highlights, and action items for the entire history.

---

## ⚙️ Configuration Details

* **Default Model**: Choose OpenAI, Gemini, or Claude from the **sidebar**.
  Each persona can optionally set its own `llm_model` (`"Default"`, `"OpenAI"`, `"Gemini"`, `"Claude"`).
* **Web Search**: When using OpenAI as the backend for a persona, answers can call a `web_search` tool powered by **Tavily**. Set `TAVILY_API_KEY` to enable.
* **First-Person Enforcement**: The app wraps prompts so personas always respond **in first person**, with no emojis, even when tools are used.
* **No Cross-Turn Carryover**: By default, new questions do **not** inherit prior-turn context. Use **Re-ask same question** when you want per-persona “self memory.”

---

## 🧩 Persona Schema (fields & what they influence)

* **name**: Unique display name (used in cards and transcript).
* **age\_group**: “Gen Z”, “Millennial”, “Gen X”, “Boomer”, “Silent” (guides tone/examples).
* **occupation**: Role/career lens.
* **location**: E.g., “Austin, TX” or “London, UK”.
* **cultural\_background**: Identity descriptors that shape viewpoint.
* **education\_level**: “High School”, “Trade School”, “Bachelor’s”, “Master’s”, “PhD”.
* **personality\_traits**: Short adjectives, e.g., “Curious, Pragmatic, Empathetic”.
* **hobbies**: Interests that flavor examples/analogies.
* **tech\_savviness**: “Low”, “Moderate”, “High”, “Expert”.
* **goals\_motivations**: What drives the persona (1–2 sentences).
* **pain\_points**: Constraints/frustrations.
* **speaking\_style**: Voice/tone; e.g., “Professional but warm; uses analogies”.
* **affinity\_keywords**: Values/themes they emphasize (e.g., “sustainability, ROI, community”).
* **llm\_model**: “Default”, “OpenAI”, “Gemini”, “Claude”.

> All fields are editable in the UI with inline tooltips. Changes persist to `personas.json`.

---

## 🧱 Architecture (high level)

* **Streamlit UI** with three tabs (Chat / Persona Management / Transcript).
* **State**: personas, active set, turns (each turn stores the question and a dict of `{persona_name: answer}`).
* **Speaking engine**:

  * Randomize active personas’ order per turn.
  * Build intra-turn context so later speakers can react to earlier replies.
* **Self-memory re-run**:

  * Uses the **last turn**.
  * Each persona sees their own previous answer and is nudged to revise/extend it.
  * Saves results as a new turn.

---

## 🧪 Troubleshooting

* **“ERROR: OpenAI/Gemini/Claude API key not found”**
  Add the corresponding key to `.streamlit/secrets.toml`. You need at least one provider key.
* **Tavily search not working**
  Add `TAVILY_API_KEY` to `.streamlit/secrets.toml` (optional, only needed for web search).
* **ModuleNotFoundError**
  Ensure you activated your venv and installed dependencies.
* **Nothing happens on click**
  Check the terminal for errors. Also verify you have at least **one active persona** in Persona Management.

---

## 🔒 Security & Privacy

* Keep API keys in `.streamlit/secrets.toml` (never commit them).
* Consider redacting transcripts before sharing if they include sensitive content.
* Rate limits and costs depend on your model provider(s).

---

## 🗂 Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.venv/
.env/

# Streamlit secrets
.streamlit/secrets.toml

# OS / IDE
.DS_Store
.vscode/
.idea/
```

---

## 📜 License

Choose a license (e.g., MIT) and add it as `LICENSE`. Example MIT blurb:

```
MIT License — Copyright (c) <Year> <Your Name>
```

---

## 🙌 Contributing

PRs welcome! Ideas: persona galleries, per-turn sliders for temperature/length, richer analytics, or export to PDF/CSV.

---

## 💬 Questions?

Open an issue in the repo with:

* Steps to reproduce
* Streamlit version (`streamlit --version`)
* Python version (`python --version`)
* A snippet of your `personas.json` (if relevant)


