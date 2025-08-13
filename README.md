# Focus Group Chatbot (Streamlit)

Simulate a live **multi-persona focus group** on any question. Each turn picks a **random first speaker**, and all subsequent personas **see and can react** to earlier replies from that same turn. Answers are kept in **first-person voice** per persona (even when using web search), with **no carryover context** between questions by default.

---

## ✨ Features

* **Multiple personas** loaded from a `personas.json` file (add/edit/delete in the UI, persisted to disk).
* **Random speaking order** each turn to keep discussions fresh.
* **In-turn awareness:** later speakers can reference/critique earlier replies.
* **First-person voice** enforced across OpenAI, Gemini, and Claude paths.
* **Optional web facts** via Tavily Search API (tool-style call), woven naturally into persona responses.
* **Transcript export** to `.txt`.
* **Reload from file** button for hot-reloading `personas.json`.
* **Helpful tooltips** on every persona field for consistent data entry.

---

## 🧱 Tech

* **Streamlit** UI
* **OpenAI / Google Generative AI (Gemini) / Anthropic (Claude)** clients
* **Tavily Search API** for web results
* Python 3.10+ recommended

---

## 📦 Setup

### 1) Clone & create a virtual environment

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, this minimal set works:

```txt
streamlit
requests
openai
google-generativeai
anthropic
```

### 3) Add API keys (`.streamlit/secrets.toml`)

Create `.streamlit/secrets.toml` in the repo root:

```toml
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
CLAUDE_API_KEY = "..."
TAVILY_API_KEY = "tvly-..."
```

> Only the keys you actually use need to be set. If a key is missing, the respective model/tool will gracefully report an error.

### 4) Personas file (`personas.json`)

Place a `personas.json` (UTF-8) in the repo root. Example:

```json
[
  {
    "name": "Zara Chen",
    "age_group": "Gen Z",
    "occupation": "Computer Science Student",
    "location": "San Francisco, CA",
    "cultural_background": "Chinese-American",
    "education_level": "Bachelor's",
    "personality_traits": ["Creative", "Tech-savvy", "Idealistic", "Impatient"],
    "hobbies": ["Gaming", "TikTok creation", "Anime", "Hackathons"],
    "tech_savviness": "Expert",
    "goals_motivations": "Build innovative apps that make a social impact",
    "pain_points": "Student debt, finding authentic connections, climate anxiety",
    "speaking_style": "Casual with tech slang, direct",
    "affinity_keywords": ["innovation", "sustainability", "diversity", "disruption"],
    "llm_model": "Default"
  }
]
```

You can also manage personas entirely in the UI; changes are saved back to `personas.json`.

---

## ▶️ Run

```bash
streamlit run app.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`).

---

## 🧭 How it works

* **Per-turn flow**

  1. You ask a question.
  2. The app randomly shuffles the active personas.
  3. The first persona answers with no previous-turn context.
  4. Each subsequent persona receives the earlier replies from the **same turn** and can agree, disagree, or add nuance.
  5. All answers are stored (and exportable) in a transcript.

* **First-person voice**

  * Persona identity + style are injected as a **system-level instruction** to the selected LLM path, enforcing first-person responses even when web search is used.

* **No carryover context**

  * Each question starts fresh (no prior turns). You can change this in code if you ever want cross-turn continuity.

* **Web search (optional)**

  * When a model chooses to fetch facts, it calls Tavily under the hood. Results are summarized into the persona’s own words.

---

## 🧩 UI Primer

* **Chat tab:** ask a question → see ordered persona cards (speaking order preserved).
* **Persona Management:**

  * Toggle **Active** to include/exclude personas from the next question.
  * **Edit/Delete** personas (saved to `personas.json` automatically).
  * Tooltips on every field clarify expected values.
* **Transcript:** download `.txt`, or clear the in-memory history.
* **Sidebar Settings:**

  * Set default LLM (applies when a persona’s model is “Default”).
  * **Reload personas from file** (if you edited `personas.json` outside the app).

---

## 🧾 Persona schema (summary)

| Field                 | Type          | Example / Guidance                                              |
| --------------------- | ------------- | --------------------------------------------------------------- |
| `name`                | string        | Unique display name (e.g., “Zara Chen”).                        |
| `age_group`           | enum          | “Gen Z”, “Millennial”, “Gen X”, “Boomer”, “Silent”.             |
| `occupation`          | string        | Job/role shaping priorities and examples.                       |
| `location`            | string        | “City, Region/Country” (e.g., “Austin, TX”).                    |
| `cultural_background` | string        | Heritage/identity notes (e.g., “Cuban-American”).               |
| `education_level`     | enum          | “High School”, “Trade School”, “Bachelor's”, “Master's”, “PhD”. |
| `personality_traits`  | list\<string> | Adjectives (e.g., “Curious”, “Pragmatic”).                      |
| `hobbies`             | list\<string> | Interests (e.g., “Hiking”, “Podcasts”).                         |
| `tech_savviness`      | enum          | “Low”, “Moderate”, “High”, “Expert”.                            |
| `goals_motivations`   | string        | 1–2 sentences on what drives them.                              |
| `pain_points`         | string        | Constraints/frustrations (time, budget…).                       |
| `speaking_style`      | string        | Voice/tone (“Professional and warm; uses analogies”).           |
| `affinity_keywords`   | list\<string> | Values/topics they prioritize (“sustainability”, “ROI”).        |
| `llm_model`           | enum          | “Default”, “OpenAI”, “Gemini”, “Claude”.                        |

> In the form, list-type fields also accept **comma-separated strings**; they are converted to lists on save.

---

## 🔧 Configuration Notes

* **Model selection**

  * Persona-level `llm_model` overrides the global default in the sidebar.
* **Tooling**

  * Tavily is only used when the model requests it; if `TAVILY_API_KEY` is unset, web search is skipped with an error message captured in the response flow.
* **Constraints**

  * Responses aim for **120–180 words**, **no emojis**, and explicit assumptions if uncertain.

---

## 🚀 Deploy

### Streamlit Community Cloud

1. Push to GitHub.
2. Create a new Streamlit app from your repo.
3. Add the secrets in the app’s **Settings → Secrets** UI.
4. Add (or upload) `personas.json`.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🧰 Troubleshooting

* **Model errors like “library not installed”**
  Install the missing client:

  ```bash
  pip install openai anthropic google-generativeai
  ```

* **Web search returns an error**
  Ensure `TAVILY_API_KEY` is set in `.streamlit/secrets.toml`.

* **Personas not updating**

  * Edits in the UI autosave to `personas.json`.
  * If you edit the file externally, click **Reload personas from file** in the sidebar.

* **Empty label warning in Streamlit**
  The input field uses a hidden label (`label_visibility="collapsed"`) to avoid this warning; if you change it, ensure labels aren’t empty strings.

---

## 🤝 Contributing

1. Fork, create a feature branch, commit changes.
2. Add/adjust unit tests if applicable.
3. Submit a PR with a clear description.

---

## 📄 License

MIT (or your preferred license). Add a `LICENSE` file at the repo root.

---

## 🙌 Acknowledgements

Thanks to OpenAI, Google, Anthropic, and Tavily for the APIs used.
