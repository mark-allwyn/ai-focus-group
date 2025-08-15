
"""
Multi-Persona Focus Group Chatbot
A Streamlit app for simulating diverse perspectives on user questions
"""

import streamlit as st
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from datetime import datetime
import json
import re
from enum import Enum
import requests  # <-- ADDED for web tool calls
import random    # <-- ADDED for random speaking order
import os        # <-- ADDED for file persistence
from copy import deepcopy  # <-- ADDED for safe defaults

# ----------- File persistence config -----------
PERSONAS_FILE = "personas.json"  # can be .json or .txt containing JSON


# Persona structure
@dataclass
class Persona:
    """Structured representation of a focus group persona"""
    name: str
    age_group: str  # "Gen Z", "Millennial", "Gen X", "Boomer", "Silent"
    occupation: str
    location: str
    cultural_background: str
    education_level: str  # "High School", "Bachelor's", "Master's", "PhD", "Trade School"
    personality_traits: List[str] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    tech_savviness: str = "Moderate"  # "Low", "Moderate", "High", "Expert"
    goals_motivations: str = ""
    pain_points: str = ""
    speaking_style: str = "Conversational"
    affinity_keywords: List[str] = field(default_factory=list)
    llm_model: str = "Default"  # "Default", "OpenAI", "Gemini", "Claude"

@dataclass
class ConversationTurn:
    """Single turn in the conversation"""
    timestamp: str
    user_question: str
    answers_by_persona: Dict[str, str]

# Default personas
SEED_PERSONAS = [
    Persona(
        name="Zara Chen",
        age_group="Gen Z",
        occupation="Computer Science Student",
        location="San Francisco, CA",
        cultural_background="Chinese-American",
        education_level="Bachelor's",
        personality_traits=["Creative", "Tech-savvy", "Idealistic", "Impatient"],
        hobbies=["Gaming", "TikTok creation", "Anime", "Hackathons"],
        tech_savviness="Expert",
        goals_motivations="Build innovative apps that make a social impact",
        pain_points="Student debt, finding authentic connections, climate anxiety",
        speaking_style="Casual with tech slang, direct",
        affinity_keywords=["innovation", "sustainability", "diversity", "disruption"],
        llm_model="Default"
    ),
    Persona(
        name="Marcus Thompson",
        age_group="Millennial",
        occupation="Marketing Manager & Parent",
        location="Austin, TX",
        cultural_background="African-American",
        education_level="Master's",
        personality_traits=["Pragmatic", "Ambitious", "Family-oriented", "Time-conscious"],
        hobbies=["BBQ", "Youth coaching", "Podcasts", "Home automation"],
        tech_savviness="High",
        goals_motivations="Balance career growth with quality family time",
        pain_points="Work-life balance, childcare costs, keeping skills current",
        speaking_style="Professional but warm, uses analogies, efficiency-focused",
        affinity_keywords=["productivity", "growth", "family", "ROI", "scalability"],
        llm_model="Default"
    ),
    Persona(
        name="Elena Rodriguez",
        age_group="Gen X",
        occupation="Small Business Owner (Bakery)",
        location="Miami, FL",
        cultural_background="Cuban-American",
        education_level="Trade School",
        personality_traits=["Resilient", "Community-focused", "Practical", "Skeptical of trends"],
        hobbies=["Salsa dancing", "Recipe development", "Local politics", "Gardening"],
        tech_savviness="Moderate",
        goals_motivations="Keep business thriving while supporting local community",
        pain_points="Rising costs, competing with chains, regulatory complexity",
        speaking_style="Warm and personal, uses food metaphors, storytelling approach",
        affinity_keywords=["community", "tradition", "quality", "local", "authenticity"],
        llm_model="Default"
    ),
    Persona(
        name="Robert Walsh",
        age_group="Boomer",
        occupation="Recently Retired Engineer",
        location="Phoenix, AZ",
        cultural_background="Irish-American",
        education_level="Bachelor's",
        personality_traits=["Analytical", "Traditional", "Curious", "Value-conscious"],
        hobbies=["Golf", "Woodworking", "Historical documentaries", "RV travel"],
        tech_savviness="Low",
        goals_motivations="Enjoy retirement while staying mentally active and connected",
        pain_points="Technology frustrations, healthcare costs, staying relevant",
        speaking_style="Methodical, references past experiences, prefers clarity over brevity",
        affinity_keywords=["reliability", "value", "experience", "security", "simplicity"],
        llm_model="Default"
    ),
    Persona(
        name="Priya Patel",
        age_group="Millennial",
        occupation="NGO Program Director",
        location="Washington, DC",
        cultural_background="Indian-American",
        education_level="Master's",
        personality_traits=["Empathetic", "Mission-driven", "Collaborative", "Optimistic"],
        hobbies=["Yoga", "Documentary films", "Volunteering", "International cuisine"],
        tech_savviness="High",
        goals_motivations="Scale social impact while maintaining program quality",
        pain_points="Funding constraints, burnout, measuring impact effectively",
        speaking_style="Inclusive language, evidence-based, passionate but measured",
        affinity_keywords=["impact", "equity", "collaboration", "sustainability", "empowerment"],
        llm_model="Default"
    )
]

# ----------- Persistence Helpers -----------

def _persona_from_dict(d: Dict) -> Persona:
    """Create a Persona from a dict with safe defaults."""
    # accept string lists or comma-separated strings for list fields
    def _as_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return []

    return Persona(
        name=d.get("name", "Unnamed"),
        age_group=d.get("age_group", "Millennial"),
        occupation=d.get("occupation", "Unknown"),
        location=d.get("location", "Unknown"),
        cultural_background=d.get("cultural_background", ""),
        education_level=d.get("education_level", "Bachelor's"),
        personality_traits=_as_list(d.get("personality_traits", [])),
        hobbies=_as_list(d.get("hobbies", [])),
        tech_savviness=d.get("tech_savviness", "Moderate"),
        goals_motivations=d.get("goals_motivations", ""),
        pain_points=d.get("pain_points", ""),
        speaking_style=d.get("speaking_style", "Conversational"),
        affinity_keywords=_as_list(d.get("affinity_keywords", [])),
        llm_model=d.get("llm_model", "Default"),
    )

def load_personas_from_file(path: str = PERSONAS_FILE) -> List[Persona]:
    """Load personas from a JSON file; if missing/invalid, fall back to seed and write file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                personas = [_persona_from_dict(item) for item in data]
                if personas:
                    return personas
        except Exception:
            pass  # fall back to seeds and overwrite

    # fallback: write seeds to file and return them
    save_personas_to_file(SEED_PERSONAS, path)
    return deepcopy(SEED_PERSONAS)

def save_personas_to_file(personas: List[Persona], path: str = PERSONAS_FILE) -> None:
    """Save personas to a JSON file (pretty-printed, UTF-8)."""
    try:
        serializable = [asdict(p) for p in personas]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save personas to {path}: {e}")


def init_session_state():
    """Initialize session state variables"""
    if 'personas' not in st.session_state:
        st.session_state.personas = load_personas_from_file()
    if 'selected_persona_names' not in st.session_state:
        # pick first 3 from the loaded file
        st.session_state.selected_persona_names = {p.name for p in st.session_state.personas[:3]}
    if 'turns' not in st.session_state:
        st.session_state.turns = []
    if 'editing_persona' not in st.session_state:
        st.session_state.editing_persona = None
    if 'default_model' not in st.session_state:
        st.session_state.default_model = "OpenAI"

    # Initialize per-persona checkbox state exactly once; Streamlit will manage it after that
    if 'persona_keys_initialized' not in st.session_state:
        for p in st.session_state.personas:
            st.session_state.setdefault(f"active_{p.name}", p.name in st.session_state.selected_persona_names)
        st.session_state.persona_keys_initialized = True

def sync_active_personas_from_keys():
    """Make selected_persona_names match the current checkbox widget states."""
    st.session_state.selected_persona_names = {
        p.name for p in st.session_state.personas
        if st.session_state.get(f"active_{p.name}", False)
    }

def generate_persona_prompt(persona: Persona, question: str, context: str = "") -> str:
    """Generate a persona-specific prompt for the LLM"""
    system_prompt = f"""You are roleplaying as {persona.name}, age group {persona.age_group}, occupation {persona.occupation}, from {persona.location}.
Cultural background: {persona.cultural_background}
Education: {persona.education_level}
Personality traits: {', '.join(persona.personality_traits)}
Speaking style: {persona.speaking_style}
Hobbies/interests: {', '.join(persona.hobbies)}
Goals/motivations: {persona.goals_motivations}
Pain points: {persona.pain_points}
Affinity keywords: {', '.join(persona.affinity_keywords)}
Tech savviness: {persona.tech_savviness}

CRITICAL INSTRUCTIONS:
1. When answering, stay in character and reflect the above context
2. Be concise (120-180 words)
3. NEVER use emojis or emoticons in your response
4. Do not use any Unicode emoji characters
5. Express emotions through words, not symbols
6. If uncertain, state assumptions explicitly from {persona.name}'s perspective"""

    user_prompt = f"""Group question: {question}
{f'Helpful prior context: {context}' if context else ''}
Task: Provide {persona.name}'s perspective, in their voice, addressing key tradeoffs and practical advice. Remember: absolutely no emojis or emoticons.
If prior speakers are mentioned in the helpful context, briefly acknowledge or react to them (agree/disagree, add nuance), then add your own view."""

    return system_prompt + "\n\n" + user_prompt

# --- Helper for tool execution (used by OpenAI path) ---
def _perform_web_search(query: str, max_results: int = 5) -> Dict:
    """
    Calls Tavily Search API for recent web results.
    Returns a dict with 'results': [ {title, url, content, published_date?}, ... ]
    """
    api_key = st.secrets.get("TAVILY_API_KEY", "")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY not set in st.secrets",
            "results": []
        }
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max(1, min(int(max_results or 5), 8)),
                "include_answer": False,
                "include_images": False
            },
            timeout=20
        )
        data = resp.json()
        out = []
        for item in data.get("results", []):
            out.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": (item.get("content") or "")[:500],
                "published_date": item.get("published_date")
            })
        return {"results": out}
    except Exception as e:
        return {"error": f"web_search failed: {e}", "results": []}

# --- helper to keep first-person persona voice across models/tools ---
def _persona_system(prompt: str) -> str:
    # `prompt` is the combined persona identity + user question from generate_persona_prompt(...)
    return (
        f"{prompt}\n\n"
        "CRITICAL: Always respond in the FIRST PERSON as this persona, maintaining their voice, "
        "background, values, and speaking style. If you use the web_search tool or any external sources, "
        "weave facts naturally into the persona’s own perspective. Keep responses concise (120–180 words) "
        "and never use emojis."
    )

def call_openai(prompt: str) -> str:
    """Call OpenAI API with tool-use (web_search) enabled"""
    try:
        from openai import OpenAI

        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return "ERROR: OpenAI API key not found. Please add OPENAI_API_KEY to .streamlit/secrets.toml"

        client = OpenAI(api_key=api_key)

        # Tool schema the model can call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for fresh/recent factual information and return a concise set of sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 8}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Persona-as-system so it persists through tool calls
        sys = _persona_system(prompt)
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": "Respond now in first person as the persona."}
        ]

        # Up to 3 tool iterations
        for _ in range(3):
            resp = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",  # keep your original model
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=200,
                temperature=0.8
            )
            msg = resp.choices[0].message

            # If the model decided to call a tool
            if getattr(msg, "tool_calls", None):
                # Append the assistant message that requested the tool(s)
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {"query": tc.function.arguments or ""}

                        result_obj = _perform_web_search(
                            query=args.get("query", ""),
                            max_results=args.get("max_results", 5)
                        )
                        # Return the tool output for this tool call
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result_obj)
                        })
                # Continue loop so the model can incorporate tool results
                continue

            # No tool call -> final answer
            return msg.content or ""

        # Safety fallback if loop exits unexpectedly
        return "Sorry, I couldn't complete the tool-assisted reasoning."

    except ImportError:
        return "ERROR: OpenAI library not installed. Run: pip install openai"
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

def call_gemini(prompt: str) -> str:
    """Call Google Gemini API"""
    try:
        import google.generativeai as genai
        
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return "ERROR: Gemini API key not found. Please add GEMINI_API_KEY to .streamlit/secrets.toml"
        
        genai.configure(api_key=api_key)

        # Persona-as-system so first-person persists
        sys = _persona_system(prompt)
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=sys
        )
        
        response = model.generate_content("Respond now in first person as the persona.")
        return response.text
    except ImportError:
        return "ERROR: Google Generative AI library not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def call_claude(prompt: str) -> str:
    """Call Anthropic Claude API"""
    try:
        import anthropic
        
        api_key = st.secrets.get("CLAUDE_API_KEY", "")
        if not api_key:
            return "ERROR: Claude API key not found. Please add CLAUDE_API_KEY to .streamlit/secrets.toml"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Persona-as-system so first-person persists
        sys = _persona_system(prompt)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0.8,
            system=sys,
            messages=[
                {"role": "user", "content": "Respond now in first person as the persona."}
            ]
        )
        return response.content[0].text
    except ImportError:
        return "ERROR: Anthropic library not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Claude API Error: {str(e)}"

def call_llm(prompt: str, model: str = "OpenAI") -> str:
    """Route to appropriate LLM based on model selection"""
    # If model is "Default", use the global default setting
    if model == "Default":
        model = st.session_state.default_model
    
    if model == "OpenAI":
        return call_openai(prompt)
    elif model == "Gemini":
        return call_gemini(prompt)
    elif model == "Claude":
        return call_claude(prompt)
    else:
        return f"Unknown model: {model}"

# ----------- NEW: generic (non-persona) LLM calls for summaries -----------

def call_openai_plain(prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """Plain OpenAI call without persona system or tools (for summaries)."""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return "ERROR: OpenAI API key not found. Please add OPENAI_API_KEY to .streamlit/secrets.toml"
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a concise, neutral meeting summarizer. No emojis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content or ""
    except ImportError:
        return "ERROR: OpenAI library not installed. Run: pip install openai"
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

def call_gemini_plain(prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """Plain Gemini call without persona system (for summaries)."""
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return "ERROR: Gemini API key not found. Please add GEMINI_API_KEY to .streamlit/secrets.toml"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction="You are a concise, neutral meeting summarizer. No emojis."
        )
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return getattr(resp, "text", "") or ""
    except ImportError:
        return "ERROR: Google Generative AI library not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def call_claude_plain(prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """Plain Claude call without persona system (for summaries)."""
    try:
        import anthropic
        api_key = st.secrets.get("CLAUDE_API_KEY", "")
        if not api_key:
            return "ERROR: Claude API key not found. Please add CLAUDE_API_KEY to .streamlit/secrets.toml"
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a concise, neutral meeting summarizer. No emojis.",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text if resp.content else ""
    except ImportError:
        return "ERROR: Anthropic library not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Claude API Error: {str(e)}"

def call_llm_plain(prompt: str) -> str:
    """Route plain (non-persona) summarization to the default model."""
    model = st.session_state.get("default_model", "OpenAI")
    if model == "OpenAI":
        return call_openai_plain(prompt)
    elif model == "Gemini":
        return call_gemini_plain(prompt)
    elif model == "Claude":
        return call_claude_plain(prompt)
    else:
        return "Unknown default model for summarization."

def summarize_conversation(full_text: str) -> str:
    """Summarize the entire transcript text into an overview, highlights, and actions."""
    prompt = (
        "Summarize the following multi-persona conversation into:\n"
        "1) A single-paragraph overview (<=120 words)\n"
        "2) 5–8 concise bullet highlights (agreements, disagreements, notable nuances)\n"
        "3) Action items or follow-up questions (if any)\n\n"
        "Keep it neutral and specific. No emojis.\n\n"
        f"=== Conversation Transcript Start ===\n{full_text}\n=== Conversation Transcript End ==="
    )
    return call_llm_plain(prompt)

# ----------- Persona answer generation -----------

def generate_persona_answer(persona: Persona, question: str, context: str = "") -> str:
    """Generate an answer from a specific persona's perspective"""
    prompt = generate_persona_prompt(persona, question, context)
    response = call_llm(prompt, persona.llm_model)
    # Clean any emojis that might slip through
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        "]+", flags=re.UNICODE)
    cleaned_response = emoji_pattern.sub('', response)
    return cleaned_response

def get_conversation_context(max_turns: int = 3) -> str:
    """Get recent conversation context for continuity"""
    if not st.session_state.turns:
        return ""
    recent_turns = st.session_state.turns[-max_turns:]
    context_parts = []
    for i, turn in enumerate(recent_turns, 1):
        context_parts.append(f"Q{i}: {turn.user_question[:50]}...")
        if turn.answers_by_persona:
            sample_answer = list(turn.answers_by_persona.values())[0][:50]
            context_parts.append(f"Group discussed: {sample_answer}...")
    return " | ".join(context_parts) if context_parts else ""

def export_transcript() -> str:
    """Export conversation transcript to text format"""
    if not st.session_state.turns:
        return "No conversation to export yet."
    transcript = []
    transcript.append("FOCUS GROUP TRANSCRIPT")
    transcript.append("=" * 50)
    transcript.append(f"Generated: {datetime.now().isoformat()}")
    transcript.append(f"Participants: {', '.join(sorted(st.session_state.selected_persona_names))}")
    transcript.append("=" * 50)
    transcript.append("")
    for i, turn in enumerate(st.session_state.turns, 1):
        transcript.append(f"=== Turn {i} — {turn.timestamp} ===")
        transcript.append(f"User: {turn.user_question}")
        transcript.append("")
        for name, answer in turn.answers_by_persona.items():
            transcript.append(f"{name}: {answer}")
            transcript.append("")
        transcript.append("-" * 30)
        transcript.append("")
    return "\n".join(transcript)

def render_persona_card(persona: Persona, answer: str):
    """Render a single persona's answer as a card"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"### {persona.name}")
            st.caption(f"{persona.age_group} • {persona.occupation}")
            st.caption(f"Location: {persona.location}")
            model_display = persona.llm_model if persona.llm_model != "Default" else st.session_state.default_model
            st.caption(f"Model: {model_display}")
        with col2:
            traits_html = " ".join([f"`{trait}`" for trait in persona.personality_traits[:3]])
            st.markdown(traits_html)
            st.markdown(f"**{persona.name}:** {answer}")
        st.divider()

# ----------- NEW: Re-ask same question with self memory -----------

def _trim(txt: str, limit: int = 900) -> str:
    if not txt:
        return ""
    return txt if len(txt) <= limit else txt[:limit] + "…"

def rerun_last_turn_with_self_memory():
    """Re-run the most recent question; each persona sees their own prior answer and can update it.
    Preserves current behavior: random speaking order and in-turn awareness."""
    if not st.session_state.turns:
        st.warning("No previous turn to re-run.")
        return

    past_turn = st.session_state.turns[-1]
    question = past_turn.user_question

    # Use the personas who answered that past turn and still exist now
    prior_names = list(past_turn.answers_by_persona.keys())
    personas = [p for p in st.session_state.personas if p.name in prior_names]

    if not personas:
        st.warning("No matching personas from the last turn are available to re-run.")
        return

    ordered = personas[:]
    random.shuffle(ordered)

    answers: Dict[str, str] = {}
    discussion_lines: List[str] = []

    for persona in ordered:
        self_prior = past_turn.answers_by_persona.get(persona.name)
        if self_prior:
            self_block = (
                "Your previous answer to this same question was:\n"
                f"\"{_trim(self_prior)}\"\n\n"
                "Guidance: In first person, reflect on your prior stance. "
                "Do not repeat verbatim. State what you keep, what you change, and why."
            )
        else:
            self_block = (
                "You did not answer this question previously. "
                "Provide a concise first-person perspective now."
            )

        if discussion_lines:
            in_turn_block = (
                "Prior speakers in this re-run:\n"
                + "\n".join(discussion_lines)
                + "\nBriefly acknowledge or react where relevant before adding your view."
            )
        else:
            in_turn_block = "You are the first to speak in this re-run. Be clear and concise."

        turn_context = f"{self_block}\n\n{in_turn_block}"

        answer = generate_persona_answer(persona, question, turn_context)
        answers[persona.name] = answer
        discussion_lines.append(f"- {persona.name}: {answer}")

    st.session_state.turns.append(ConversationTurn(
        timestamp=datetime.now().isoformat(),
        user_question=f"{question} (self-memory re-run)",
        answers_by_persona=answers
    ))
    st.success("Re-asked the same question with self memory.")

# ----------- UI: Chat / Personas / Transcript -----------

def render_chat_tab():
    """Render the main chat interface"""

    sync_active_personas_from_keys()
    
    if not st.session_state.selected_persona_names:
        st.warning("Please select at least one persona from the Persona Management tab")
        st.info("Go to the 'Persona Management' tab and check the boxes next to the personas you want to activate")
        return
    
    # Show active personas
    st.success(f"Active Personas: {', '.join(sorted(st.session_state.selected_persona_names))}")
    
    # Question input
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="What do you think about remote work?",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        ask_button = st.button("Ask Personas", type="primary", use_container_width=True)
        reask_button = st.button(
            "Re-ask same question",
            help="Ask the previous question again; each persona sees their own prior answer from that turn."
        )
    
    # Process new question
    if ask_button and question:
        with st.spinner(f"Getting responses from {len(st.session_state.selected_persona_names)} personas..."):
            base_context = ""  # no cross-turn carryover
            answers: Dict[str, str] = {}

            # Determine speaking order randomly each turn
            selected_personas = [p for p in st.session_state.personas if p.name in st.session_state.selected_persona_names]
            random.shuffle(selected_personas)

            # Build sequential discussion so later personas can see earlier answers
            discussion_lines: List[str] = []

            for idx, persona in enumerate(selected_personas, start=1):
                if discussion_lines:
                    turn_context = (
                        "Prior speakers this turn:\n"
                        + "\n".join(discussion_lines)
                        + "\nGuidance: Briefly acknowledge or react to prior speakers where relevant "
                          "(agree/disagree, add nuance) before sharing your own view."
                    )
                else:
                    turn_context = (
                        "You are the first to speak this turn. "
                        "State your perspective clearly and concisely."
                    )

                if base_context:
                    turn_context = f"{base_context} || {turn_context}"

                answer = generate_persona_answer(persona, question, turn_context)
                answers[persona.name] = answer  # dict preserves insertion order
                discussion_lines.append(f"- {persona.name}: {answer}")

            # Save turn
            turn = ConversationTurn(
                timestamp=datetime.now().isoformat(),
                user_question=question,
                answers_by_persona=answers
            )
            st.session_state.turns.append(turn)
        
        st.success("Responses collected!")

    # Re-ask same question with self memory (re-run last turn)
    if reask_button:
        if not st.session_state.turns:
            st.warning("No previous turn to re-ask yet.")
        else:
            with st.spinner("Re-asking the same question with self memory..."):
                rerun_last_turn_with_self_memory()
    
    # Display latest responses (in the order they were spoken)
    if st.session_state.turns:
        st.header("Latest Responses")
        latest_turn = st.session_state.turns[-1]
        
        st.info(f"**Question:** {latest_turn.user_question}")
        
        for name, answer in latest_turn.answers_by_persona.items():
            persona = next((p for p in st.session_state.personas if p.name == name), None)
            if persona:
                render_persona_card(persona, answer)
    else:
        st.info("Ask a question to start the conversation!")

def render_persona_form(persona: Optional[Persona] = None, form_key: str = "persona_form"):
    """Render a form for creating or editing a persona"""
    is_edit = persona is not None
    
    with st.form(form_key):
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Name*",
                value=persona.name if is_edit else "",
                help="Unique display name for this persona (used in responses and cards)."
            )
            age_group = st.selectbox(
                "Age Group*",
                ["Gen Z", "Millennial", "Gen X", "Boomer", "Silent"],
                index=["Gen Z", "Millennial", "Gen X", "Boomer", "Silent"].index(persona.age_group) if is_edit else 0,
                help="Demographic cohort. This can affect tone and examples."
            )
            occupation = st.text_input(
                "Occupation*",
                value=persona.occupation if is_edit else "",
                help="Primary job/role. Guides perspective, priorities, and examples."
            )
            location = st.text_input(
                "Location*",
                value=persona.location if is_edit else "",
                help="City and region (e.g., 'Austin, TX' or 'London, UK')."
            )
            cultural_bg = st.text_input(
                "Cultural Background*",
                value=persona.cultural_background if is_edit else "",
                help="Heritage/identity descriptors that shape worldview (e.g., 'Cuban-American')."
            )
        
        with col2:
            education = st.selectbox(
                "Education Level",
                ["High School", "Trade School", "Bachelor's", "Master's", "PhD"],
                index=["High School", "Trade School", "Bachelor's", "Master's", "PhD"].index(persona.education_level) if is_edit else 2,
                help="Highest completed education credential."
            )
            tech_level = st.selectbox(
                "Tech Savviness",
                ["Low", "Moderate", "High", "Expert"],
                index=["Low", "Moderate", "High", "Expert"].index(persona.techsavviness) if is_edit and hasattr(persona, "techsavviness") else ["Low", "Moderate", "High", "Expert"].index(persona.tech_savviness) if is_edit else 1,
                help="How comfortable this persona is with technology and digital tools."
            )
            llm_model = st.selectbox(
                "LLM Model",
                ["Default", "OpenAI", "Gemini", "Claude"],
                index=["Default", "OpenAI", "Gemini", "Claude"].index(persona.llm_model) if is_edit and hasattr(persona, 'llm_model') else 0,
                help="Which backend model to use for this persona. 'Default' uses the sidebar selection."
            )
        
        st.subheader("Personality & Style")
        col3, col4 = st.columns(2)
        
        with col3:
            personality = st.text_input(
                "Personality Traits (comma-separated)",
                value=", ".join(persona.personality_traits) if is_edit else "",
                help="Comma-separated adjectives describing temperament (e.g., 'Curious, Pragmatic, Empathetic')."
            )
            hobbies = st.text_input(
                "Hobbies (comma-separated)",
                value=", ".join(persona.hobbies) if is_edit else "",
                help="Comma-separated interests (e.g., 'Hiking, Podcasts, Cooking')."
            )
            style = st.text_input(
                "Speaking Style",
                value=persona.speaking_style if is_edit else "Conversational",
                help="Short description of voice/tone (e.g., 'Professional and warm; uses analogies')."
            )
        
        with col4:
            keywords = st.text_input(
                "Affinity Keywords (comma-separated)",
                value=", ".join(persona.affinity_keywords) if is_edit else "",
                help="Comma-separated topics/values they prioritize (e.g., 'sustainability, ROI, community')."
            )
            goals = st.text_area(
                "Goals/Motivations",
                value=persona.goals_motivations if is_edit else "",
                help="What drives them. 1–2 sentences on aims/ambitions."
            )
            pain_points = st.text_area(
                "Pain Points",
                value=persona.pain_points if is_edit else "",
                help="Key frustrations or constraints they face (e.g., budget, time, skills)."
            )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button(
                "Update Persona" if is_edit else "Add Persona",
                type="primary",
                help="Save your changes to this persona."
            )
        with col2:
            if is_edit:
                cancel_button = st.form_submit_button(
                    "Cancel Edit",
                    help="Discard unsaved changes and return to the list."
                )
            else:
                cancel_button = False
        
        if submit_button:
            if name and occupation and location and cultural_bg:
                new_persona = Persona(
                    name=name,
                    age_group=age_group,
                    occupation=occupation,
                    location=location,
                    cultural_background=cultural_bg,
                    education_level=education,
                    tech_savviness=tech_level if isinstance(tech_level, str) else persona.tech_savviness,
                    personality_traits=[t.strip() for t in personality.split(",") if t.strip()],
                    hobbies=[h.strip() for h in hobbies.split(",") if h.strip()],
                    goals_motivations=goals,
                    pain_points=pain_points,
                    speaking_style=style,
                    affinity_keywords=[k.strip() for k in keywords.split(",") if k.strip()],
                    llm_model=llm_model
                )
                
                if is_edit:
                    old_name = persona.name
                    # Update existing persona
                    for i, p in enumerate(st.session_state.personas):
                        if p.name == persona.name:
                            st.session_state.personas[i] = new_persona
                            # Migrate checkbox state key if name changed
                            old_key = f"active_{old_name}"
                            new_key = f"active_{new_persona.name}"
                            if old_key in st.session_state and new_key not in st.session_state:
                                st.session_state[new_key] = st.session_state.pop(old_key)
                            # Update selected_persona_names if name changed
                            if persona.name != new_persona.name and persona.name in st.session_state.selected_persona_names:
                                st.session_state.selected_persona_names.remove(persona.name)
                                st.session_state.selected_persona_names.add(new_persona.name)
                            break
                    # Persist changes
                    save_personas_to_file(st.session_state.personas)
                    st.session_state.editing_persona = None
                    st.success(f"Updated {new_persona.name}!")
                else:
                    # Add new persona
                    st.session_state.personas.append(new_persona)
                    # Initialize its checkbox key once; let user decide if active
                    st.session_state.setdefault(f"active_{new_persona.name}", False)
                    # Persist changes
                    save_personas_to_file(st.session_state.personas)
                    st.success(f"Added {name}!")
                
                st.rerun()
            else:
                st.error("Please fill all required fields (*)")
        
        if cancel_button:
            st.session_state.editing_persona = None
            st.rerun()

def render_personas_tab():
    """Render the persona management interface"""
    st.header("Persona Management")
    
    # Active personas selector - MORE PROMINENT
    st.subheader("ACTIVATE/DEACTIVATE PERSONAS")
    st.info("Check the boxes below to make personas active. Active personas will respond to your questions in the Chat tab.")
    
    # Create a more visual active/inactive selection
    for persona in st.session_state.personas:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            # IMPORTANT: do NOT pass a dynamic value=...; rely solely on key-managed state
            is_active = st.checkbox(
                "Active",
                key=f"active_{persona.name}",
                on_change=sync_active_personas_from_keys,  # <-- update only when toggled
                help="Toggle to include/exclude this persona from answering the next question."
            )
        with col2:
            st.markdown(f"**{persona.name}** ({persona.age_group}) - {persona.occupation}")
            model_display = persona.llm_model if hasattr(persona, 'llm_model') else "Default"
            if model_display == "Default":
                model_display = f"Default ({st.session_state.default_model})"
            st.caption(f"Location: {persona.location} | Model: {model_display}")
        
        with col3:
            col_edit, col_delete = st.columns(2)
            with col_edit:
                if st.button("Edit", key=f"edit_btn_{persona.name}", help="Edit this persona's details."):
                    st.session_state.editing_persona = persona.name
                    st.rerun()
            with col_delete:
                if st.button("Delete", key=f"delete_btn_{persona.name}", help="Remove this persona permanently."):
                    # Remove persona
                    st.session_state.personas = [p for p in st.session_state.personas if p.name != persona.name]
                    # Drop from active set and cleanup widget key
                    st.session_state.selected_persona_names.discard(persona.name)
                    st.session_state.pop(f"active_{persona.name}", None)
                    # Persist changes
                    save_personas_to_file(st.session_state.personas)
                    st.success(f"Deleted {persona.name}")
                    st.rerun()
    
    # Show count of active personas
    st.success(f"Currently {len(st.session_state.selected_persona_names)} personas are active")
    
    st.divider()
    
    # Add new persona section
    if st.session_state.editing_persona is None:
        st.subheader("Add New Persona")
        render_persona_form(form_key="new_persona_form")
    else:
        # Show edit form
        persona_to_edit = next((p for p in st.session_state.personas if p.name == st.session_state.editing_persona), None)
        if persona_to_edit:
            st.subheader(f"Editing: {persona_to_edit.name}")
            render_persona_form(persona_to_edit, form_key=f"edit_{persona_to_edit.name}")

def render_transcript_tab():
    """Render the transcript view and export"""
    st.header("Conversation Transcript")
    
    if not st.session_state.turns:
        st.info("No conversation yet. Start asking questions in the Chat tab!")
        return
    
    # Export / Clear / Summarize buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_text = export_transcript()
        st.download_button(
            label="Download Transcript (.txt)",
            data=transcript_text,
            file_name=f"focus_group_transcript_{timestamp}.txt",
            mime="text/plain",
            type="primary",
            help="Download the full conversation history as a text file."
        )
    
    with col2:
        if st.button("Clear Transcript", help="Remove all previous turns from this session."):
            st.session_state.turns = []
            st.session_state.pop("conversation_summary", None)
            st.success("Transcript cleared!")
            st.rerun()

    with col3:
        if st.button("Summarize Conversation", help="Generate a concise summary of the entire transcript."):
            with st.spinner("Summarizing the conversation..."):
                full_text = export_transcript()
                summary = summarize_conversation(full_text)
                st.session_state.conversation_summary = summary or "No summary available."
            st.success("Summary generated below.")
    
    # Show conversation summary if available
    if st.session_state.get("conversation_summary"):
        st.subheader("Conversation Summary")
        st.markdown(st.session_state.conversation_summary)
        st.divider()
    
    # Display transcript
    for i, turn in enumerate(st.session_state.turns, 1):
        with st.expander(f"Turn {i} - {turn.user_question[:50]}...", expanded=(i == len(st.session_state.turns))):
            st.markdown(f"**Timestamp:** {turn.timestamp}")
            st.markdown(f"**Question:** {turn.user_question}")
            st.divider()
            for name, answer in turn.answers_by_persona.items():
                st.markdown(f"**{name}:**")
                st.markdown(answer)
                st.markdown("")

def check_api_keys():
    """Check which API keys are configured"""
    keys_status = {}
    keys_status['OpenAI'] = bool(st.secrets.get("OPENAI_API_KEY", ""))
    keys_status['Gemini'] = bool(st.secrets.get("GEMINI_API_KEY", ""))
    keys_status['Claude'] = bool(st.secrets.get("CLAUDE_API_KEY", ""))
    return keys_status

def main():
    st.set_page_config(
        page_title="Focus Group Chatbot",
        page_icon="",
        layout="wide"
    )
    
    init_session_state()

    # Title and description
    st.title("Focus Group Chatbot")
    st.markdown("Get diverse perspectives on any question from multiple personas")
    
    # Settings in sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Default LLM Model
        st.subheader("Default LLM Model")
        default_model = st.selectbox(
            "Select default model for all personas",
            ["OpenAI", "Gemini", "Claude"],
            index=["OpenAI", "Gemini", "Claude"].index(st.session_state.default_model),
            help="Sets the default backend used when a persona's model is 'Default'."
        )
        st.session_state.default_model = default_model

        st.caption(f"Personas file: `{os.path.abspath(PERSONAS_FILE)}`")
        if st.button("Reload personas from file", help="Re-read personas from disk (useful if you edited the JSON externally)."):
            st.session_state.personas = load_personas_from_file()
            # re-init checkbox keys for any new/renamed personas
            for p in st.session_state.personas:
                st.session_state.setdefault(f"active_{p.name}", p.name in st.session_state.selected_persona_names)
            st.success("Personas reloaded from file.")
            st.rerun()
        
        # Quick stats
        st.metric("Total Personas", len(st.session_state.personas))
        st.metric("Active Personas", len(st.session_state.selected_persona_names))
        st.metric("Conversation Turns", len(st.session_state.turns))
    
    # Main tabbed interface
    tab1, tab2, tab3 = st.tabs(["Chat", "Persona Management", "Transcript"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_personas_tab()
    
    with tab3:
        render_transcript_tab()

if __name__ == "__main__":
    main()

