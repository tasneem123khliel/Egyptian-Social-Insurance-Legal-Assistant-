# ⚖️ Egyptian Social Insurance Legal Assistant (RAG)

An Arabic (RTL) AI legal assistant that answers questions about the **Egyptian Social Insurance Law No. 79 of 1975** using Retrieval-Augmented Generation (RAG). It retrieves relevant law passages from a local Chroma vector database and generates a single, documented answer citing article numbers — no file upload needed, the knowledge base is prebuilt.

> ⚖️ **Disclaimer:** This tool provides general legal *information* from the provided document, not legal *advice*. Consult a qualified lawyer for legal decisions.

---

## ✨ Features

- 💬 **Arabic RTL chat interface** styled like a modern legal portal (dark-green theme, Cairo font)
- 👤 **User accounts** — register / log in; each user keeps a persistent, private conversation history
- 🕘 **Conversation history panel** — reload or delete past conversations; every question is auto-saved
- ⚖️ **Single grounded answer per question** with automatic article-number badges (e.g. المادة 2, المادة 3)
- 👍👎 **Feedback buttons** (helpful / inaccurate) + 📋 copy-answer button
- 💡 **Suggested questions**, 📄 downloadable law PDF card, ⏱️ answer-time display
- 🔑 **API key hidden from the UI** — configured in code / secrets file only
- 🚀 **Fast inference** — single retrieval per question + Groq `openai/gpt-oss-20b` with low reasoning effort
- 🔌 Optional **REST API** (`api.py`) for Postman / integrations

## 🖼️ Screenshots

| Login | Main interface | History & references |
|---|---|---|
| ![Login](image/01-login.png) | ![Main interface](image/02-main-interface.png) | ![History](image/03-question-history.png) |

## 🧠 How it works

```
User question (Arabic)
        │
        ▼
Arabic normalization (tashkeel/tatweel/alef unification for better retrieval)
        │
        ▼
Chroma vector search (prebuilt chroma_db, FastEmbed intfloat/multilingual-e5-large)
        │  → single retrieval, top-5 passages
        ▼
Strict Arabic legal prompt → Groq LLM (openai/gpt-oss-20b, reasoning_effort=low)
        │
        ▼
One final answer + article-number badges + per-user saved history
```

The system prompt enforces legal fidelity: no added facts, preserves wording (يسري / لا يسري / يستثنى / يجوز …), never inverts categories and exceptions, cites article numbers, and replies *"لا أعرف بناءً على المستند المقدم."* when the context is insufficient.

## 📁 Project structure

```
egyptian-social-insurance-rag-assistant/
├── app.py                  # Streamlit UI (accounts, chat, history panel)
├── rag_core.py             # RAG pipeline: PDF → embeddings → Chroma → Groq LLM
├── api.py                  # Optional FastAPI endpoint (POST /ask, GET /health)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml         # green theme
│   └── secrets.toml.example# template — copy to secrets.toml (never commit the real one)
├── data/
│   └── law-79-1975.pdf     # the source law document
├── chroma_db/              # prebuilt vector store (215 chunks) — reuse as-is
├── image/                  # screenshots for this README (add your own)
├── history/                # created at runtime — per-user conversations (not committed)
└── users.json              # created at runtime — hashed credentials (not committed)
```

## 🚀 Setup & run

**Requirements:** Python 3.10+ (tested on 3.13), a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd egyptian-social-insurance-rag-assistant

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (pick ONE option below 👇)

# 5. Run the app
streamlit run app.py
```

> First launch downloads the embedding model once (~1 GB, one time only), then every question answers in seconds. The `chroma_db/` folder is prebuilt, so no re-indexing is needed.

### 🔑 Where to put your Groq API key

**Option A — inside the code (simplest for local use):** open `app.py` and set the constant at the top:

```python
GROQ_API_KEY = "gsk_your_key_here"
```

**Option B — secrets file (recommended, never uploaded):**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

then put your key in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

**Option C — environment variable:**

```bash
# Windows (PowerShell):
setx GROQ_API_KEY "gsk_your_key_here"
# macOS/Linux:
export GROQ_API_KEY="gsk_your_key_here"
```

Priority order: code constant → `secrets.toml` → environment variable.
⚠️ **Before pushing to GitHub:** empty the key in `app.py` (`GROQ_API_KEY = ""`), never commit `secrets.toml`, `users.json`, or `history/`.

### 🔌 REST API (optional)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Swagger docs: http://localhost:8000/docs
- `GET /health` → service + knowledge-base status
- `POST /ask` with `{"question": "...", "top_k": 5}` → `{"answer": "...", "sources": [...]}`

## 🛠️ Configuration

| Setting | Where | Default |
|---|---|---|
| LLM model | `rag_core.py` → `LLM_MODEL` env or constant | `openai/gpt-oss-20b` (Groq) |
| Embedding model | `rag_core.py` → `EMBEDDING_MODEL` env or constant | `intfloat/multilingual-e5-large` ⚠️ must match the model used to build `chroma_db/` |
| Retrieved passages | `app.py` → `FIXED_TOP_K` | `5` |
| Chroma folder | `CHROMA_DIR` env or `./chroma_db` | `./chroma_db` |

## ☁️ Deploy notes (Streamlit Cloud)

1. Push this repo to GitHub (secrets excluded by `.gitignore`).
2. Deploy on [Streamlit Cloud](https://streamlit.io/cloud) pointing to `app.py`.
3. In the app settings add your secret: `GROQ_API_KEY = "gsk_..."`.
4. Keep `chroma_db/` and `data/law-79-1975.pdf` in the repo so the knowledge base loads instantly.

## 📄 License

MIT — free for educational and research use.
