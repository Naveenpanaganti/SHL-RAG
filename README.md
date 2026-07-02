# SHL Assessment Recommender — RAG Agent

Conversational agent that recommends SHL Individual Test Solutions via a stateless FastAPI service.

## Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI app, lifespan startup
│   ├── chat.py          # Core orchestration: retrieve → prompt → parse
│   ├── retriever.py     # FAISS semantic search
│   ├── vectorstore.py   # Index build + in-memory storage
│   ├── llm.py           # Gemini / Groq client
│   ├── prompts.py       # System + user prompt templates
│   ├── models.py        # Pydantic request/response schema
│   ├── utils.py         # JSON extraction, turn counting
│   ├── scraper.py       # One-time catalog scraper (optional)
│   └── routers/
│       ├── health.py    # GET /health
│       └── chat.py      # POST /chat
├── data/
│   └── catalog.json     # SHL catalog (paste provided JSON here)
├── tests/
│   ├── test_health.py
│   └── test_chat.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY or GROQ_API_KEY
```

### 3. Add catalog data
Paste the SHL-provided `catalog.json` into the `data/` folder.
The file must be a JSON array where each item has at minimum:
```json
{
  "name": "Assessment Name",
  "url": "https://www.shl.com/products/product-catalog/view/slug/",
  "test_type": "K",
  "description": "...",
  "keys": [],
  "job_levels": [],
  "languages": [],
  "duration": "10 minutes"
}
```

### 4. Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test endpoints
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"}
    ]
  }'
```

## Running Tests
```bash
pytest tests/ -v
```

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "Agent response text",
  "recommendations": [
    {
      "name": "Assessment Name",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `null` when clarifying or refusing; an array of 1-10 items when committed.
- `end_of_conversation` is `true` only when the user confirms the conversation is complete.

## Deployment (Render)
1. Push to GitHub
2. Create a new Web Service on Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env.example`
