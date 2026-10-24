# Town of Cedarhaven — Municipal AI Knowledge Assistant

A source-grounded Retrieval-Augmented Generation (RAG) web chatbot built for the fictional **Town of Cedarhaven**. The assistant answers resident questions strictly using the three supplied municipal policy PDFs, provides exact document/page/section citations, and includes an interactive evidence panel for live verification.

---

## 1. Technology Stack

| Layer       | Technology                                                                                  |
|-------------|--------------------------------------------------------------------------------------------|
| **Backend** | Python 3.10+, FastAPI, `pypdf` (page-preserving PDF extraction), `scikit-learn` (TF-IDF)   |
| **Frontend**| React 18, TypeScript, Vite                                                                  |
| **LLM**     | Google Gemini API (`gemini-2.5-flash`) via REST, with local fallback if unavailable          |

---

## 2. Knowledge Base Documents

Place the three supplied PDFs inside `backend/data/`:

1. `01_Cedarhaven_Service_Request_Management.pdf` — CH-SRM-01 (8 pages)
2. `02_Cedarhaven_Bylaw_Enforcement.pdf` — CH-BYL-02 (8 pages)
3. `03_Cedarhaven_Permits_and_Licensing.pdf` — CH-PER-03 (8 pages)

These PDFs are the **only** authoritative source. The chatbot will not use external knowledge.

---

## 3. Setup & Run Instructions

### Prerequisites

- Python 3.10 or newer
- Node.js 18+ and npm
- A Google Gemini API key (free tier available at https://aistudio.google.com/)

### Step 1: Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

Create a `.env` file in the `backend/` directory:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

> **Security:** The `.env` file is listed in `.gitignore` and will not be committed. Never commit API keys.

### Step 3: Start the Backend

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/api/health` to verify the PDFs loaded.

### Step 4: Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## 4. How It Works

### PDF Ingestion
On startup, the backend reads all three PDFs page-by-page using `pypdf`. Each page becomes a chunk preserving:
- Original PDF filename
- A document ID (e.g., `CH-SRM-01`)
- The 1-indexed original page number
- Detected section identifier (e.g., `SRM 3`, `BYL 5`)

### Document Retrieval
When a user asks a question, the system:
1. Converts the question and all chunks into TF-IDF vectors (bigram, sublinear TF, English stop words removed).
2. Computes cosine similarity between the question and every chunk.
3. Applies keyword-based score boosting for domain terms.
4. Returns the top 6 most relevant chunks (may come from any or all three PDFs).

### Answer Generation
The top chunks are formatted as context and sent to the Gemini LLM alongside a strict system prompt. The system prompt:
- Requires the model to answer only from the provided excerpts
- Prohibits inventing fees, deadlines, eligibility, or contact details
- Requires inline citations with document ID and page number
- Requires the answer to be written in the user's selected language
- Rejects prompt-override attempts

If the Gemini API is unavailable (no key, timeout, network issues), a local fallback returns the raw retrieved excerpts with proper citations.

### Citation Verification
After the LLM responds, the backend:
1. Parses the `cited_pages` from the JSON response
2. Cross-references each citation against the actual loaded chunks
3. Only includes verified citations (where the doc_id + page_number match a real chunk)
4. Also scans the answer text for inline citation patterns and verifies those

### Multilingual Support
The language selector in the header lets residents choose from 11 languages common in Canadian municipalities. The selected language is sent to the backend and included in the system prompt so the LLM responds in that language. Document names, IDs, and page numbers remain in English for accuracy.

### Unsupported Questions
If a question asks about something not covered in the three PDFs, the system prompt instructs the LLM to clearly state the information is not available. If a user tries to override the grounding restriction, the system politely declines.

---

## 5. Environment Variables

| Variable        | Required | Description                    |
|----------------|----------|--------------------------------|
| `GEMINI_API_KEY` | Yes*     | Google Gemini API key          |

*Without a key, the chatbot falls back to showing raw retrieved excerpts instead of LLM-generated answers.

---

## 6. Brief Design Notes

- **LLM chosen:** Google Gemini 2.5 Flash — fast, cost-effective, supports structured JSON output and multilingual responses.
- **Retrieval approach:** TF-IDF with bigrams + cosine similarity + domain keyword boosting. Chosen for simplicity, zero external dependencies, and no paid embedding services.
- **Citation tracking:** Each PDF page is ingested as a single chunk with its original filename and 1-indexed page number. The LLM returns structured JSON with `cited_pages`, which are verified against loaded chunks before display.
- **Unsupported questions:** The system prompt explicitly instructs the LLM to decline answering questions not supported by the documents and to reject override attempts.

---

## 7. AI-Tool Disclosure

The following AI coding assistants were used during development:

- **Google Gemini (Antigravity/Claude)** — Used for code scaffolding, debugging, and reviewing the RAG pipeline logic.
- **ChatGPT / Copilot** — Used for boilerplate generation and CSS styling assistance.

All generated code was reviewed, understood, and adapted to meet the specific assignment requirements. The developer can explain every component during the demonstration.

---

## 8. Project Structure

```
cedarhaven-assistant/
├── backend/
│   ├── data/               # Three Cedarhaven PDF policy documents
│   ├── main.py             # FastAPI application with /api/chat and /api/health
│   ├── rag_engine.py       # PDF ingestion, TF-IDF retrieval, Gemini LLM integration
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # API key (not committed)
│   └── .env.example        # Template for .env
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React chat interface
│   │   ├── App.css         # Application styles
│   │   └── main.tsx        # React entry point
│   ├── index.html          # HTML entry point
│   ├── package.json        # Node dependencies
│   ├── tsconfig.json       # TypeScript configuration
│   └── vite.config.ts      # Vite build configuration
├── .gitignore
└── README.MD
```
