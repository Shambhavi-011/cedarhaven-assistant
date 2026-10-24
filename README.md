# 🏛️ Town of Cedarhaven — Municipal AI Assistant

A multilingual, document-grounded AI chatbot that helps residents find information about municipal services, by-law enforcement, permits, and licensing. Built using **Retrieval-Augmented Generation (RAG)**, it provides answers backed by the town's official project documents.

## ✨ Features

- **Multilingual Support:** Choose from English, French, German, Spanish, Mandarin, and Hindi.
- **RAG-Based Question Answering:** Retrieves relevant information from three municipal policy PDFs before generating an answer.
- **Clickable Source Citations:** Each citation can be clicked to inspect its supporting source.
- **PDF Evidence Highlighting:** Opens the relevant source evidence with yellow highlighting for easy verification.
- **Multi-Document Retrieval:** Answers questions requiring information from multiple policy documents.
- **AI Guardrails:** Avoids unsupported claims, handles missing information, and rejects attempts to override document-grounding rules.
- **Citation Validation:** Verifies document IDs and page numbers against the loaded source content.

## 🛠️ Tech Stack

- **Frontend:** React, TypeScript, Vite
- **Backend:** Python, FastAPI
- **Retrieval:** TF-IDF, cosine similarity, keyword-based ranking
- **LLM:** Google Gemini 2.5 Flash
- **Document Processing:** pypdf

## ⚙️ How It Works

1. Extracts text from the three policy PDFs while preserving page numbers and section details.
2. Retrieves relevant passages for the user's question using TF-IDF similarity and keyword boosting.
3. Generates a response using Gemini with strict document-grounding instructions.
4. Validates citations and displays the answer with clickable source evidence.

## 🚀 Getting Started

**Prerequisites:** Python 3.10+, Node.js 18+, npm, and a Google Gemini API key.

**1. Clone the repository**

```bash
git clone https://github.com/Shambhavi-011/cedarhaven-assistant.git
cd cedarhaven-assistant
```

**2. Start the backend**

```bash
cd backend
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```env
GEMINI_API_KEY=your-gemini-api-key
```

Run the backend:

```bash
uvicorn main:app --reload --port 8000
```

**3. Start the frontend**

Open a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

**Note:** Place the three required Cedarhaven policy PDFs in `backend/data/` before running the application. Keep API keys private.

## 🎯 Project Goal

To demonstrate how a RAG-based AI assistant can make policy information easier to access while keeping responses traceable, verifiable, and grounded in source documents.

**GitHub:** [cedarhaven-assistant](https://github.com/Shambhavi-011/cedarhaven-assistant)
