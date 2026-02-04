# AI Research Assistant (RAG Application)

This is a Streamlit-based AI Chat Application that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded academic papers. It also integrates with OpenAlex, ArXiv, and Semantic Scholar.

## Prerequisites

- Python 3.10+
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- (Optional) [OpenAlex API Key](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables** (Optional):
   Create a `.env` file in this directory and add your keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   OPENALEX_API_KEY=your_openalex_api_key_here
   ```
   *Alternatively, you can enter these keys in the App's sidebar.*

## Running the App

Run the following command in your terminal:

```bash
streamlit run app.py
```

## Features

- **Upload PDF**: Upload academic papers in the sidebar.
- **Process Documents**: Click "Process Documents" to index them into the local vector database (`chroma_db`).
- **Chat**: Ask questions. The AI will prioritize local documents. usage logic:
    1. search_local_papers
    2. search_openalex (if local is empty)
    3. search_arxiv
    4. search_semantic_scholar
