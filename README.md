# Literature Study Acceleration System

A comprehensive toolkit for accelerating literature study, featuring a RAG-powered chat assistant, automated O-RAN specification scraping, and specific paper version control.

## 🚀 Features

### 1. AI Research Assistant (RAG Chat)
-   **Interactive Chat**: Ask questions about your PDF documents.
-   **Hybrid Retrieval**: Combines local document search with external academic databases.
-   **Sources**:
    -   **Local**: Uploaded PDFs in the `paper/` directory.
    -   **External**: OpenAlex, ArXiv, Semantic Scholar.

### 2. O-RAN Scraper
-   **Automated Download**: Scrapes and downloads O-RAN specifications.
-   **Indexing**: Automatically prepares downloaded PDFs for the RAG engine.

### 3. Automated Paper Versioning
-   **Nested Repository**: The `paper/` directory is a standalone git repository, keeping your dataset feature separate from code.
-   **Daily Updates**: A Python script (`daily_paper_update.py`) automatically stages, commits, and pushes new papers with descriptive messages.

---

## 🛠️ Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd project
pip install -r requirements.txt
```

### 2. Configure Environment `.env`
Create a `.env` file in the project root:
```ini
GOOGLE_API_KEY=your_google_ai_studio_key
OPENALEX_API_KEY=your_openalex_key_optional
```

### 3. Setup Paper Repository (Critical)
The `paper/` directory is designed to be a separate git repository.
```bash
# Initialize the paper directory
cd paper
git init

# Link to your separate data repository
git remote add origin <YOUR_PAPER_REPO_URL>

# Pull existing data (if any)
git pull origin master
```

---

## 📖 Usage

### Running the AI Assistant
Start the Streamlit interface:
```bash
streamlit run app.py
```

### Automating Daily Updates
We provide a script to automatically push new papers to your remote data repository.

**Manual Run:**
```bash
python daily_paper_update.py
```

**Automatic Run (Windows Task Scheduler):**
1.  Open **Task Scheduler**.
2.  Create a **Basic Task** ("Daily Paper Update").
3.  Trigger: **Daily**.
4.  Action: **Start a program**.
    -   **Program**: Path to your `python.exe` (run `where python` to find it).
    -   **Arguments**: `daily_paper_update.py`
    -   **Start in**: Full path to this project folder (e.g., `C:\Users\...\project`).

---

## 📂 Project Structure
-   `app.py`: Main Streamlit application.
-   `rag_engine.py`: Core logic for retrieval and generation.
-   `oran_scraper.py`: Selenium script for O-RAN specs.
-   `daily_paper_update.py`: Automation script for git operations.
-   `paper/`: **(Nested Repo)** Stores PDFs and extracted images.
