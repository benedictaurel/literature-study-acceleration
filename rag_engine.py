import os
import shutil
import requests
import arxiv
import base64
import glob
import subprocess
from dotenv import load_dotenv
from PIL import Image

# Use SentenceTransformer directly for CLIP
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from oran_scraper import ORANScraper
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
import uuid

# Import the new Multimodal Loader
from multimodal_loader import MultimodalPDFLoader

class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    # We won't use weights for simple RRF, but keeping interface similar
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Gather results
        all_docs_lists = []
        for retriever in self.retrievers:
            all_docs_lists.append(retriever.invoke(query))
        
        # Reciprocal Rank Fusion (RRF)
        rrf_score = {}
        c = 60
        for doc_list in all_docs_lists:
            for rank, doc in enumerate(doc_list):
                # Use page_content as key for deduplication (simplistic)
                key = doc.page_content 
                if key not in rrf_score:
                    rrf_score[key] = {"doc": doc, "score": 0.0}
                rrf_score[key]["score"] += 1.0 / (c + rank + 1)
        
        # Sort by score
        sorted_docs = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]



# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, google_api_key=None, openalex_api_key=None):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.openalex_api_key = openalex_api_key or os.getenv("OPENALEX_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("Google API Key is required.")

        # Setup LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=16384,
            google_api_key=self.google_api_key
        )

        # Setup Embedding Models
        # 1. Text Embedding: MiniLM
        self.text_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Image Embedding: CLIP (ViT-B-32)
        # Using SentenceTransformer directly
        self.clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
        
        # Vector Stores
        self.vectorstore_text = None
        self.vectorstore_images = None
        
        self.retriever = None
        self.agent = None
        self.on_file_processed_callback = None

    def set_file_callback(self, callback):
        """
        Set a callback function to be called when a new file is processed.
        callback(file_path)
        """
        self.on_file_processed_callback = callback

    def _init_text_store_if_needed(self):
        if self.vectorstore_text is None:
             pass

    def initialize_vector_store(self, db_path=None, folder_path="./paper/papers", process_new=False):
        """
        Initializes the vector store in-memory using MultimodalPDFLoader.
        Scans values in folder_path.
        """
        self.vectorstore_text = None
        self.vectorstore_images = None
        self.retriever = None
        # self.processed_files is now persistent and NOT reset here
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print("Loading documents from", folder_path)
        
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        all_text_docs = []
        all_image_docs= [] # List of (image_path, embedding)
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file}...")
                docs, image_data_list = self._process_single_pdf_cached(pdf_file)
                all_text_docs.extend(docs)
                all_image_docs.extend(image_data_list)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

        # --- 1. Text Vector Store ---
        if all_text_docs:
            self._build_text_store(all_text_docs)
        else:
            print("No text documents found.")

        # --- 2. Image Vector Store (CLIP) ---
        self._build_image_store(all_image_docs)
        
    def _process_single_pdf_cached(self, pdf_file):
        """
        Helper to run loader and return text docs and image embedding data.
        """
        loader = MultimodalPDFLoader(pdf_file)
        docs = loader.load()
        
        image_data_list = []
        seen_images = set()

        # 1. Inspect Docs for image paths
        for doc in docs:
            paths = doc.metadata.get("image_paths", "")
            if paths:
                for path in paths.split(","):
                    if path and path not in seen_images and os.path.exists(path):
                        seen_images.add(path)
                        try:
                            image = Image.open(path)
                            # Skip problematic images (too small or unusual dimensions)
                            width, height = image.size
                            if width < 10 or height < 10:
                                print(f"Skipping small image {path}: {width}x{height}")
                                continue
                            # Convert to RGB if needed (fixes grayscale/RGBA issues)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            embedding = self.clip_model.encode(image).tolist()
                            image_data_list.append({
                                "path": path,
                                "embedding": embedding,
                                "filename": os.path.basename(path)
                            })
                        except Exception as e:
                            print(f"Failed to embed {path}: {e}")
        
        return docs, image_data_list

    def _build_text_store(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        if self.vectorstore_text:
             self.vectorstore_text.add_documents(chunks)
        else:
            self.vectorstore_text = Chroma.from_documents(
                documents=chunks,
                embedding=self.text_embedding_model,
                collection_name="text_collection"
            )
            
        # Rebuild Hybrid Logic
        all_chunks = self.vectorstore_text.get(include=["documents", "metadatas"])
        recreated_docs = []
        if all_chunks['documents']:
             for i, content in enumerate(all_chunks['documents']):
                 recreated_docs.append(Document(page_content=content, metadata=all_chunks['metadatas'][i]))

        chroma_retriever = self.vectorstore_text.as_retriever(search_kwargs={"k": 5})
        
        if recreated_docs:
            bm25_retriever = BM25Retriever.from_documents(recreated_docs)
            bm25_retriever.k = 5
            self.retriever = HybridRetriever(retrievers=[bm25_retriever, chroma_retriever])
            print(f"Text Vector Store updated. Total chunks: {len(recreated_docs)}")

    def _build_image_store(self, image_data_list):
        if not image_data_list:
            return

        print(f"Embedding {len(image_data_list)} images...")
        embeddings = [item['embedding'] for item in image_data_list]
        metadatas = [{"image_path": item['path']} for item in image_data_list]
        texts = [item['filename'] for item in image_data_list] # Dummy text
        ids = [str(uuid.uuid4()) for _ in image_data_list]

        if self.vectorstore_images:
            self.vectorstore_images._collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
                ids=ids
            )
        else:
             self.vectorstore_images = Chroma(
                collection_name="image_collection",
                embedding_function=None 
            )
             self.vectorstore_images._collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
                ids=ids
            )
        print("Image Vector Store updated.")

    def process_file(self, file_path):
        """
        Incrementally process a new PDF file and add it to the stores.
        """
        print(f"Incrementally processing: {file_path}")
        try:
             docs, image_data_list = self._process_single_pdf_cached(file_path)
             if docs:
                 self._build_text_store(docs)
             if image_data_list:
                 self._build_image_store(image_data_list)
        except Exception as e:
            print(f"Error incrementally processing {file_path}: {e}")
            return False
        
        return True

    # ------------------------------------------------------------------------
    # Tools (Moved to Class Methods for easier testing/access)
    # ------------------------------------------------------------------------

    def search_local_papers(self, query: str):
        """
        Use this tool FIRST. It searches the user's uploaded documents (PDFs) for text, tables, AND semantically matching images.
        Returns text content and paths to relevant images.
        If this tool returns "EMPTY", you MUST use the 'search_openalex' tool next.
        """
        results_text = []
        results_images = []
        
        # A. Search Text Store
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
                for doc in docs:
                    source_name = os.path.basename(doc.metadata.get('source', 'unknown'))
                    page_num = doc.metadata.get('page', 0)
                    doc_type = doc.metadata.get('type', 'text')
                    
                    # Existing metadata link
                    image_paths_meta = doc.metadata.get('image_paths', "")
                    
                    entry = f"[Type: {doc_type} | Source: {source_name} | Page: {page_num}]\nContent: {doc.page_content}"
                    if image_paths_meta:
                            entry += f"\n(Page contains images: {image_paths_meta})"
                    results_text.append(entry)
            except Exception as e:
                print(f"Text search error: {e}")

        # B. Search Image Store (CLIP)
        if self.vectorstore_images:
            try:
                # Search using text query -> images
                # 1. Embed text query using CLIP
                query_embedding = self.clip_model.encode(query).tolist()
                
                # 2. Search Chroma using vector
                # Chroma's similarity_search_by_vector returns just docs
                image_results = self.vectorstore_images.similarity_search_by_vector(query_embedding, k=3)
                
                for img_doc in image_results:
                    path = img_doc.metadata.get('image_path')
                    if path:
                        results_images.append(f"Found visual match: {path}")
            except Exception as e:
                print(f"Image search error: {e}")

        if not results_text and not results_images:
            return "LOCAL_SEARCH_EMPTY: No documents or images found. Proceed to 'search_openalex'."

        # Combine
        final_output = "### Text Retrieval Results:\n" + ("\n\n".join(results_text) if results_text else "None")
        final_output += "\n\n### Visual Retrieval Results (CLIP Matches):\n" + ("\n".join(results_images) if results_images else "None")
        
        return final_output

    def inspect_image(self, image_path: str, query: str = "Describe this image in detail and extract any text or data."):
        """
        Use this tool to see/analyze an image file found by 'search_local_papers'.
        Provide the absolute 'image_path' and a 'query' about what to look for.
        """
        try:
            if not os.path.exists(image_path):
                return f"IMAGE_ERROR: File not found at {image_path}"
            
            # Read image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Create Multimodal Message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            
            # Invoke LLM directly
            response = self.llm.invoke([message])
            return f"IMAGE_ANALYSIS_RESULT:\n{response.content}"
            
        except Exception as e:
            return f"IMAGE_INSPECTION_ERROR: {str(e)}"

    def search_openalex(self, query: str):
        """
        Use this tool SECOND if 'search_local_papers' fails or is empty.
        Searches OpenAlex for research papers.
        """
        base_url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "filter": "open_access.is_oa:true", 
            "per_page": 3,
        }
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key

        try:
            response = requests.get(base_url, params=params, timeout=20)
            if response.status_code != 200:
                return f"OPENALEX_ERROR: Status {response.status_code}. Proceed to 'search_arxiv'."
            
            data = response.json()
            results = []
            for work in data.get('results', []):
                title = work.get('title', 'No Title')
                item_id = work.get('id')
                results.append(f"Title: {title}\nID: {item_id}")

            if not results:
                return "OPENALEX_EMPTY: No results found. Proceed to 'search_arxiv'."
            
            # DOWNLOAD Logic
            downloaded_infos = []
            for work in data.get('results', []):
                try:
                    pdf_url = work.get('best_oa_location', {}).get('pdf_url')
                    title = work.get('title', 'Unknown Title')
                    if pdf_url:
                        # Sanitize filename
                        safe_title = "".join(x for x in title if x.isalnum() or x in " _-")[:50]
                        filename = f"openalex_{safe_title}.pdf"
                        path = os.path.join("./paper/papers", filename)
                        
                        if not os.path.exists("./paper/papers"):
                            os.makedirs("./paper/papers")
                        
                        # Check exist
                        if os.path.exists(path):
                            print(f"File exists, skipping download: {path}")
                            downloaded_infos.append(f"Found local: {title}")
                        else:
                            response_pdf = requests.get(pdf_url, timeout=30)
                            if response_pdf.status_code == 200:
                                with open(path, 'wb') as f:
                                    f.write(response_pdf.content)
                                downloaded_infos.append(f"Downloaded: {title}")
                                
                                # --- AUTOMATIC BIBTEX GENERATION ---
                                try:
                                    # Generate a simple BibTeX entry
                                    # Extract authors
                                    authors_list = work.get('authorships', [])
                                    authors_names = [a.get('author', {}).get('display_name', '') for a in authors_list]
                                    authors_str = " and ".join(filter(None, authors_names))
                                    
                                    year = work.get('publication_year', '')
                                    journal = work.get('primary_location', {}).get('source', {}).get('display_name', 'OpenAlex')
                                    doi = work.get('doi', '').replace("https://doi.org/", "")
                                    
                                    # Create a unique key: AuthorYearTitleWord
                                    first_author_last = authors_names[0].split()[-1] if authors_names else "Unknown"
                                    safe_first_word = "".join(x for x in title.split()[0] if x.isalnum())
                                    cite_key = f"{first_author_last}{year}{safe_first_word}"
                                    
                                    bibtex = f"@article{{{cite_key},\n"
                                    bibtex += f"  title={{{title}}},\n"
                                    bibtex += f"  author={{{authors_str}}},\n"
                                    bibtex += f"  journal={{{journal}}},\n"
                                    bibtex += f"  year={{{year}}},\n"
                                    if doi:
                                        bibtex += f"  doi={{{doi}}}\n"
                                    bibtex += "}"
                                    
                                    self._append_bibtex(bibtex)
                                    downloaded_infos.append(f" (Auto-BibTeX added: {cite_key})")
                                    
                                except Exception as bib_err:
                                    print(f"Failed to auto-generate BibTeX for OpenAlex: {bib_err}")
                                # -----------------------------------

                            else:
                                downloaded_infos.append(f"Failed to download: {title}")
                                continue

                        # Process immediately (checks cache inside)
                        self.process_file(path)

                except Exception as e:
                    print(f"Error processing OpenAlex paper: {e}")

            # Verify and Return Content
            # Instead of asking agent to search again, we do it for them.
            verification_results = self.search_local_papers(query)
            
            summary = "\n".join(downloaded_infos)
            return f"OPENALEX SEARCH RESULT:\n{summary}\n\nAUTOMATIC LOCAL RETRIEVAL:\n{verification_results}"
        except Exception as e:
            return f"OPENALEX_EXCEPTION: {str(e)}. Proceed to 'search_arxiv'."

    def search_arxiv(self, query: str):
        """
        Use this tool THIRD if 'search_openalex' fails or is empty.
        Searches ArXiv for scientific papers.
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
            results = []
            for r in client.results(search):
                results.append(f"Title: {r.title}\nSummary: {r.summary}\nURL: {r.entry_id}")

            if not results:
                return "ARXIV_EMPTY: No papers found on ArXiv. Proceed to 'search_semantic_scholar'."
            
            # DOWNLOAD Logic
            downloaded_infos = []
            for r in client.results(search):
                try:
                    title = r.title
                    safe_title = "".join(x for x in title if x.isalnum() or x in " _-")[:50]
                    filename = f"arxiv_{safe_title}.pdf"
                    path = os.path.join("./paper/papers", filename)
                    
                    if not os.path.exists("./paper/papers"):
                        os.makedirs("./paper/papers")
                        
                    if os.path.exists(path):
                        print(f"File exists, skipping download: {path}")
                        downloaded_infos.append(f"Found local: {title}")
                    else:
                        r.download_pdf(dirpath="./paper/papers", filename=filename)
                        downloaded_infos.append(f"Downloaded: {title}")
                        
                        # --- AUTOMATIC BIBTEX GENERATION ---
                        try:
                            # ArXiv result:
                            # r.authors (list of Author objects w/ name)
                            # r.published (datetime)
                            # r.entry_id (url)
                            # r.doi
                            
                            authors_names = [a.name for a in r.authors]
                            authors_str = " and ".join(authors_names)
                            year = str(r.published.year)
                            
                            # Key
                            first_author_last = authors_names[0].split()[-1] if authors_names else "Unknown"
                            safe_first_word = "".join(x for x in title.split()[0] if x.isalnum())
                            cite_key = f"{first_author_last}{year}{safe_first_word}"
                            
                            bibtex = f"@article{{{cite_key},\n"
                            bibtex += f"  title={{{title}}},\n"
                            bibtex += f"  author={{{authors_str}}},\n"
                            bibtex += f"  journal={{arXiv preprint arXiv:{r.entry_id.split('/')[-1]}}},\n"
                            bibtex += f"  year={{{year}}},\n"
                            bibtex += f"  url={{{r.entry_id}}}\n"
                            bibtex += "}"
                            
                            self._append_bibtex(bibtex)
                            downloaded_infos.append(f" (Auto-BibTeX added: {cite_key})")
                            
                        except Exception as bib_err:
                             print(f"Failed to auto-generate BibTeX for ArXiv: {bib_err}")
                        # -----------------------------------
                    
                    # Process immediately
                    self.process_file(path)
                    
                except Exception as e:
                    print(f"Error processing ArXiv paper: {e}")
                    
            # Verify and Return Content
            verification_results = self.search_local_papers(query)
            
            summary = "\n".join(downloaded_infos)
            return f"ARXIV SEARCH RESULT:\n{summary}\n\nAUTOMATIC LOCAL RETRIEVAL:\n{verification_results}"
        except Exception as e:
            return f"ARXIV_EXCEPTION: {str(e)}. Proceed to 'search_semantic_scholar'."



    def fetch_oran_specs(self, query: str):
        """
        Use this tool when the user asks about "O-RAN specifications", "specs", "O-RAN documents", or technical details likely found in O-RAN standards (e.g., "7.2x split", "E2 interface").
        This tool downloads relevant PDFs from the O-RAN website and re-indexes the knowledge base.
        """
        try:
            scraper = ORANScraper()
            downloaded = scraper.search_and_download(query)
            scraper.close()
            
            if not downloaded:
                return "ORAN_SCRAPER_EMPTY: No specifications found to download for this query."
            
            # Re-index to include new files
            # We need to re-scan the ./paper/papers directory (which includes ./paper/papers/O-RAN)
            self.initialize_vector_store(folder_path="./paper/papers") 
            
            return f"ORAN_SCRAPER_SUCCESS: Downloaded and indexed the following specifications: {', '.join([os.path.basename(f) for f in downloaded])}. You can now answer questions based on these documents."
        except Exception as e:
            return f"ORAN_SCRAPER_ERROR: {str(e)}"

    def read_latex_file(self, filename: str):
        """
        Use this tool to READ the current content of a LaTeX file before modifying it.
        This is crucial when you want to append to a file or fix a specific section.
        """
        paper_dir = os.path.join(os.getcwd(), "paper")
        file_path = os.path.join(paper_dir, filename)
        
        if not os.path.exists(file_path):
            return f"ERROR: File {filename} does not exist."
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"CONTENT OF {filename}:\n{content}"
        except Exception as e:
            return f"read_latex_error: {str(e)}"

    def write_latex_section(self, filename: str, content: str, action: str = "overwrite"):
        r"""
        Use this tool to write or update a specific LaTeX file locally.
        
        Arguments:
        - filename: The name of the .tex file (e.g., "paper.tex").
        - content: The LaTeX content to write.
        - action: "overwrite" (default) or "append".
        
        CRITICAL INSTRUCTION - CONTENT FORMAT:
        1. IF action="overwrite":
           The 'content' must be a COMPLETE, COMPILABLE LaTeX document.
           It MUST include:
           - \documentclass{article}
           - \usepackage{...}
           - \title{...} \author{...} \date{...}
           - \begin{document} ... 
           - ... content ...
           - \bibliographystyle{...} \bibliography{...}
           - \end{document}
           
        2. IF action="append":
           The 'content' must be the NEW SECTION content only (e.g., \section{New Section} ...).
           Do NOT include \documentclass or \begin{document} again.
           
           **PLACEMENT:** The tool will automatically insert your content BEFORE the Bibliography (if present) or BEFORE \end{document}.
        
        CITATION FORMAT (MANDATORY):
        - You MUST use \cite{KEY} for ALL citations in LaTeX content.
        - The KEY must come from the `list_bibtex_keys` tool output.
        - Example: "Recent studies \cite{Smith2023Deep} show that..."
        - FORBIDDEN: [Source: ...], [1], or any placeholder format.
        - Do not use Markdown formatting. Use proper LaTeX syntax only.
        """
        paper_dir = os.path.join(os.getcwd(), "paper")
        if not os.path.exists(paper_dir):
            os.makedirs(paper_dir)
            
        file_path = os.path.join(paper_dir, filename)
        
        try:
            if action == "append":
                if not os.path.exists(file_path):
                    return f"ERROR: Cannot append to {filename} because it does not exist. Use action='overwrite' to create it first."
                
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
                
                # Determine insertion point
                # Priority 1: Before \bibliographystyle
                # Priority 2: Before \bibliography
                # Priority 3: Before \end{document}
                
                insert_pos = -1
                
                if "\\bibliographystyle" in existing_content:
                    insert_pos = existing_content.find("\\bibliographystyle")
                elif "\\bibliography" in existing_content:
                    insert_pos = existing_content.find("\\bibliography")
                elif "\\end{document}" in existing_content:
                    insert_pos = existing_content.find("\\end{document}")
                else:
                    return f"ERROR: {filename} is malformed (missing \\end{{document}}). Cannot append automatically."
                
                # Insert content
                new_full_content = existing_content[:insert_pos] + "\n" + content + "\n" + existing_content[insert_pos:]
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_full_content)
                    
                return f"SUCCESS: Appended new content to {filename}."
                
            else:
                # Overwrite mode
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"SUCCESS: Overwritten {filename} locally at {file_path}."
            
        except Exception as e:
            return f"latex_writer_error: {str(e)}"

    def _append_bibtex(self, bibtex_content: str):
        """
        Internal helper to safely append BibTeX to references.bib
        """
        paper_dir = os.path.join(os.getcwd(), "paper")
        if not os.path.exists(paper_dir):
            os.makedirs(paper_dir)
            
        bib_path = os.path.join(paper_dir, "references.bib")
        
        try:
            import re
            match = re.search(r'@\w+\s*{\s*([^,]+),', bibtex_content)
            if not match:
                print("Invalid BibTeX format in _append_bibtex")
                return False
            
            key = match.group(1).strip()
            
            existing_content = ""
            if os.path.exists(bib_path):
                with open(bib_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
            
            if key in existing_content:
                print(f"BibTeX key '{key}' already exists.")
                return False
            
            with open(bib_path, "a", encoding="utf-8") as f:
                f.write("\n" + bibtex_content.strip() + "\n")
            print(f"Appended BibTeX for '{key}'")
            return True
        except Exception as e:
            print(f"Error appending BibTeX: {e}")
            return False

    def list_bibtex_keys(self):
        """
        Use this tool to LIST all available BibTeX keys in 'references.bib'.
        ALWAYS call this before writing a LaTeX section to ensure you use the correct citation keys.
        """
        paper_dir = os.path.join(os.getcwd(), "paper")
        bib_path = os.path.join(paper_dir, "references.bib")
        
        if not os.path.exists(bib_path):
            return "References file NOT found. No keys available."
            
        try:
            import re
            with open(bib_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find all @type{KEY,
            keys = re.findall(r'@\w+\s*{\s*([^,]+),', content)
            return f"AVAILABLE BIBTEX KEYS: {', '.join(keys)}"
        except Exception as e:
            return f"Error listing keys: {str(e)}"

    def add_bibtex_entry(self, bibtex_content: str):
        """
        Use this tool to MANUALLY add a BibTeX entry if the automatic system missed it.
        Arguments:
          - bibtex_content: Valid BibTeX entry.
        """
        if self._append_bibtex(bibtex_content):
            return "SUCCESS: Added BibTeX entry."
        else:
            return "INFO: Entry already exists or failed to add."

    def create_agent(self):
        """
        Creates the LangGraph React Agent with tools.
        """
        # Tools
        # We need to use StructuredTool.from_function because we are using instance methods.
        # This preserves the tool docstring, name, and args.
        
        tools = [
            StructuredTool.from_function(self.search_local_papers),
            StructuredTool.from_function(self.inspect_image),
            StructuredTool.from_function(self.search_openalex),
            StructuredTool.from_function(self.search_arxiv),
            StructuredTool.from_function(self.fetch_oran_specs),
            StructuredTool.from_function(self.read_latex_file),
            StructuredTool.from_function(self.list_bibtex_keys),
            StructuredTool.from_function(self.write_latex_section),
            StructuredTool.from_function(self.add_bibtex_entry)
        ]

        # System Message
        system_message = (
            "You are an AI Research Assistant. You help users find academic evidence and write research papers.\n\n"
            
            "## CORE RULES\n"
            "1.  ALWAYS respond with text after using tools. Never return empty.\n"
            "2.  Provide comprehensive, detailed responses. Explain concepts thoroughly.\n"
            "3.  For chat responses, use plain English or Markdown.\n"
            "4.  For LaTeX file content, use ONLY LaTeX syntax (no Markdown).\n\n"
            
            "## ANSWERING QUESTIONS (CHAT RESPONSES)\n"
            "1.  Call `search_local_papers` first.\n"
            "2.  If local search is empty, call `search_openalex` or `search_arxiv`.\n"
            "3.  In your CHAT response (not LaTeX), cite sources as [Source: filename | Page: X].\n"
            "4.  This [Source: ...] format is ONLY for chat responses, NEVER for LaTeX files.\n\n"
            
            "## WRITING/UPDATING LATEX FILES (MANDATORY WORKFLOW)\n"
            "When asked to write or add to a .tex file, you MUST follow these steps IN ORDER:\n"
            "1.  FIRST: Search for relevant papers to use as sources:\n"
            "    - Call `search_local_papers` to check existing documents.\n"
            "    - If insufficient, call `search_openalex` or `search_arxiv` to find and download papers.\n"
            "    - These search tools automatically download PDFs AND create BibTeX entries.\n"
            "2.  SECOND: Call `list_bibtex_keys` to get citation keys from DOWNLOADED papers only.\n"
            "3.  THIRD: Call `read_latex_file` to see existing content (if appending).\n"
            "4.  FOURTH: Call `write_latex_section` with content based on the DOWNLOADED papers.\n"
            "    - Use ONLY citations from papers you actually searched and downloaded.\n"
            "    - NEVER invent or hallucinate citation keys. Only use keys from `list_bibtex_keys`.\n"
            "    - If `list_bibtex_keys` returns no keys, you MUST search for papers first.\n\n"
            
            "## LATEX CITATION FORMAT (CRITICAL - READ CAREFULLY)\n"
            r"Inside LaTeX file content, you MUST:" "\n"
            r"- Use \cite{{KEY}} for ALL citations. Example: \cite{{Smith2023Deep}}" "\n"
            "- The KEY must EXACTLY match one from `list_bibtex_keys` output.\n"
            "- NEVER use [Source: ...] or [1] or any other format inside LaTeX.\n"
            "- If no matching BibTeX key exists, call `add_bibtex_entry` to create one first.\n\n"
            
            "## LATEX FORMATTING\n"
            r"- Use \textbf{{...}} for bold (NOT **)." "\n"
            r"- Use \textit{{...}} for italic (NOT *)." "\n"
            r"- Use \begin{{itemize}}...\end{{itemize}} for bullet lists." "\n"
            r"- Use \section{{...}}, \subsection{{...}} for headings." "\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{messages}"),
        ])

        # Using langgraph prebuilt agent as in notebook
        self.agent = create_react_agent(self.llm, tools, prompt=prompt)

    def ask(self, question: str):
        if not self.agent:
            return "Agent not initialized."
        
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"recursion_limit": 30}
        
        try:
            result = self.agent.invoke(inputs, config=config)
            last_message = result['messages'][-1]
            content = last_message.content
            
            # Handle case where content is a list (multi-part response)
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            if content and isinstance(content, str) and content.strip():
                return content
            
            # Fallback: Extract tool results and construct a response
            print(f"DEBUG: Model returned empty. Attempting to use tool results directly.")
            
            # Find the last tool message with content
            tool_content = None
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and msg.content:
                    # Check if it's a ToolMessage (has tool_call_id) or contains our markers
                    if hasattr(msg, 'tool_call_id') or 'Text Retrieval Results' in str(msg.content):
                        tool_content = msg.content
                        break
            
            if tool_content:
                # Use the LLM to summarize the tool content (single call, no tools)
                summary_prompt = f"Based on the following search results, provide a clear answer to: '{question}'\n\nSearch Results:\n{tool_content[:8000]}\n\nProvide a concise summary with source citations."
                try:
                    summary_response = self.llm.invoke(summary_prompt)
                    if summary_response.content:
                        return summary_response.content
                except Exception as sum_err:
                    print(f"Summary fallback failed: {sum_err}")
            
            # Final fallback: return the tool content directly
            if tool_content:
                return f"**Retrieved Information:**\n\n{tool_content[:6000]}"
            
            return "No relevant information found in the available documents."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL ERROR in ask(): {e}")
            return f"Error during generation: {str(e)}"
