import os
import io
import json
import re
import urllib.parse
from collections import defaultdict
import zipfile
import mimetypes
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from docx import Document
import requests
from bs4 import BeautifulSoup

# --- Configuration ---

load_dotenv()

# It's recommended to set API_KEY in your environment for security
API_KEY = os.getenv("API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Error: GOOGLE_API_KEY environment variable not set. "
        "Please create a file named .env in the project root and add the following line:\n"
        "GOOGLE_API_KEY=\"YOUR_API_KEY_HERE\""
    )

ADGM_CHECKLISTS = {
    "Company Incorporation": [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution",
        "Shareholder Resolution",
        "Incorporation Application Form"
    ]
}

class Issue(BaseModel):
    document: str = Field(description="The name of the document where the issue was found.")
    section: str = Field(description="The specific clause, section, or paragraph where the issue is located.")
    issue: str = Field(description="A clear description of the red flag or non-compliance issue.")
    severity: str = Field(description="The severity of the issue (e.g., High, Medium, Low).")
    suggestion: str = Field(description="A legally compliant suggestion for remediation, citing ADGM rules where possible.")

class AnalysisResult(BaseModel):
    """A model for the direct output of the issue analysis chain."""
    issues_found: List[Issue] = Field(description="A list of all issues found in the document.")

class ReviewReport(BaseModel):
    process: str = Field(description="The legal process identified from the documents, e.g., 'Company Incorporation'.")
    documents_uploaded: int = Field(description="The number of documents uploaded by the user.")
    required_documents: int = Field(description="The total number of documents required for the identified process.")
    missing_documents: List[str] = Field(description="A list of names of the missing documents.")
    issues_found: List[Issue] = Field(description="A list of all issues found across all documents.")


class CorporateAgent:
    """
    An intelligent agent for reviewing, validating, and assisting with ADGM
    compliance documentation.
    """
    def __init__(self, knowledge_base_path: str):
        """
        Initializes the agent by loading the ADGM knowledge base and setting up the LLM chain.
        """
        print("Initializing Corporate Agent...")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Load all documents from the knowledge base directory
        print(f"Loading knowledge base from {knowledge_base_path}...")
        loaders = {
            ".pdf": PyMuPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
        }
        all_docs = []
        for ext, loader_cls in loaders.items():
            loader = DirectoryLoader(knowledge_base_path, glob=f"**/*{ext}", loader_cls=loader_cls, show_progress=True)
            docs = loader.load()
            if docs:
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} {ext} files.")

        if not all_docs:
            print("⚠️ Warning: Knowledge base is empty. Document review will be disabled.")
            print("   Please add reference documents via the 'Knowledge Base Management' tab and restart the app.")
            self.retriever = None
            self.llm = llm
            print("✅ Corporate Agent is ready (with no knowledge base).")
            return
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)

        print("Creating vector store...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        self.retriever = vectorstore.as_retriever()

        # --- Chains for different tasks ---
        self.llm = llm
        self.analysis_parser = JsonOutputParser(pydantic_object=AnalysisResult)
        self.doc_type_identifier_chain = self._create_doc_type_chain()
        self.analysis_chain = self._create_analysis_chain()

        print("✅ Corporate Agent is ready.")

    def _create_doc_type_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "Based on the following text, what is the most likely document type? "
            "Choose from: Articles of Association, Memorandum of Association, Board Resolution, "
            "Shareholder Resolution, Incorporation Application Form, or Unknown. "
            "Document Text:\n\n{text}\n\nDocument Type:"
        )
        return prompt | self.llm

    def _create_analysis_chain(self):
        system_prompt = """You are an expert legal assistant specializing in ADGM (Abu Dhabi Global Market) regulations. Your task is to review a legal document and identify any red flags, ambiguities, or areas of non-compliance based on the provided ADGM context.

Analyze the user's document text thoroughly. Compare it against the provided ADGM context to find issues.

For each issue you find, provide the following details:
- The specific clause or section where the issue is located.
- A clear description of the issue (e.g., 'Incorrect jurisdiction mentioned').
- The severity of the issue (High, Medium, or Low).
- A suggestion for how to fix the issue, citing the relevant ADGM rule from the context if possible.

If you find no issues, return an empty list.

<context>
{context}
</context>

Document Text:
{input}

Respond with a JSON object containing a single key "issues_found" which is a list of objects, where each object has "section", "issue", "severity", and "suggestion" keys.
"""
        prompt = ChatPromptTemplate.from_template(system_prompt)
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def _read_docx_text(self, file_path):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _add_comments_to_docx(self, file_path, issues):
        doc = Document(file_path)
        for issue in issues:
            for para in doc.paragraphs:
                # Simple search for the section text. A more robust implementation
                # might use fuzzy matching or character-level indexing.
                if issue['section'] in para.text:
                    comment_text = f"Issue: {issue['issue']}\nSeverity: {issue['severity']}\nSuggestion: {issue['suggestion']}"
                    para.add_comment(comment_text, author="Corporate Agent")
                    break # Add comment to the first matching paragraph and move to next issue
        return doc

    def review_documents(self, files: list):
        """
        Orchestrates the full document review process.
        """
        if not files:
            return "Please upload at least one document.", {}, None

        if self.retriever is None:
            return "The agent's knowledge base is empty. Please add reference documents via the 'Knowledge Base Management' tab and restart the application before reviewing documents.", {}, None

        # 1. Identify document types
        identified_docs = {}
        for file_obj in files:
            doc_text = self._read_docx_text(file_obj.name)
            response = self.doc_type_identifier_chain.invoke({"text": doc_text[:4000]}) # Use first 4k chars for speed
            doc_type = response.content.strip()
            identified_docs[os.path.basename(file_obj.name)] = {"type": doc_type, "path": file_obj.name}

        # 2. Checklist Verification (assuming 'Company Incorporation' for this example)
        process = "Company Incorporation"
        required = ADGM_CHECKLISTS[process]
        uploaded_types = [d['type'] for d in identified_docs.values()]
        missing_docs = [doc for doc in required if doc not in uploaded_types]

        report = {
            "process": process,
            "documents_uploaded": len(files),
            "required_documents": len(required),
            "missing_documents": missing_docs,
            "issues_found": []
        }

        if missing_docs:
            missing_str = ", ".join(missing_docs)
            message = (
                f"It appears that you’re trying to incorporate a company in ADGM. "
                f"Based on our reference list, you have uploaded {len(files)} out of {len(required)} required documents. "
                f"The missing document(s) appear to be: '{missing_str}'."
            )
            return message, report, None

        # 3. Full analysis if all documents are present
        all_issues = []
        reviewed_doc_paths = []

        for doc_name, doc_info in identified_docs.items():
            print(f"Analyzing document: {doc_name} (Type: {doc_info['type']})")
            doc_text = self._read_docx_text(doc_info['path'])
            response = self.analysis_chain.invoke({"input": doc_text})

            try:
                # The response['answer'] should be a JSON string
                analysis_result = json.loads(response['answer'])
                doc_issues = analysis_result.get("issues_found", [])
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse LLM response for {doc_name} as JSON. Response: {response['answer']}")
                doc_issues = []
            
            for issue in doc_issues:
                issue['document'] = doc_name # Add document name to each issue

            if doc_issues:
                all_issues.extend(doc_issues)
                # Create and save the reviewed docx with comments
                reviewed_doc = self._add_comments_to_docx(doc_info['path'], doc_issues)
                reviewed_doc_path = f"reviewed_{doc_name}"
                reviewed_doc.save(reviewed_doc_path)
                reviewed_doc_paths.append(reviewed_doc_path)

        report["issues_found"] = all_issues
        message = "Review complete. All required documents are present."
        if all_issues:
            message += f" Found {len(all_issues)} issue(s)."
        else:
            message += " No issues found."

        # 4. Package reviewed files into a zip
        if not reviewed_doc_paths:
            return message, report, None

        zip_path = "reviewed_documents.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in reviewed_doc_paths:
                zip_file.write(file_path, os.path.basename(file_path))

        # Clean up individual reviewed files
        for file_path in reviewed_doc_paths:
            os.remove(file_path)

        return message, report, zip_path


# --- New Functions for Knowledge Base Management ---

def _sanitize_for_path(text: str) -> str:
    """
    Replaces spaces with underscores and removes characters that are invalid
    for file or directory names.
    """
    text = text.replace(" ", "_").replace("&", "and")
    return re.sub(r'[\\/*?:"<>|]', "", text)

def download_and_save_reference(url: str, category: str, doc_type: str):
    """
    Downloads a reference document from a URL and saves it to a structured
    directory within the knowledge base.
    """
    if not all([url, category, doc_type]):
        return "URL, Category, and Document Type are required."

    s_category = _sanitize_for_path(category)
    s_doc_type = _sanitize_for_path(doc_type)

    knowledge_base_path = "ADGM_REFERENCES"
    # Create the nested directory structure: ADGM_REFERENCES/Category/Document_Type/
    target_dir = os.path.join(knowledge_base_path, s_category, s_doc_type)
    os.makedirs(target_dir, exist_ok=True)

    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        # --- Determine a sensible filename ---
        filename = ""
        if "content-disposition" in response.headers:
            # Try to get filename from Content-Disposition header
            disposition = response.headers['content-disposition']
            filenames = re.findall('filename="?([^"]+)"?', disposition)
            if filenames:
                filename = filenames[0]

        if not filename:
            # Fallback to getting filename from URL path
            filename = os.path.basename(urllib.parse.urlparse(url).path)

        # If still no filename, it's likely a webpage, not a direct file link.
        # We'll save it as text.
        is_html_page = 'html' in response.headers.get('content-type', '')
        if not filename or is_html_page:
            # Use the doc_type as the filename and ensure .txt extension for HTML
            filename = f"{s_doc_type}.txt"

        # Sanitize the final filename just in case
        filename = _sanitize_for_path(filename)

        file_path = os.path.join(target_dir, filename)

        if os.path.exists(file_path):
            return f"✅ File already exists at: {file_path}. Skipping download."

        # --- Save the content ---
        if file_path.endswith('.txt'):
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        else:
            with open(file_path, 'wb') as f:
                f.write(response.content)

        return f"✅ Successfully saved to {file_path}.\n\n⚠️ Please restart the application to load the new document into the agent's knowledge base."

    except requests.exceptions.RequestException as e:
        return f"❌ Error downloading from URL: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

# --- Gradio UI and Application Entry Point ---

def main():
    """Initializes the agent and launches the Gradio web interface."""
    
    # Path to the folder containing your ADGM reference documents
    knowledge_base_path = "ADGM_REFERENCES"

    os.makedirs(knowledge_base_path, exist_ok=True)

    try:
        agent = CorporateAgent(knowledge_base_path=knowledge_base_path)
    except RuntimeError as e:
        print(e)
        return

    def gradio_interface_fn(files):
        """Wrapper function for the Gradio interface that handles UI updates."""
        if not files:
            return "Please upload documents to review.", {}, gr.DownloadButton(visible=False)

        message, report_dict, zip_path = agent.review_documents(files)

        download_button_update = gr.DownloadButton(value=zip_path, visible=True) if zip_path else gr.DownloadButton(visible=False)
        return message, report_dict, download_button_update

    with gr.Blocks(theme=gr.themes.Soft(), title="ADGM Corporate Agent") as demo:
        gr.Markdown("# ADGM-Compliant Corporate Agent with Document Intelligence")

        with gr.Tabs():
            with gr.TabItem("Document Reviewer"):
                gr.Markdown("Upload one or more `.docx` files for review. The agent will check for completeness, identify red flags based on ADGM rules, and return a downloadable zip file with commented documents and a JSON summary.")
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload .docx Documents", file_count="multiple", file_types=[".docx"])
                        submit_btn = gr.Button("Review Documents", variant="primary")

                    with gr.Column(scale=2):
                        output_message = gr.Textbox(label="Summary")
                        output_json = gr.JSON(label="Analysis Report")
                        download_btn = gr.DownloadButton("Download Reviewed Documents (.zip)", visible=False)

                submit_btn.click(
                    fn=gradio_interface_fn,
                    inputs=[file_input],
                    outputs=[output_message, output_json, download_btn]
                )

            with gr.TabItem("Knowledge Base Management"):
                gr.Markdown("Add new reference documents from a URL to the agent's knowledge base. The application must be **restarted** for new documents to take effect.")
                with gr.Row():
                    with gr.Column(scale=2):
                        url_input = gr.Textbox(label="Document URL", placeholder="https://www.adgm.com/documents/...")
                        category_input = gr.Textbox(label="Category", placeholder="e.g., Company Formation & Governance")
                        doc_type_input = gr.Textbox(label="Document/Template Type", placeholder="e.g., General Incorporation, AoA")
                        add_ref_btn = gr.Button("Add to Knowledge Base", variant="primary")
                    with gr.Column(scale=1):
                        kb_output_message = gr.Textbox(label="Status", interactive=False)

                add_ref_btn.click(
                    fn=download_and_save_reference,
                    inputs=[url_input, category_input, doc_type_input],
                    outputs=[kb_output_message]
                )

    print("Launching Gradio Interface...")
    demo.launch()

if __name__ == "__main__":
    main()