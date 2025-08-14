Upload document → review_documents(files) is called with the uploaded file objects.

Read content → For each file, _read_docx_text() (or similar) extracts the text for AI preview.

AI folder selection → AI reads a short preview and picks the most relevant folders from your local directory tree.

Load reference docs → _load_docs_from_folders() reads all files inside those selected folders.

Chunk & embed → Those reference docs are chunked and embedded fresh every run (dynamic RAG).

Pass to LLM → Only the retrieved top chunks plus the uploaded doc are sent to the analysis chain.

AI analysis output → Parsed JSON of issues found.

Review doc creation → _add_comments_to_docx() creates a commented version of the uploaded doc.

ZIP packaging → All reviewed docs are zipped into reviewed_documents.zip for download.

So basically: Upload → Preview → Folder pick → Load refs → Embed → Retrieve → Analyze → Annotate → Zip.
