[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)


Start Script: you run python simple_agent.py.

↓

Setup: The script creates a folder named ADGM_REFERENCES to store its legal knowledge. (this is used for donloadable pdf's from website (when the direct link is not provided )  )

↓

Agent Initialization: The script reads all the documents (.pdf, .docx, .txt) from the ADGM_REFERENCES folder.  |( .txt is added because some links may in 'future' require to solve googles captcha ) , .pdf some donnloadable's are pdf's .

↓

Create "Memory": It breaks down all the text from the documents into small chunks and converts them into numerical codes (embeddings). These codes are stored in a fast, searchable database called a vector store. This vector store is the agent's memory of the legal rules.

↓

Launch Web Interface: The script starts the web interface (Gradio), which you will be able to see in your browser. 

    Interface lvl : You can Upload Documents for Review(.docx )


This is what happens when you interact with the agent through the web interface.

Upload Documents: You upload your documents (e.g., a Memorandum of Association) and click "Review Documents".

↓

it Identify's Document Type: The agent looks at the first part of each document with an AI model (gemini).

↓

Checklist: The agent checks if all the required documents for our task (e.g., company incorporation) have been uploaded.

If documents are missing: The agent stops and tells you which files are needed.

If all documents are present: The agent moves on to the next step. ( but to the current testing phase and ease of use . we only need to provide files in the Category folder ,

 example in "Category" folder we have a subfile as "Document/Template Type" ,here it must have at least one file corresponding to "General Incorporation, AoA,MoA, Registers, UBO, Board Resolutions" .

#ADGM_REFERENCES > Category > CategoryName > Document/Template Type > Document/Template Type > files  

↓

Deep Analysis (The "RAG" part): For each of your uploaded documents:

Retrieve: Here The agent searches its "memory" (the vector store) to find the most relevant legal rules and sections from our ADGM_REFERENCES folder.

Analyze: It takes our document text and the relevant rules it just found. It then sends both to an AI model (Gemini) with a prompt , to review this files Based on these rule and find any issues.

Output: The AI model gives back a structured response with any issues it found.

↓

Add Comments and Package:

The agent reads the AI's response and finds the exact spot in our original .docx file where each issue is .

It adds a comment to the document for each issue found.

It saves a new, commented version of the document.

Finally, it zips up all the new, commented documents into a single .zip file for you to download.

↓

Display Results: The web interface shows you a summary of the findings, a detailed report, and the link to download the .zip file. The process is complete.

Remember :a donloadable zip is provided , only if our agent has enough resources to prepare the findings and a detailed report .
