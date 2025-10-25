# -----------------------------------------------------------------------------
# File name: Load_docs.py
# Authors: 1. Sufiya Sarwath - 1DS22CS218, 
#          2. Supriya R - 1DS22CS223, 
#          3. Syed Ashraf Gufran - 1DS22CS229, 
#          4. Yaseen Ahmed Khan - 1DS22CS257
#
# Guide: Dr Shobhana Padmanabhan
# Description: Script to extract text content from PDF, DOCX, and TXT files 
#              inside a specified folder for further processing or analysis.
# ------------------------------------------------------------------------------

from pathlib import Path
import pypdf
import docx


# Function to load and extract text from a single file (PDF, DOCX, or TXT)
def load_text_from_file(file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        return file_path.read_text(encoding="utf-8")
    else:
        return None


# Function to load all supported files from a folder
def load_folder(folder="docs"):
    texts = []
    for file in Path(folder).glob("**/*"):
        txt = load_text_from_file(file)
        if txt:
            texts.append((file.name, txt))
    return texts
