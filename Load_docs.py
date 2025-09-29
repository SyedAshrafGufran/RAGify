# load_docs.py
from pathlib import Path
import pypdf, docx

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

def load_folder(folder="docs"):
    texts = []
    for file in Path(folder).glob("**/*"):
        txt = load_text_from_file(file)
        if txt:
            texts.append((file.name, txt))
    return texts
