from pypdf import PdfReader

def load_pdfs(folder):
    texts = []
    for pdf in folder.glob("*.pdf"):
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts
