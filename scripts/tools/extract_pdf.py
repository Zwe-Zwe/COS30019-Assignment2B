from pypdf import PdfReader

reader = PdfReader("COS30019-2025 S2-A2_B.pdf")
with open("pdf_content.txt", "w", encoding="utf-8") as f:
    for page in reader.pages:
        f.write(page.extract_text() + "\n")
