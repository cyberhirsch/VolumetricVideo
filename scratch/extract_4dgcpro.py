import fitz # PyMuPDF
import os

pdf_path = r"g:\3D Generation\Long Volumetric Video\docs\papers\4DGCPro_2025_arXiv2509.17513.pdf"
output_dir = r"g:\3D Generation\Long Volumetric Video\scratch"
output_path = os.path.join(output_dir, "4dgcpro_text.txt")

os.makedirs(output_dir, exist_ok=True)

print(f"Opening PDF: {pdf_path}")
doc = fitz.open(pdf_path)
print(f"Number of pages: {len(doc)}")

text_content = []
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    text_content.append(f"--- PAGE {page_num + 1} ---")
    text_content.append(text)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(text_content))

print(f"Extracted text successfully written to {output_path}")
