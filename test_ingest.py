from ingest_paper import ingest_paper

# Replace with a real PDF on your machine
pdf_path = "e:\\WangResearch\\EllaModel_Wang.pdf"

paper = ingest_paper(pdf_path, figures_output_dir = "E:\\WangResearch\\Figures")

print("Pages extracted:", paper.num_pages)
print("First 500 chars of page 0:\n", paper.pages[0].text[:500])
print("\nExtracted figure paths:")

for fig in paper.figures:
    print(f"Page {fig.page_number}, image {fig.image_index}: {fig.image_path}")
