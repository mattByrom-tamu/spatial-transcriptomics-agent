# reextract_figures.py

import os
from ingest_paper import ingest_paper


# Folder containing your PDFs
PDF_FOLDER = r"E:\WangResearch\Papers for GPT"

# Where all extracted figures will be stored (per-paper subfolders inside this)
FIGURES_OUTPUT_DIR = r"E:\WangResearch\extracted_figures"


def find_pdfs(folder: str):
    """
    Return full paths of all .pdf files in the given folder.
    """
    pdf_paths = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(folder, name))
    return sorted(pdf_paths)


def main():
    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder does not exist: {PDF_FOLDER}")

    pdf_paths = find_pdfs(PDF_FOLDER)
    if not pdf_paths:
        print(f"No PDFs found in {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_paths)} PDF(s) in {PDF_FOLDER}.\n")

    for pdf_path in pdf_paths:
        print(f"=== Re-extracting figures for: {pdf_path} ===")
        ingested = ingest_paper(
            pdf_path,
            figures_output_dir=FIGURES_OUTPUT_DIR,
            verbose=True,  # so you see pages + image counts
        )
        print(
            f"  -> Pages: {ingested.num_pages}, "
            f"Figures extracted: {len(ingested.figures)}\n"
        )

    print("All PDFs processed. New figures are in:", FIGURES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
