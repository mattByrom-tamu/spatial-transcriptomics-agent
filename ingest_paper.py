import os
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF

# page_number is the index of the page 
# text is raw text being extracted from a page 
@dataclass
class PageContent:
    page_number: int
    text: str

# page the image comes from
# index of the image on the page
# image path 
# Maybe should add more info if multiple papers will store to same folder? 
@dataclass
class FigureContent:
    page_number: int
    image_index: int
    image_path: str

# path of the paper being ingested
# total page count
# list of text objects for each page
# list of image metadata objects 
@dataclass
class IngestedPaper:
    source_path: str
    num_pages: int
    pages: List[PageContent]
    figures: List[FigureContent]

# function takes in PDF path, figure directory 
# function pulls data from PDF 
def ingest_paper(pdf_path: str, figures_output_dir: Optional[str] = None) -> IngestedPaper:
    if figures_output_dir is None:
        figures_output_dir = os.path.join(os.getcwd(), "extracted_figures")
    os.makedirs(figures_output_dir, exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    # list to hold accumilated data as its pulled from loop 
    pages: List[PageContent] = []
    figures: List[FigureContent] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Extract text
        text = page.get_text("text")
        pages.append(PageContent(page_number=page_idx, text=text))

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            image_filename = f"page{page_idx}_img{img_index}.png"
            image_path = os.path.join(figures_output_dir, image_filename)

            if pix.n > 4:  # handle CMYK images and converts to RBG for compatability 
                pix = fitz.Pixmap(fitz.csRGB, pix)

            pix.save(image_path)

            figures.append(
                FigureContent(
                    page_number=page_idx,
                    image_index=img_index,
                    image_path=image_path
                )
            )

            pix = None

    doc.close()

    return IngestedPaper(
        source_path=os.path.abspath(pdf_path),
        num_pages=len(pages),
        pages=pages,
        figures=figures,
    )


