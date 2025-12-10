# section_summarizer.py
# https://docs.python.org/3/library/re.html
# https://docs.python.org/3/library/uuid.html
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from agents import Agent, Runner  # your existing Agents SDK
from ingest_paper import IngestedPaper

# ---------- Data classes ----------

# Contains a chunk of text from one logical section 
@dataclass
class SectionChunk:
    section_name: str
    chunk_index: int
    text: str

# contains a merged summary of an entire section
@dataclass
class SectionSummary:
    section_name: str
    summary_text: str
    key_points: List[str]

# High lvl overview of entire paper
@dataclass
class PaperSummary:
    source_path: str
    sections: List[SectionSummary]

# RAG chunk, text that's easily retrieveable 
@dataclass
class RAGDocument:
    """
    A single text unit suitable for RAG.
    Later, you'll store:
      - text      -> TEXT column
      - metadata  -> JSONB column
      - embedding -> vector (OpenAI embedding model)
    """
    doc_id: str
    text: str
    metadata: Dict[str, str]


# ---------- Section detection ----------
# need help from Dr. Wong on these section paterns !!! 
# section headings expected in the paper (verbatim)
SECTION_PATTERNS = [
    r"ABSTRACT",
    r"INTRODUCTION",
    r"BACKGROUND",
    r"METHODS",
    r"MATERIALS AND METHODS",
    r"RESULTS",
    r"DISCUSSION",
    r"CONCLUSION",
    r"CONCLUSIONS",
    r"SUPPLEMENTARY",
]

# (?m) = multiline mode so ^ and $ work per line
# returns exact position of words in SECTION_PATTERNS 
SECTION_REGEX = re.compile(
    r"(?m)^(?P<header>" + "|".join(SECTION_PATTERNS) + r")\s*$"
)


def split_into_sections(full_text: str) -> List[Tuple[str, str]]:
    """
    Split the paper into rough sections based on headings.

    Returns a list of (section_name, section_text) tuples.
    If no headings are found, returns a single ('FULL_TEXT', full_text).
    """
    matches = list(SECTION_REGEX.finditer(full_text))

    if not matches:
        return [("Full_Text", full_text)]

    sections: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        header = m.group("header").strip()
        # normalize capitalization
        section_name = header.title()

        start = m.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(full_text)

        section_text = full_text[start:end].strip()
        if not section_text:
            continue

        sections.append((section_name, section_text))

    return sections


# ---------- Chunking ----------

def chunk_section(
    section_name: str,
    section_text: str,
    max_chars: int = 4000,
    overlap_chars: int = 500,
) -> List[SectionChunk]:
    """
    Split a long section into overlapping chunks for LLM + RAG.
    """
    chunks: List[SectionChunk] = []
    start = 0
    chunk_index = 0
    text_len = len(section_text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk_text = section_text[start:end].strip()
        if not chunk_text:
            break

        chunks.append(
            SectionChunk(
                section_name=section_name,
                chunk_index=chunk_index,
                text=chunk_text,
            )
        )

        if end == text_len:
            break

        # move forward with overlap so context isn't cut harshly
        start = end - overlap_chars
        chunk_index += 1

    return chunks


# ---------- Main summarizer + RAG prep ----------

def summarize_paper_sections(
    agent: Agent,
    ingested: IngestedPaper,
    paper_id: Optional[str] = None,
) -> Tuple[PaperSummary, List[RAGDocument]]:
    
    """
    - Combines pages into full text
    - Splits into sections
    - Further chunks sections
    - Summarizes each chunk with the LLM agent
    - Builds RAG-ready documents from raw chunks
    """
    
    if paper_id is None:
        # Later you'll replace this with a DB-generated id
        paper_id = str(uuid.uuid4())

    # 1) Combine page texts into a single string
    full_text = "\n\n".join(page.text for page in ingested.pages)

    # 2) Rough section detection
    rough_sections = split_into_sections(full_text)

    # 3) Chunk each section
    all_chunks: List[SectionChunk] = []
    for section_name, section_text in rough_sections:
        section_chunks = chunk_section(section_name, section_text)
        all_chunks.extend(section_chunks)

    # will give an idea of how many chunks being made
    print(f"[DEBUG] Total text chunks for summarization: {len(all_chunks)}")
    # 4) Summarize chunks + create RAG docs
    rag_docs: List[RAGDocument] = []
    per_section_summaries: Dict[str, List[str]] = {}

    for chunk in all_chunks:
        # RAG doc from raw chunk text
        doc_id = f"{paper_id}:{chunk.section_name}:{chunk.chunk_index}"

        rag_docs.append(
            RAGDocument(
                doc_id=doc_id,
                text=chunk.text,
                metadata={
                    "paper_id": paper_id,
                    "source_path": ingested.source_path,
                    "section_name": chunk.section_name,
                    "chunk_index": str(chunk.chunk_index),
                },
            )
        )

        # Summarize this chunk with your Agent, Can get Dr. Wongs input on this 
        prompt = f"""
You are a spatial transcriptomics research assistant.

You will be given a chunk of text from the **{chunk.section_name}** section
of a scientific paper. Summarize the key points with emphasis on:

- The main purpose or message of this chunk.
- Any spatial transcriptomics methods or platforms mentioned.
- Any performance comparisons or benchmarks, if present.

Return your answer as a short natural-language summary.
        
Chunk text:
\"\"\"
{chunk.text}
\"\"\"
"""

        result = Runner.run_sync(agent, prompt)
        chunk_summary_text = result.final_output.strip()

        per_section_summaries.setdefault(chunk.section_name, []).append(
            chunk_summary_text
        )

    # 5) Merge chunk summaries into per-section summaries
    section_summaries: List[SectionSummary] = []
    for section_name, pieces in per_section_summaries.items():
        combined_summary = "\n\n".join(pieces)
        section_summaries.append(
            SectionSummary(
                section_name=section_name,
                summary_text=combined_summary,
                key_points=[],  # can parse bullets later if needed
            )
        )

    paper_summary = PaperSummary(
        source_path= ingested.source_path,
        sections= section_summaries,
    )

    return paper_summary, rag_docs
