from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import CHROMA_DIR, get_embeddings


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


def load_documents():
    """Load PDF documentation files from docs folder."""

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            "Create a docs folder and add Upwork API PDFs."
        )

    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF documentation files found in docs."
        )

    documents = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    """Split documents into chunks small enough for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)

def sanity_check(documents):
    """Verify documents loaded correctly."""

    combined_text = "\n".join(
        doc.page_content for doc in documents
    )

    print("\n===== SANITY CHECK =====")
    print(f"Total characters: {len(combined_text)}")

    print("\nSample text:")
    print(combined_text[:500])
    print("========================\n")


def main():
    documents = load_documents()

    sanity_check(documents)

    chunks = split_documents(documents)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
)

    print(f"Ingested {len(documents)} files into {len(chunks)} chunks.")
    print(f"Chroma database saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
