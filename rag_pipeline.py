import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL ="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOP_K = 3
FALLBACK_ANSWER = "I'm sorry, but the provided documentation does not contain that information."


PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Upwork API Consultant.

You must answer ONLY using the provided documentation context.

If the answer is not contained in the context, reply exactly:

{fallback_answer}

Context:
{context}

Question:
{question}

Answer:"""
)
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

VECTOR_STORE = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=EMBEDDINGS,
)


def get_embeddings():
    """Create the same embedding model for ingestion and retrieval."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store():
    """Open the local Chroma database created by ingest.py."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def get_llm():
    """Connect to a hosted OpenAI-compatible Llama endpoint."""
    api_key = os.getenv("LLAMA_API_KEY")
    base_url = os.getenv("LLAMA_API_BASE_URL")

    print("API KEY FOUND:", bool(api_key))
    print("BASE URL:", base_url)

    if not api_key:
        raise ValueError("LLAMA_API_KEY is missing. Add it to your .env file.")
    if not base_url:
        raise ValueError("LLAMA_API_BASE_URL is missing. Add it to your .env file.")

    return ChatOpenAI(
        model=LLAMA_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=60,
    )


def format_docs(docs):
    """Turn retrieved documents into plain text for the prompt."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def answer_question(question):
    """
    Retrieve the top matching chunks and ask the Llama model to answer from them.

    No previous chat messages are passed here, so the model has no chat memory.
    """
    vector_store = VECTOR_STORE
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "context": format_docs(docs),
            "question": question,
            "fallback_answer": FALLBACK_ANSWER,
        }
    ).strip()

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}
