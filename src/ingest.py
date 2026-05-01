from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

from config import (
    PDF_PATH,
    PDF_CHUNKS,
    PDF_OVERLAP,
    getStore, getAIembeddings
)

if not PDF_PATH.exists():
    raise RuntimeError(f"PDF file {PDF_PATH} not found")

def ingest_pdf():
    print(f"Ingesting {PDF_PATH}...")
    enriched = enrichedDocument(splitDocument())
    ids = [f"doc-{idx}" for idx in range(len(enriched))]
    insertIntoDb(enriched, ids, getAIembeddings())
    print("Done!")

def enrichedDocument(split):
    return [
        Document(
            page_content=doc.page_content,
            metadata={key: value for key, value in doc.metadata.items() if key != "source"},
        )
        for doc in split
    ]

def splitDocument():
    documents = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNKS, chunk_overlap=PDF_OVERLAP, add_start_index=False
    )
    split = splitter.split_documents(documents)
    if not split:
        raise RuntimeError(f"Error on get split documents from pdf {PDF_PATH}")
    return split

def insertIntoDb(enriched, ids, embeddings):
    vectorstore = getStore(embeddings)
    vectorstore.add_documents(enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()
