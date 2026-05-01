import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

CURRENT_DIR = Path(__file__).parent.parent
PDF_PATH = CURRENT_DIR / os.getenv("PDF_PATH", "")
PDF_CHUNKS = int(os.getenv("PDF_CHUNKS", 1000))
PDF_OVERLAP = int(os.getenv("PDF_OVERLAP", 150))

OPENAI_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

CLAUDE_API_KEY=os.getenv("CLAUDE_API_KEY")
CLAUDE_EMBEDDING_MODEL=os.getenv("CLAUDE_EMBEDDING_MODEL", "claude-2")


DB_CONNECTION = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def getLLM():
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        raise RuntimeError("No LLM API key set (OPENAI_API_KEY or GOOGLE_API_KEY required)")

    model = ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL)
    return model.with_fallbacks([ChatOpenAI(model=OPENAI_CHAT_MODEL, max_retries=0)])

def getAIembeddings():
    try:
        embeddings = OpenAIEmbeddings(model=OPENAI_MODEL_NAME)
        embeddings.embed_query("test")
        return embeddings
    except Exception:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY failed and GOOGLE_API_KEY is not set")
        return GoogleGenerativeAIEmbeddings(model=GOOGLE_MODEL_NAME)

def getStore(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )