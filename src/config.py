import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

logger = logging.getLogger(__name__)


class _LLMLogCallback(BaseCallbackHandler):
    def __init__(self, label: str) -> None:
        self.label = label

    def on_chat_model_start(self, serialized, messages, **kwargs):
        logger.info("[LLM] chamando modelo: %s", self.label)

    def on_llm_end(self, response: LLMResult, **kwargs):
        logger.info("[LLM] resposta recebida de: %s", self.label)

    def on_llm_error(self, error: BaseException, **kwargs):
        logger.warning("[LLM] erro em %s [%s]: %s", self.label, type(error).__name__, error)

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

DB_CONNECTION = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def getLLM():
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        raise RuntimeError("No LLM API key set (OPENAI_API_KEY or GOOGLE_API_KEY required)")

    primary = ChatGoogleGenerativeAI(
        model=GOOGLE_CHAT_MODEL,
        callbacks=[_LLMLogCallback(GOOGLE_CHAT_MODEL)],
    )

    if not OPENAI_API_KEY:
        logger.info("[LLM] configurado: primário=%s (sem fallback)", GOOGLE_CHAT_MODEL)
        return primary

    fallback = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        max_retries=0,
        callbacks=[_LLMLogCallback(OPENAI_CHAT_MODEL)],
    )
    logger.info("[LLM] configurado: primário=%s  fallback=%s", GOOGLE_CHAT_MODEL, OPENAI_CHAT_MODEL)
    return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))

def getAIembeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_MODEL_NAME)
        embeddings.embed_query("test")
        return embeddings
    except Exception:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY failed and OPENAI_API_KEY is not set")
        return OpenAIEmbeddings(model=OPENAI_MODEL_NAME)

def getStore(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )


def formatLLMError(e: Exception) -> str:
    module = type(e).__module__ or ""
    if "google" in module:
        llm = f"Google Gemini ({GOOGLE_CHAT_MODEL})"
        reason = str(e).split("{")[0].strip().rstrip(".")
    elif "openai" in module:
        llm = f"OpenAI ({OPENAI_CHAT_MODEL})"
        reason = str(e).split("\n")[0]
    else:
        llm = type(e).__name__
        reason = str(e).split("\n")[0][:120]
    return f"{llm} — {reason}"