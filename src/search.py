import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import getLLM, getAIembeddings, getStore

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt():
    retriever = getStore(getAIembeddings()).as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["contexto", "pergunta"])
    return (
        {"contexto": retriever | _format_docs, "pergunta": RunnablePassthrough()}
        | prompt
        | getLLM()
        | StrOutputParser()
    )

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


