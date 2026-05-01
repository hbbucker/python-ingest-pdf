# Desafio MBA Engenharia de Software com IA - Full Cycle

## Visão Geral

Esta aplicação implementa um pipeline de **RAG (Retrieval-Augmented Generation)** sobre um documento PDF. O objetivo é permitir que o usuário faça perguntas em linguagem natural sobre o conteúdo do PDF e receba respostas fundamentadas **exclusivamente** no texto do documento — sem alucinações ou conhecimento externo.

O fluxo é dividido em duas etapas:

1. **Ingestão** (`ingest.py`): lê o PDF, divide em chunks, gera embeddings e armazena no PostgreSQL com a extensão `pgvector`.
2. **Chat** (`chat.py`): recebe perguntas do usuário, busca os trechos mais relevantes no banco e envia contexto + pergunta para um LLM gerar a resposta.

```
PDF → chunks → embeddings → pgvector
                                ↑
         pergunta → busca semântica → LLM → resposta
```

Provedores suportados (com fallback automático):
- **Embeddings**: OpenAI (`text-embedding-3-small`) ou Google Gemini (`gemini-embedding-2-preview`)
- **LLM**: Google Gemini (`gemini-2.5-flash-lite`) com fallback para OpenAI (`gpt-4o-mini`)

---

## Pré-requisitos

- Docker e Docker Compose
- Python 3.11+
- Chave de API da OpenAI e/ou Google Gemini

---

## 1. Subindo o banco de dados

O `docker-compose.yml` sobe um PostgreSQL 17 com a extensão `pgvector` já habilitada.

```bash
docker compose up -d
```

Aguarde até o container `bootstrap_vector_ext` concluir (ele habilita a extensão `vector` no banco). Verifique:

```bash
docker compose ps
```

Todos os serviços devem estar com status `running` (postgres) ou `exited 0` (bootstrap).

Para encerrar:

```bash
docker compose down
```

Para destruir os dados também:

```bash
docker compose down -v
```

---

## 2. Configurando o ambiente

Copie o arquivo de exemplo e preencha as chaves:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# Pelo menos uma das chaves abaixo é obrigatória
GOOGLE_API_KEY=sua-chave-google
OPENAI_API_KEY=sua-chave-openai

# Modelos (os valores abaixo são os padrões)
GOOGLE_EMBEDDING_MODEL=gemini-embedding-2-preview
GOOGLE_GEMINI_DEFAULT_MODEL=gemini-2.5-flash-lite
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Banco de dados
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=minha_colecao

# PDF a ser ingerido
PDF_PATH=document.pdf
PDF_CHUNCKS=1000
PDF_OVERLAP=150
```

> `DATABASE_URL` usa `localhost` quando os scripts rodam fora do Docker. O valor `host.docker.internal` no `.env.example` é para execução dentro de containers.

---

## 3. Instalando as dependências Python

```bash
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
pip install -U langchain langchain-openai langchain-google-genai langchain-community \
               langchain-anthropic langchain-text-splitters langchain-postgres \
               psycopg python-dotenv beautifulsoup4 pypdf
```

---

## 4. Ingestão do PDF

Coloque o arquivo PDF na raiz do projeto (ou ajuste `PDF_PATH` no `.env`), depois execute a partir da pasta `src/`:

```bash
cd src
python ingest.py
```

O script irá:
1. Carregar o PDF definido em `PDF_PATH`
2. Dividir em chunks de `PDF_CHUNCKS` caracteres com overlap de `PDF_OVERLAP`
3. Gerar embeddings via OpenAI ou Google (com fallback automático)
4. Inserir os vetores na tabela `pgvector` no banco

Saída esperada:

```
Ingesting document.pdf...
Done!
```

> A ingestão é idempotente por ID de documento. Rodar novamente com o mesmo PDF sobrescreve os chunks existentes.

---

## 5. Executando o chat

```bash
cd src
python chat.py
```

Inicia um loop interativo no terminal:

```
Inicializando chat...
Pronto! Digite sua pergunta ou 'sair'/'exit' para encerrar.

Você: O que o documento fala sobre arquitetura de microsserviços?

Assistente: ...resposta baseada no documento...

Você: sair
Encerrando chat.
```

O assistente responde **somente** com base no conteúdo do PDF. Para perguntas sem resposta no documento, retorna:

> "Não tenho informações necessárias para responder sua pergunta."

---

## Estrutura do projeto

```
.
├── docker-compose.yml       # PostgreSQL + pgvector
├── .env.example             # Variáveis de ambiente (template)
├── document.pdf             # PDF a ser ingerido
├── requirements.txt
└── src/
    ├── config.py            # Carrega .env, inicializa provedores e store
    ├── ingest.py            # Pipeline de ingestão do PDF
    ├── search.py            # Chain de busca semântica + prompt RAG
    └── chat.py              # Interface de chat interativo no terminal
```
