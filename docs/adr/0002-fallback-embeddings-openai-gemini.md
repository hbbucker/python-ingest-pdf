---
status: accepted
date: 2026-05-02
---

# ADR-0002: Estratégia de Fallback para Embeddings (`getAIembeddings`)

## Contexto e Problema

O pipeline RAG usa embeddings em dois momentos críticos:

1. **Ingestão** (`ingest.py`): cada chunk do PDF é convertido em vetor e armazenado no pgvector
2. **Busca** (`search.py`): a query do usuário é convertida em vetor para busca por similaridade coseno

Ambos os momentos usam a mesma coleção no pgvector (`PG_VECTOR_COLLECTION_NAME`). Isso impõe uma restrição rígida: **o modelo de embedding deve ser o mesmo em toda a sessão** — e idealmente o mesmo em todas as sessões que escrevem na mesma coleção. Trocar de modelo de embedding mid-session ou entre ingestão e busca invalida a busca semântica, pois vetores de modelos distintos habitam espaços de dimensão e semântica incompatíveis.

As chaves disponíveis são *free tier*:
- **Google AI Studio** (`gemini-embedding-2-preview`): dimensão configurável (default 3072), quota free tier superior para embeddings
- **OpenAI** (`text-embedding-3-small`): dimensão 1536, ~150 RPM no free tier

A questão é: **como garantir disponibilidade do embedding ao atingir rate limit, sem comprometer a consistência vetorial da coleção?**

## Drivers de Decisão

- **Consistência vetorial obrigatória:** todos os vetores de uma coleção devem usar o mesmo modelo e dimensão
- Validar a disponibilidade do provider **antes** de qualquer escrita no banco
- Não usar fallback reativo mid-stream (que permitiria vetores de modelos mistos no store)
- **Alinhamento com o provider primário do LLM:** `getLLM` usa Google como primário; manter o mesmo provider padrão simplifica a gestão de chaves e o raciocínio sobre o sistema

## Opções Consideradas

1. **Smoke test na inicialização com Google como primário e OpenAI como fallback** *(escolhida)*
2. **`Runnable.with_fallbacks()` como no LLM (fallback reativo)**
3. **Pré-checagem da presença da chave antes de instanciar o modelo**

## Decisão

Usar Google (`gemini-embedding-2-preview`) como provider primário e OpenAI (`text-embedding-3-small`) como fallback, selecionado de forma **eager** via smoke test na inicialização da sessão.

```python
def getAIembeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_MODEL_NAME)
        embeddings.embed_query("test")   # smoke test: valida chave e disponibilidade
        return embeddings
    except Exception:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY failed and OPENAI_API_KEY is not set")
        return OpenAIEmbeddings(model=OPENAI_MODEL_NAME)
```

**Por que Google como primário (alinhado com `getLLM`):** Ambas as funções — `getLLM` e `getAIembeddings` — usam agora o mesmo provider de preferência (Google). Isso simplifica a gestão de chaves: um ambiente com apenas `GOOGLE_API_KEY` configurada funciona integralmente sem fallback em nenhuma das duas funções. Reduz também a carga cognitiva ao raciocinar sobre o comportamento padrão do sistema.

**Por que smoke test e não fallback reativo:** Embeddings devem ser uniformes em toda a sessão. Um fallback reativo via `with_fallbacks()` poderia acionar o provider alternativo no meio da ingestão (ex.: após o chunk 200 de 500), resultando em vetores de modelos diferentes na mesma coleção. O smoke test força a decisão de provider **uma única vez**, antes de qualquer escrita ou leitura do banco.

**Por que `except Exception` e não exceções específicas:** O SDK do Google GenAI pode lançar `google.api_core.exceptions.ResourceExhausted` (rate limit), `google.auth.exceptions.DefaultCredentialsError` (auth) e outros. Capturar `Exception` garante que qualquer falha de inicialização — independentemente da causa — aciona o fallback. A granularidade do tipo de erro é desnecessária aqui porque a decisão é binária: o provider está disponível nesta sessão ou não.

**Por que a checagem de `OPENAI_API_KEY` após a falha:** O guard garante uma mensagem de erro clara quando nenhum provider está disponível, em vez de lançar uma exceção genérica do SDK OpenAI com mensagem menos informativa.

## Consequências

### Positivas
- Garantia de consistência vetorial: o provider é fixado no início da sessão e não muda
- Detecção antecipada de falha: o pipeline falha imediatamente se nenhum provider funciona, antes de qualquer operação custosa
- **Alinhamento com `getLLM`:** Google como primário em ambas as funções — um único `GOOGLE_API_KEY` é suficiente para operar o sistema sem fallback

### Negativas / Limitações
- **Latência de startup:** o smoke test adiciona uma chamada de API real na inicialização (~200–500 ms de latência de rede, custo mínimo em tokens).
- **Incompatibilidade de dimensão no fallback:** `text-embedding-3-small` tem dimensão 1536; `gemini-embedding-2-preview` tem dimensão 3072 por default. Se o fallback OpenAI for acionado em uma coleção criada com vetores Google, a inserção falhará com erro de dimensão no pgvector. A coleção deve ser recriada com a dimensão 1536 antes de usar OpenAI como provider ativo.
- **Sem fallback mid-session:** se o provider escolhido atingir rate limit *durante* a ingestão, a sessão falha. Não há recuperação automática — o design deliberadamente prioriza consistência sobre disponibilidade contínua.
- **`except Exception` pode mascarar erros de programação:** uma exceção não relacionada a disponibilidade (ex.: bug no código) também acionaria o fallback silenciosamente durante o desenvolvimento.

## Histórico de decisão

A versão original desta ADR usava OpenAI como primário e Google como fallback. A inversão foi feita para alinhar `getAIembeddings` com a ordem de preferência de `getLLM` (ADR-0001), onde Google é o provider primário. A coleção pgvector deve ser recriada para refletir a dimensão do modelo Google (3072) após esta mudança.

## Prós e Contras das Opções

### Opção 1 — Smoke test com Google primário (escolhida)
- **Pro:** Garante consistência vetorial — provider fixado antes de qualquer operação no banco
- **Pro:** Alinhado com `getLLM` — um único provider (Google) cobre o sistema inteiro
- **Contra:** Latência adicional de startup (~1 chamada de API)
- **Contra:** `except Exception` pode mascarar erros de programação durante o desenvolvimento

### Opção 2 — `with_fallbacks()` reativo (como no LLM)
- **Pro:** Sem custo de startup; integra com o protocolo `Runnable`
- **Contra:** **Inviável para embeddings:** fallback mid-session produziria vetores de modelos distintos na mesma coleção, corrompendo a busca semântica por similaridade coseno
- **Contra:** Não detecta falha antes do início da ingestão

### Opção 3 — Pré-checagem da presença da chave
- **Pro:** Sem latência de rede; determinístico
- **Contra:** A presença da variável de ambiente não garante que a chave é válida ou que o serviço está dentro do rate limit — uma `GOOGLE_API_KEY` expirada passaria na checagem e falharia na primeira chamada real
