---
status: accepted
date: 2026-05-02
---

# ADR-0001: Estratégia de Fallback para o LLM de Chat (`getLLM`)

## Contexto e Problema

O pipeline RAG precisa de um LLM para gerar respostas a partir do contexto recuperado do pgvector. As chaves de API disponíveis são do nível gratuito (*free tier*):

- **Google AI Studio** (Gemini): 15 RPM e 1 000 000 TPM para `gemini-2.5-flash-lite`
- **OpenAI** (free tier): ~3 RPM para novos projetos com `gpt-4o-mini`

Com apenas um provider, o sistema para completamente ao atingir o rate limit durante sessões de uso contínuo. A questão é: **como garantir disponibilidade do LLM sob esse constraint sem introduzir retry loops que aumentem a latência percebida?**

## Drivers de Decisão

- Minimizar downtime causado por rate limit de API gratuita
- Não aumentar latência com estratégias de retry no provider primário
- Ser compatível com o protocolo LCEL (streaming, async, batch) do LangChain sem duplicar código na chain downstream
- Manter a configuração lazy (sem chamada de API na importação do módulo)

## Opções Consideradas

1. **`Runnable.with_fallbacks()` do LangChain com `max_retries=0` no fallback** *(escolhida)*
2. **Try-except manual em torno do `.invoke()`**
3. **Retry com backoff exponencial no provider primário**

## Decisão

Usar Google Gemini (`gemini-2.5-flash-lite`) como provider primário e OpenAI (`gpt-4o-mini`) como fallback, declarado via `Runnable.with_fallbacks()` do LangChain com `max_retries=0` no fallback.

```python
model = ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL)
return model.with_fallbacks([ChatOpenAI(model=OPENAI_CHAT_MODEL, max_retries=0)])
```

**Por que Gemini como primário:** O Google AI Studio free tier oferece limites substancialmente mais altos para modelos Gemini Flash do que o OpenAI free tier, tornando Gemini a escolha natural para absorver a maior parte da carga.

**Por que `with_fallbacks()` e não try-except manual:** O método `with_fallbacks()` opera ao nível do protocolo `Runnable` do LangChain. Isso significa que o fallback funciona de forma transparente para `.invoke()`, `.stream()`, `.ainvoke()` e `.batch()` — sem necessidade de alterar nenhum código na chain downstream (`search.py`). Um try-except manual exigiria replicar a lógica para cada modo de invocação.

**Por que `max_retries=0` no fallback:** O `ChatOpenAI` do LangChain por padrão tenta retries internos antes de lançar a exceção. No contexto de fallback, o primário já falhou; forçar retries no fallback adiciona latência sem ganho — o objetivo é rerouting imediato para o segundo provider.

**Por que o guard de chaves ocorre na chamada e não na importação:** A validação `if not OPENAI_API_KEY and not GOOGLE_API_KEY` está dentro da função `getLLM()`, mantendo a configuração lazy. Isso evita falhas em importações em contextos onde a variável de ambiente é definida após a carga do módulo (ex.: testes unitários, containers com secrets tardios).

## Consequências

### Positivas
- Zero mudança necessária na chain do `search.py` — o fallback é transparente
- Compatibilidade total com streaming e async, sem lógica adicional
- Eliminação de retry loops no primário que aumentariam a latência percebida

### Negativas / Limitações
- **Sem discriminação de erro:** `with_fallbacks()` aciona o fallback para qualquer exceção do primário — incluindo erros não relacionados a rate limit (ex.: erro de rede transitório, payload inválido). Não há como configurar, nesta versão, quais tipos de exceção acionam o fallback sem usar `exceptions_to_handle`.
- **Ausência de observabilidade:** Não há log explícito quando o fallback é acionado. Em produção, seria necessário instrumentar via LangSmith ou um callback customizado para detectar a frequência de failover.
- **Sem persistência de estado entre sessões:** Se o Gemini atingiu o rate limit, a próxima sessão iniciará novamente com Gemini como primário — não há memória de estado de degradação.

## Prós e Contras das Opções

### Opção 1 — `with_fallbacks()` (escolhida)
- **Pro:** Transparente para o código da chain; suporta streaming e async nativamente
- **Pro:** Declarativo — a lógica de fallback é visível no ponto de criação do modelo
- **Contra:** Fallback em qualquer exceção (sem filtragem por tipo de erro, a menos que se use `exceptions_to_handle`)

### Opção 2 — Try-except manual
- **Pro:** Controle explícito sobre quais exceções acionam o fallback
- **Contra:** Requer reimplementação para cada modo de invocação (`invoke`, `stream`, `ainvoke`)
- **Contra:** Quebra o contrato do protocolo `Runnable` se não implementado completamente

### Opção 3 — Retry com backoff exponencial
- **Pro:** Pode recuperar falhas transitórias sem mudar de provider
- **Contra:** Incompatível com rate limit — o backoff aumenta a latência sem garantia de sucesso enquanto a janela de limite estiver ativa
- **Contra:** Penaliza a experiência do usuário com espera perceptível no chat interativo
